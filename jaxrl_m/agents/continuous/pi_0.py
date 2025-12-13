# pi05_policy_minimal.py

import functools
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp

import flax.nnx as nnx
import flax.linen as nn
import flax.traverse_util as traverse_util

# ---- PA-RL interface ----
from jaxrl_m.utils.timer_utils import Timer
from .base_policy import BasePolicy

# ---- Use the real training functions directly ----
from jaxrl_m.utils.pi0_train_utils import (
    init_train_state,
    init_train_state_for_target,
    train_step,
)
import openpi.training.checkpoints as _checkpoints
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.transforms as _transforms
from openpi.training.config import TrainConfig, DataConfig
import openpi.models.model as _model
from openpi.training.data_loader import DataLoader
from openpi.models.pi0 import make_attn_mask
import openpi.shared.array_typing as at
from typing import Callable


from jaxrl_m.common.typing import Batch, Data, PRNGKey
import logging
import numpy as np


class Dummy_Dataloader(DataLoader):
    def __init__(self, data_config: DataConfig):
        self._data_config = data_config
    
    def data_config(self) -> DataConfig:
        return self._data_config
 

class PiPolicy(BasePolicy):
    """
    Minimal wrapper that integrates Pi0.5 into PA-RL without
    rewriting OpenPI internals.
    """

    def __init__(  self,
        rng: Optional[PRNGKey] = None,
        # example arrays for model init
        config: TrainConfig | None = None,
        is_target: bool = False,
        **kwargs,
        ):
        # NOTE: Pi0.5 initialization does NOT depend on observations or actions.
        if config is None:
            raise ValueError("Pi-0.5 Policy requires config=<OpenPI TrainConfig>") 

        self.config = config

        # Get action_horizon from config
        self.action_horizon = config.model.action_horizon

        # RNG split
        if rng is None:
            rng = jax.random.key(config.seed)
        self.train_rng, init_rng = jax.random.split(rng)

        # Create mesh and sharding
        # breakpoint()
        self.mesh = sharding.make_mesh(config.fsdp_devices)
        self.replicated_sharding = jax.sharding.NamedSharding(
            self.mesh, jax.sharding.PartitionSpec()
        )
        self.data_sharding: Optional[jax.sharding.Sharding] = None

        checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
            self.config.checkpoint_dir,
            keep_period=self.config.keep_period,
            overwrite=self.config.overwrite,
            resume=self.config.resume,
        )

        # Initialize model + sharding
        if is_target:
            self.train_state, self.train_state_sharding = init_train_state_for_target(
                self.config, init_rng, self.mesh, resume=resuming
            )
        else:
            self.train_state, self.train_state_sharding = init_train_state(
            self.config, init_rng, self.mesh, resume=resuming
        )

        # Print trainable params
        self.print_trainable_params()

        # If actually resuming, load checkpoint
        if resuming:
            self.train_state = _checkpoints.restore_state(
                checkpoint_manager, self.train_state, None
            )

        # Save the manager for later saves
        self.checkpoint_manager = checkpoint_manager

        # Pre-compile train step and inference step
        self._ptrain_step = jax.jit(
            functools.partial(train_step, config),
            in_shardings=(
                self.replicated_sharding,
                self.train_state_sharding,
                self.replicated_sharding,
            ),
            out_shardings=(
                self.train_state_sharding,
                self.replicated_sharding,
            ),
            donate_argnums=(1,),
        )
        self.model_def = self.train_state.model_def  # static graphdef
        self.params_sharding = self.train_state_sharding.params 
        # def _infer(params, observation, rng):
        #     model = nnx.merge(self.model_def, params)
        #     model.eval()
        #     return model.sample_actions(rng, observation)

        # self._pinfer = jax.jit(
        #     _infer,
        #     in_shardings=(
        #         self.params_sharding,   # state
        #         None,    # observation
        #         self.replicated_sharding,    # rng
        #     ),
        #     out_shardings=self.replicated_sharding,
        # )


        # Data configs, setup and utils
        self.data_config = config.data.create(config.assets_dirs, config.model)
        logging.info(f"data_config: {self.data_config}")
        self.data_norm_stats = self.data_config.norm_stats
        self.input_data_transforms = [*self.data_config.repack_transforms.inputs,
            *self.data_config.data_transforms.inputs,
            _transforms.Normalize(self.data_norm_stats, use_quantiles=self.data_config.use_quantile_norm),
            *self.data_config.model_transforms.inputs,]
        self.input_data_transforms = _transforms.compose(self.input_data_transforms)
        self.output_data_transforms = [
            *self.data_config.model_transforms.outputs,
            _transforms.Unnormalize(self.data_norm_stats, use_quantiles=self.data_config.use_quantile_norm),
            *self.data_config.data_transforms.outputs,
            *self.data_config.repack_transforms.outputs,
        ]
        self.output_data_transforms = _transforms.compose(self.output_data_transforms)

        self.data_loader_dummy = Dummy_Dataloader(self.data_config)
        self.output_data_transforms_without_unnorm = [
            *self.data_config.model_transforms.outputs,
            *self.data_config.data_transforms.outputs,
            *self.data_config.repack_transforms.outputs,
        ]
        self.output_data_transforms_without_unnorm = _transforms.compose(self.output_data_transforms_without_unnorm)
        self.unnormalize = _transforms.Unnormalize(self.data_norm_stats, use_quantiles=self.data_config.use_quantile_norm)
        self._infer_cache: dict[int, Callable] = {}
        self._vlm_cache: dict[int, Callable] = {}

    # Misc to print trainable params #
    def print_trainable_params(self):
        # Dump trainable parameter names to pickle file
        trainable_params = self.train_state.params.filter(self.config.trainable_filter)
        trainable_param_names = list(traverse_util.flatten_dict(trainable_params.to_pure_dict()).keys())
        trainable_param_names_str = ["/".join(map(str, key)) for key in trainable_param_names]
        
        # Calculate total number of trainable parameters
        trainable_params_flat = traverse_util.flatten_dict(trainable_params.to_pure_dict())
        total_trainable_params = sum(np.prod(p.shape) for p in trainable_params_flat.values())
        
        # Calculate total number of all model parameters (including frozen)
        all_params_flat = traverse_util.flatten_dict(self.train_state.params.to_pure_dict())
        total_model_params = sum(np.prod(p.shape) for p in all_params_flat.values())
        trainable_percentage = (total_trainable_params / total_model_params * 100) if total_model_params > 0 else 0
        
        # logs_dir = epath.Path("logs")
        # logs_dir.mkdir(parents=True, exist_ok=True)
        # trainable_params_file = logs_dir / f"trainable_{config.name}.pkl"
        
        # with open(trainable_params_file, "wb") as f:
        #     pickle.dump(trainable_param_names_str, f)
        
        # logging.info(f"Dumped {len(trainable_param_names_str)} trainable parameter names to {trainable_params_file}")
        logging.info(f"Total model parameters: {total_model_params:,}")
        logging.info(f"Total trainable parameters: {total_trainable_params:,} ({trainable_percentage:.2f}%)")
        logging.info(f"Total frozen parameters: {total_model_params - total_trainable_params:,}")
        logging.info(f"First 10 trainable params: {trainable_param_names_str[:10]}")

    # ------------------------------------------------------------------------------------
    # INFERENCE
    # ------------------------------------------------------------------------------------
    def sample_actions(self, _observations: Data | Batch, repeat=1, cache_dir=None, timer=None, argmax=False, 
                       processed_obs=False, normalized=False, return_obs= False, obs_key: str | None = None, infer=True,
                       params: Optional[at.Params] = None,
                        **kwargs):
        with sharding.set_mesh(self.mesh):
            if not processed_obs:
                # breakpoint()
                # if infer:
                #     observations = self.convert_to_openpi_format_infer(_observations)
                # else:
                #     observations = self.convert_to_openpi_format(_observations, obs_key=obs_key)
                # # breakpoint()
                # obs = self.input_data_transforms(observations)
                if infer:
                    observations = self.convert_to_openpi_format_infer(_observations, obs_key=obs_key)

                    # 2) Apply the same input transforms as sample_actions
                    obs = self.input_data_transforms(observations)

                    # 3) Wrap into OpenPI Observation (still NumPy here is fine;
                    #    preprocess_observation inside the JIT will turn them into jax.Arrays)
                    # obs_ = _model.Observation.from_dict(obs)

                    # 4) Key JIT cache by batch size (same pattern as _infer_cache)
                    #    choose any reliable field to read batch size from:
                    # batch_size = obs_.tokenized_prompt.shape[0]
                    batch_size = obs['state'].shape[0]
                else:
                    observations = self.convert_to_openpi_format(_observations, obs_key=obs_key)

                    obs = (_model.Observation.from_dict(observations), observations["actions"])

                    batch_size = observations['actions'].shape[0]


            else:
                obs = _observations

            if repeat > 1:
                obs = repeat_tree(obs, repeat)
                obs = flatten_repeat(obs)  
            outputs = {
                "state": obs["state"],
                }            

            # breakpoint()
            seed = kwargs.pop("seed")
            
            # Split seed once more #
            seed, rng = jax.random.split(seed)

            obs_ = _model.Observation.from_dict(obs)
            if batch_size not in self._infer_cache:
                # First time seeing this batch size → compile JIT
                self._infer_cache[batch_size] = self._build_infer_jit()

            infer_fn = self._infer_cache[batch_size]
            actions = infer_fn(params if params is not None else self.train_state.params, obs_, rng)
            # actions = self._pinfer(self.train_state.params, obs, seed)

            outputs["actions"] = actions
            # print("actions shape: ", actions.shape)
            outputs = jax.tree.map(lambda x: np.asarray(x), outputs)
            if normalized:
                outputs = self.output_data_transforms_without_unnorm(outputs)
            else:
                outputs = self.output_data_transforms(outputs)
            if repeat > 1:
                outputs = unflatten_repeat(outputs,repeat)

            if outputs['actions'].shape[0] == 1 and repeat==1:
                outputs['actions'] = outputs['actions'][0]
            # print("final actions shape: ", outputs['actions'].shape)
            if return_obs:
                return outputs['actions'], obs
            return outputs['actions']

    # ------------------------------------------------------------------------------------
    # TRAINING (supervised updates)
    # ------------------------------------------------------------------------------------
    def update(self, batch: Batch, timer=None):
        with sharding.set_mesh(self.mesh):
            batch = self.convert_to_openpi_format(batch)
            batch = (_model.Observation.from_dict(batch), batch["actions"])
            # breakpoint()
            self.train_state, info = self._ptrain_step(
                self.train_rng, self.train_state, batch
            )
        return info
    
    # ------------------------------------------------------------------------------------
    # Get VLM output from the model #
    # ------------------------------------------------------------------------------------
    
    def _build_vlm_jit(self):
        """
        Build and JIT-compile a function that runs the VLM prefix and
        returns the KV cache, given an Observation.
        """

        def _vlm(params, obs: _model.Observation):
            # Reconstruct model from graphdef + params
            model = nnx.merge(self.model_def, params)
            model.eval()

            # Same preprocessing as in training / sample_actions
            observation = _model.preprocess_observation(
                None,  # no rng needed at inference
                obs,
                train=False,
            )

            # first fill KV cache with a forward pass of the prefix
            prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
            # jax.debug.breakpoint()
            prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1

            vlm_output, kv_cache = model.PaliGemma.llm(
                [prefix_tokens, None],
                mask=prefix_attn_mask,
                positions=positions,
            )

            return vlm_output, kv_cache

        compiled = jax.jit(
            _vlm,
            in_shardings=(
                self.params_sharding,      # params
                self.replicated_sharding,  # rng
            ),
            out_shardings=(self.replicated_sharding, self.replicated_sharding),
        )

        return compiled
    
    def get_vlm_output(
        self,
        rng,
        _observations: Data | Batch,
        infer=True,
        obs_key: str = None,
    ):
        with sharding.set_mesh(self.mesh):
            # 1) Convert your env obs → OpenPI format
            if infer:
                observations = self.convert_to_openpi_format_infer(_observations, obs_key=obs_key)

                # 2) Apply the same input transforms as sample_actions
                obs = self.input_data_transforms(observations)

                # 3) Wrap into OpenPI Observation (still NumPy here is fine;
                #    preprocess_observation inside the JIT will turn them into jax.Arrays)
                obs_ = _model.Observation.from_dict(obs)

                # 4) Key JIT cache by batch size (same pattern as _infer_cache)
                #    choose any reliable field to read batch size from:
                batch_size = obs_.tokenized_prompt.shape[0]
            else:
                observations = self.convert_to_openpi_format(_observations, obs_key=obs_key)

                obs_ = (_model.Observation.from_dict(observations), observations["actions"])

                batch_size = observations['actions'].shape[0]

            

            if batch_size not in self._vlm_cache:
                self._vlm_cache[batch_size] = self._build_vlm_jit()

            vlm_fn = self._vlm_cache[batch_size]

            # 5) Call the compiled function
            out1, kv_cache = vlm_fn(self.train_state.params, obs_)

            return out1, kv_cache

        

    def _build_infer_jit(self):
        """
        Build and JIT-compile an inference function specialized to `obs_example`'s batch size.
        """

        # This is the core inference function
        def _infer(params, observation, rng):
            model = nnx.merge(self.model_def, params)
            model.eval()
            return model.sample_actions(rng, observation)


        compiled = jax.jit(
            _infer,
            in_shardings=(
                self.params_sharding,      # params
                None, 
                self.replicated_sharding,  # rng
            ),
            out_shardings=self.replicated_sharding,
        )

        return compiled


    # ------------------------------------------------------------------------------------
    # CHECKPOINTING — directly use OpenPI
    # ------------------------------------------------------------------------------------
    def restore_checkpoint(self, path: str, sharding: jax.sharding.Sharding):
        """
        Restore Pi0.5 train_state from a checkpoint directory.
        Must use initialize_checkpoint_dir() not CheckpointManager().
        """
        ckpt_mgr, resuming = _checkpoints.initialize_checkpoint_dir(
            Path(path),
            keep_period=self.config.keep_period,
            overwrite=False,   # NEVER overwrite when restoring
            resume=True,        # Look for existing checkpoint
        )

        if not resuming:
            raise FileNotFoundError(f"No checkpoint found in {path}")

        self.checkpoint_manager = ckpt_mgr

        # Restore state using OpenPI's official function
        self.train_state = _checkpoints.restore_state(
            ckpt_mgr,
            self.train_state,
            None,   # no data_loader required for restore
        )
        return self

    def save_checkpoint(self, save_dir: str, step: int, keep: int = 10, overwrite=True):
        """
        Save Pi0.5 checkpoint.
        If save_dir matches the initialized checkpoint directory,
        reuse the stored checkpoint manager.
        """

        save_dir = Path(save_dir)
        cfg_dir = Path(self.config.checkpoint_dir)

        # Save to the actual Pi0.5 directory → reuse manager
        if save_dir == cfg_dir:
            ckpt_mgr = self.checkpoint_manager

        else:
            # Warn the user
            print(
                f"[WARNING] Pi0.5 save_checkpoint() called with a different directory.\n"
                f"  Expected: {cfg_dir}\n"
                f"  Given:    {save_dir}\n"
                f"Creating a NEW checkpoint manager for this external directory."
            )

            ckpt_mgr, _ = _checkpoints.initialize_checkpoint_dir(
                save_dir,
                keep_period=keep,
                overwrite=overwrite,
                resume=False,
            )

        # Use OpenPI's recommended save function
        _checkpoints.save_state(
            ckpt_mgr,
            self.train_state,
            self.data_loader_dummy,
            step,
        )

        self.checkpoint_manager = ckpt_mgr

    # ------------------------------------------------------------------------------------
    # MISC
    # ------------------------------------------------------------------------------------
    def prepare_for_finetuning(self):
        pass

    def prepare_for_inference(self):
        pass

    def clear_cache(self):
        pass

    def to_device(self, sharding):
        self.data_sharding = sharding
        return self

    
    def _apply_data_transforms(self, batch, training=False):
        """
        Apply self.data_transforms (which expect single-sample dicts)
        across a batched dictionary.
        """

        B = next(iter(batch.values())).shape[0]

        # Split into per-sample dicts (zero copy views)
        samples = [{k: v[i] for k, v in batch.items()} for i in range(B)]
        for s in samples:
            if "prompt" in s:
                p = s["prompt"]

                if isinstance(p, np.ndarray):
                    if p.dtype.type is np.bytes_ or (p.dtype == object and isinstance(p[0], bytes)):
                        s["prompt"] = np.array([x.decode("utf-8") for x in p], dtype=object)

                elif isinstance(p, bytes):
                    s["prompt"] = p.decode("utf-8")

            if "proprio" in s["observations"]:
                s["observations"]["state"] = s["observations"].pop("proprio")
            if "next_observations" in s and "proprio" in s["next_observations"]:
                s["next_observations"]["state"] = s["next_observations"].pop("proprio")

        # Apply your pre-defined CompositeTransform
        transformed = [self.data_transforms(s) for s in samples]

        # Rebatch
        out = {k: np.stack([t[k] for t in transformed], axis=0)
            for k in transformed[0].keys()}
        
        if training:
            return _model.Observation.from_dict(out), out["actions"]
        else:
            return _model.Observation.from_dict(out)
    
    def convert_to_openpi_format(self, out_new, obs_key: str = "observations"): # Assumes out_new to be a `Batch` type #
        """
        Converts your NEW nested output format into the OLD OpenPI-style format.
        No copies are made — only dict references.
        """

        old = {}

        # ------------------------------------------------
        # 1. State & Actions
        # ------------------------------------------------
        old["state"] = out_new[obs_key]["proprio"]
        old["actions"] = out_new["actions"]

        # ------------------------------------------------
        # 2. Images (map new → old)
        # ------------------------------------------------
        # new camera names → old camera names
        CAM_REVERSE = {
            "image": "base_0_rgb",
            "wrist_image": "left_wrist_0_rgb",
            "image_3": "right_wrist_0_rgb",
        }

        old["image"] = {
            oldname: out_new[obs_key][newname]
            for newname, oldname in CAM_REVERSE.items()
        }

        # ------------------------------------------------
        # 3. Image Masks
        # ------------------------------------------------
        old["image_mask"] = {
            oldname: out_new[obs_key + "_image_mask"][newname]
            for newname, oldname in CAM_REVERSE.items()
        }

        # ------------------------------------------------
        # 4. Token fields
        # ------------------------------------------------
        old["tokenized_prompt"]      = out_new["tokenized_prompt"]
        old["tokenized_prompt_mask"] = out_new["tokenized_prompt_mask"]
        if not "pi05" in self.config.name:
            old["token_ar_mask"]         = out_new["token_ar_mask"]
            old["token_loss_mask"]       = out_new["token_loss_mask"]
        return old
    
    def convert_to_openpi_format_infer(self, out_new, obs_key: str | None = None, *args, **kwargs): # Assumes out_new to be a `Data` type #
        """
        Converts your NEW nested output format into the OLD OpenPI-style format.
        No copies are made — only dict references.
        """
        if obs_key is not None:
            prompt = out_new["prompt_bytes"]
            prompt = batch_decode_prompt_bytes(prompt)
            out_new = out_new[obs_key]
            out_new["prompt"] = prompt

            state_dim = kwargs.get("state_dim", 8)
            out_new["proprio"] = out_new["proprio"][:, :state_dim]
            # breakpoint()
        
        old = {}

        # ------------------------------------------------
        # 1. State & Actions
        # ------------------------------------------------
        old["observation/state"] = self.ensure_batch(out_new["proprio"])

        # ------------------------------------------------
        # 2. Images (map new → old)
        # ------------------------------------------------
        # new camera names → old camera names
        old["observation/image"] = self.ensure_batch(out_new["image"])
        old["observation/wrist_image"] = self.ensure_batch(out_new["wrist_image"])
        
        old["prompt"] = self._ensure_prompt_batch(out_new["prompt"])
        return old
    
    def ensure_batch(self, x):
        """If x is 3D or 1D, add a batch dim."""
        if isinstance(x, dict):
            return {k: self.ensure_batch(v) for k, v in x.items()}

        arr = x
        ndim = arr.ndim

        # Image case: [H, W, C] → [1, H, W, C]
        if ndim == 3:
            return arr[None, ...]

        # Proprio/state: [S] → [1, S]
        if ndim == 1:
            return arr[None, :]

        # Already batched → leave as-is
        return arr
    
    def get_debug_metrics(self, batch, seed):
        batch = self.convert_to_openpi_format(batch)
        actions = self.sample_actions(observations=batch, processed_obs=True, seed=seed)
        B, H, D = actions.shape
        
        diff = actions - batch["actions"][:, :, : D]  # (B, H, D)
        metrics = {
            "mse": (diff ** 2).mean(),        # scalar
            "mae": jnp.abs(diff).mean(),      # scalar
        }

        return metrics

    def _ensure_prompt_batch(self, prompt):
        """
        Normalizes the prompt into a batched numpy array of byte strings:
            e.g. ['do X'] → array([b'do X'], dtype=object)
                (tuple) → array([...])
                array([...]) → returned as-is if batched
        """

        # ------------------------------------------------------------
        # Case 0: Nothing returned
        # ------------------------------------------------------------
        if prompt is None:
            return np.array([b""], dtype=object)

        # ------------------------------------------------------------
        # Case 1: Single string
        # ------------------------------------------------------------
        if isinstance(prompt, str):
            return np.array([prompt.encode("utf-8")], dtype=object)

        # ------------------------------------------------------------
        # Case 2: Single bytes
        # ------------------------------------------------------------
        if isinstance(prompt, bytes):
            return np.array([prompt], dtype=object)

        # ------------------------------------------------------------
        # Case 3: Prompt is a list or tuple (e.g. from vectorized env)
        # ------------------------------------------------------------
        if isinstance(prompt, (list, tuple)):
            out = []
            for p in prompt:
                if isinstance(p, str):
                    out.append(p.encode("utf-8"))
                elif isinstance(p, bytes):
                    out.append(p)
                else:
                    raise TypeError(f"Unsupported prompt element: {type(p)}")
            return np.array(out, dtype=object)

        # ------------------------------------------------------------
        # Case 4: Numpy array
        # ------------------------------------------------------------
        if isinstance(prompt, np.ndarray):
            # If it's unbatched and contains a single string
            if prompt.ndim == 0:
                p = prompt.item()
                return self._ensure_prompt_batch(p)

            # Already batched → ensure byte encoding
            out = []
            for p in prompt:
                if isinstance(p, str):
                    out.append(p.encode("utf-8"))
                elif isinstance(p, bytes):
                    out.append(p)
                else:
                    raise TypeError(f"Unsupported prompt element: {type(p)}")
            return np.array(out, dtype=object)

        # ------------------------------------------------------------
        # Anything else is unsupported
        # ------------------------------------------------------------
        raise TypeError(f"Unsupported prompt type: {type(prompt)}")
    
def repeat_tree(tree, repeat: int):
    """
    Repeat every array in a PyTree along a new repeat dimension.
    Input:  (B, ...)
    Output: (B, repeat, ...)
    Works for both np.ndarray and jnp.ndarray.
    Preserves the original array type.
    """
    def _repeat(x):
        if isinstance(x, (np.ndarray, jnp.ndarray)):
            # x: (B, ...)
            # x[:, None, ...] → (B, 1, ...)
            # tile/repeat along axis=1
            return x[:, None, ...].repeat(repeat, axis=1)
        return x

    return jax.tree.map(_repeat, tree)

def flatten_repeat(tree):
    """
    Convert a PyTree from (B, repeat, ...) → (B*repeat, ...)
    Works for both np.ndarray and jnp.ndarray.
    Preserves the original array type.
    """
    def _flatten(x):
        if isinstance(x, (np.ndarray, jnp.ndarray)):
            B, R = x.shape[:2]
            return x.reshape((B * R,) + x.shape[2:])
        return x

    return jax.tree.map(_flatten, tree)


def unflatten_repeat(tree, repeat: int):
    """
    Convert PyTree from (B*repeat, ...) → (B, repeat, ...)
    Works for both jnp.ndarray and np.ndarray.
    """
    def _unflatten(x):
        # Handle arrays from both JAX and NumPy
        if hasattr(x, "shape") and x.shape is not None and len(x.shape) >= 1:
            BR = x.shape[0]
            assert BR % repeat == 0, f"{BR} not divisible by repeat {repeat}"
            B = BR // repeat
            return x.reshape((B, repeat) + x.shape[1:])
        return x

    return jax.tree.map(_unflatten, tree)


def decode_prompt_bytes(prompt_bytes: np.ndarray) -> str:
    # prompt_bytes: (MAX_PROMPT_BYTES,) uint8
    # Drop trailing zeros
    non_zero = np.trim_zeros(prompt_bytes, 'b')  # 'b' trims both ends
    b = non_zero.tobytes()
    return b.decode("utf-8")

# For a batch:
def batch_decode_prompt_bytes(prompt_bytes_batch: np.ndarray) -> list[str]:
    return [decode_prompt_bytes(prompt_bytes_batch[idx, :]) for idx in range(prompt_bytes_batch.shape[0])]