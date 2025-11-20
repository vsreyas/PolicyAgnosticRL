# pi05_policy_minimal.py

import functools
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp

import flax.nnx as nnx
import flax.linen as nn

# ---- PA-RL interface ----
from jaxrl_m.utils.timer_utils import Timer
from .base_policy import BasePolicy

# ---- Use the real training functions directly ----
from jaxrl_m.utils.pi0_train_utils import (
    init_train_state,
    train_step,
)
import openpi.training.checkpoints as _checkpoints
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.transforms as _transforms
from openpi.training.config import TrainConfig
import openpi.models.model as _model

from jaxrl_m.common.typing import Batch, Data, PRNGKey
import logging
import numpy as np


class PiPolicy(BasePolicy):
    """
    Minimal wrapper that integrates Pi0.5 into PA-RL without
    rewriting OpenPI internals.
    """

    def __init__(  self,
        rng: Optional[PRNGKey] = None,
        # example arrays for model init
        config: TrainConfig | None = None,
        **kwargs,
        ):
        # NOTE: Pi0.5 initialization does NOT depend on observations or actions.
        if config is None:
            raise ValueError("Pi-0.5 Policy requires config=<OpenPI TrainConfig>") 

        self.config = config

        # RNG split
        if rng is None:
            rng = jax.random.key(config.seed)
        self.train_rng, init_rng = jax.random.split(rng)

        # Create mesh and sharding
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
        self.train_state, self.train_state_sharding = init_train_state(
            self.config, init_rng, self.mesh, resume=resuming
        )

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
                None,  # data sharding not needed if you pass replicated batch
            ),
            out_shardings=(
                self.train_state_sharding,
                self.replicated_sharding,
            ),
            donate_argnums=(1,),
        )

        # Simple forward-pass function for inference
        def _infer(state, observation):
            model = nnx.merge(state.model_def, state.params)
            model.eval()
            return model.sample_actions(observation)

        self._pinfer = jax.jit(
            _infer,
            in_shardings=(self.train_state_sharding, self.replicated_sharding),
            out_shardings=self.replicated_sharding,
        )

        # Data configs, setup and utils
        self.data_config = config.data.create(config.assets_dirs, config.model)
        logging.info(f"data_config: {self.data_config}")
        self.data_norm_stats = self.data_config.norm_stats
        self.data_transforms = [*self.data_config.repack_transforms.inputs,
            *self.data_config.data_transforms.inputs,
            _transforms.Normalize(self.data_norm_stats, use_quantiles=self.data_config.use_quantile_norm),
            *self.data_config.model_transforms.inputs,]
        self.data_transforms = _transforms.compose(self.data_transforms)



    # ------------------------------------------------------------------------------------
    # INFERENCE
    # ------------------------------------------------------------------------------------
    def sample_actions(self, observations: Data, repeat=1, cache_dir=None, timer=None, argmax=False, **unused):
        with sharding.set_mesh(self.mesh):
            obs = _model.Observation.from_dict(observations)
            return self._pinfer(self.train_state, obs)

    # ------------------------------------------------------------------------------------
    # TRAINING (supervised updates)
    # ------------------------------------------------------------------------------------
    def update(self, batch: Batch):
        with sharding.set_mesh(self.mesh):
            batch = (_model.Observation.from_dict(batch), batch["actions"])
            self.train_state, info = self._ptrain_step(
                self.train_rng, self.train_state, batch
            )
        return info

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
            None,
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



