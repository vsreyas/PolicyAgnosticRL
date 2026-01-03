"""Script for critic-only training using warm-start cached VLM actions.

This script trains only the critic network using pre-computed VLM actions from a .pkl cache file.
No forward passes through the VLM/Pi0 model are needed, making training much faster.

The cache file contains N sampled actions for each state. During training, we randomly
sample one action from these N options for each batch element to create diversity.
"""

import os
import pickle
import time
import cv2
from typing import Dict

import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
import tensorflow as tf
import wandb
from absl import app, flags, logging
from ml_collections import config_flags
from tqdm import tqdm
from matplotlib import pyplot as plt
import io
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Any


from jaxrl_m.agents.continuous.expo_pi import ExpoPiLearner
from jaxrl_m.common.wandb import WandBLogger
from jaxrl_m.utils.timer_utils import Timer
from openpi.training.config import get_config
from jaxrl_m.envs.libero import (
    get_libero_config,
    get_libero_env,
    get_libero_tfrecord_dataset,
)
from jaxrl_m.common.evaluation import evaluate_with_trajectories_libero, save_rollout_gif, supply_rng
from jaxrl_m.agents.continuous.expo_pi import ExpoPiLearner, compute_q, compute_q_all
# from train_expo_pi import get_policy_fn
from jaxrl_m.common.traj import calc_return_to_go
from jaxrl_m.common.typing import Batch, Data

try:
    from jax_smi import initialise_tracking  # type: ignore
    initialise_tracking()
except ImportError:
    pass

FLAGS = flags.FLAGS

flags.DEFINE_string("environment_name", "libero", "Environment name.")
flags.DEFINE_string("wandb_project_name", "PI-0.5-critic-warmstart", "WandB project name.")
flags.DEFINE_string("wandb_experiment_name", "", "WandB experiment name.")
flags.DEFINE_string("wandb_group", "", "WandB group.")
config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("num_train_steps", 10000, "Number of training steps.")
# flags.DEFINE_integer("batch_size", 256, "Batch size for training.")
flags.DEFINE_string(
    "vlm_cache_path",
    None,
    "Path to the .pkl file containing cached VLM actions.",
)
flags.DEFINE_float("reward_scale", 1.0, "Reward scale.")
flags.DEFINE_float("reward_bias", 0.0, "Reward bias.")
# flags.DEFINE_string(
#     "save_dir",
#     None,
#     "Directory to save checkpoints (optional).",
# )
# flags.DEFINE_integer(
#     "log_interval",
#     100,
#     "Logging interval (steps).",
# )
# flags.DEFINE_integer(
#     "save_interval",
#     1000,
#     "Checkpoint saving interval (steps).",
# )
flags.DEFINE_bool("debug", False, "Debug mode")
flags.DEFINE_string(
    "task_name",
    "put both moka pots on the stove",
    "Name of task (for logging)",
)
flags.DEFINE_integer(
    "num_actions_to_sample", 4, "Number of actions that were sampled in cache (N)."
)
flags.DEFINE_integer(
    "num_edit_samples", 4, "Number of edit samples (for agent initialization)."
)

import jax.tree_util as jtu


def _leaf_has_leading_dim(x: Any) -> bool:
    return hasattr(x, "shape") and x.shape is not None and len(x.shape) >= 1


def _infer_leading_dim(tree: Any) -> Optional[int]:
    """Return leading dim from first array-like leaf, else None."""
    leaves = jtu.tree_leaves(tree)
    for leaf in leaves:
        if _leaf_has_leading_dim(leaf):
            return int(leaf.shape[0])
    return None


def _slice_tree(tree: Any, idx: np.ndarray) -> Any:
    """Index every array leaf by idx; leaves without shape are left unchanged."""
    def slicer(x):
        if _leaf_has_leading_dim(x):
            return x[idx]
        return x  # metadata-like leaves (strings, ints) untouched
    return jtu.tree_map(slicer, tree)


@dataclass
class CacheBatchLoader:
    """
    Infinite epoch-style loader over a cache dict.

    - At epoch start: create permutation of [0..num_entries)
    - Yield sequential batches from that permuted order
    - When exhausted: reshuffle and repeat forever
    """
    cache: Dict[str, Any]
    batch_size: int
    rng: jax.random.PRNGKey
    shuffle: bool = True
    drop_last: bool = True
    include_keys: Optional[Sequence[str]] = None
    exclude_keys: Sequence[str] = ("metadata",)  # exclude non-training blobs by default
    to_jax: bool = True
    device_put: bool = False

    def __post_init__(self):
        # Decide which keys to serve
        keys = list(self.cache.keys()) if self.include_keys is None else list(self.include_keys)
        keys = [k for k in keys if k not in set(self.exclude_keys)]

        # Keep only keys that have at least one array-like leaf (arrays or dict-of-arrays, etc.)
        self.keys = []
        self.num_entries = None

        for k in keys:
            ld = _infer_leading_dim(self.cache[k])
            if ld is None:
                continue  # skip pure-metadata dicts/lists with no array leaves
            if self.num_entries is None:
                self.num_entries = ld
            elif ld != self.num_entries:
                raise ValueError(
                    f"Key '{k}' leading dim {ld} != {self.num_entries}. "
                    "All indexable entries must share the same leading dimension."
                )
            self.keys.append(k)

        if self.num_entries is None:
            raise ValueError(
                "Could not infer dataset length: none of the selected cache entries had array-like leaves."
            )

        self.epoch = 0
        self.pos = 0
        self._reset_epoch()

    def _reset_epoch(self):
        self.epoch += 1
        self.pos = 0
        self.rng, key = jax.random.split(self.rng, 2)
        if self.shuffle:
            perm = jax.random.permutation(key, self.num_entries)  # :contentReference[oaicite:2]{index=2}
            self.perm = np.asarray(perm)
        else:
            self.perm = np.arange(self.num_entries)

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, Any]:
        # restart epoch if exhausted
        if self.pos >= self.num_entries:
            self._reset_epoch()

        end = self.pos + self.batch_size
        if end > self.num_entries:
            if self.drop_last:
                self._reset_epoch()
                end = self.batch_size
            else:
                end = self.num_entries

        idx = self.perm[self.pos:end]
        self.pos = end

        batch = {k: _slice_tree(self.cache[k], idx) for k in self.keys}

        

        if self.to_jax:
            batch = jtu.tree_map(lambda x: jnp.asarray(x) if _leaf_has_leading_dim(x) else x, batch)

        if self.device_put:
            batch = jtu.tree_map(lambda x: jax.device_put(x) if _leaf_has_leading_dim(x) else x, batch)

        return batch

def get_policy_fn(
    agent: ExpoPiLearner,
    rng: jax.random.PRNGKey,
    timer: Timer | None = None,
    debug_mode: bool = False,
) -> Callable[[Data], np.ndarray]:
    def policy_fn(observations: Data, *args, **kwargs) -> np.ndarray:
        if not isinstance(observations, dict):
            observations = {"state": observations}
        if "state" in observations:
            obs_ndim = observations["state"].ndim
        else:
            assert "proprio" in observations
            obs_ndim = observations["proprio"].ndim
        
        # breakpoint()
        out_dict = jax.device_get(
            agent.sample_actions(
                observations, *args, **kwargs, timer=timer, output_action_chunk=True, debug_mode=debug_mode
            )
        )
        # breakpoint()

        return out_dict

    policy_fn = supply_rng(policy_fn, rng=rng)

    return policy_fn

def filter_cache(cache: Dict) -> Dict:
    """Filter the cache to only include entries where the current state is not None."""
    # import pickle
    # import pandas as pd
    # import numpy as np
    # file=open("outputs/vlm_actions_replay_ep30_v1_clean_v3.pkl","rb"); cache=pickle.load(file);

    # Find keys
    action_horizon = cache['current_actions'].shape[2]
    episode_ids = cache['episode_ids']
    episode_timesteps = cache['episode_timesteps']
    terminals = cache['terminals']
    df = pd.DataFrame({
        'episode_id': episode_ids,
        'episode_timestep': episode_timesteps,
        'terminals': terminals,
    })

    pos = df.groupby("episode_id").cumcount()
    ep_len = df.groupby("episode_id")["episode_timestep"].transform("size")
    terminals = df["terminals"].astype(bool)
    keep_in_sorted = terminals | (pos < (ep_len - action_horizon))
    keep_idx = df.index[keep_in_sorted]
    filtered_df = df.loc[df.index.isin(keep_idx)].copy()

    filtered_index = np.array(filtered_df.index)

    for key in cache.keys():
        if key == "metadata":
            continue
        
        if type(cache[key]) == dict:
            for subkey in cache[key].keys():
                cache[key][subkey] = cache[key][subkey][filtered_index]
        else:
            cache[key] = cache[key][filtered_index]
    
    return cache


def load_vlm_cache(cache_path: str) -> Dict:
    """Load the VLM action cache from disk.
    
    Args:
        cache_path: Path to the .pkl cache file
        
    Returns:
        Dictionary containing:
            - episode_ids: (num_entries,) - Episode IDs
            - episode_timesteps: (num_entries,) - Timesteps
            - current_vlm_outputs: (num_entries, N, vlm_dim) - N VLM outputs for current state
            - next_vlm_outputs: (num_entries, N, vlm_dim) - N VLM outputs for next state
            - current_actions: (num_entries, N, action_horizon, action_dim) - N actions for current state
            - next_actions: (num_entries, N, action_horizon, action_dim) - N actions for next state
            - current_states: (num_entries, 8) - Current states
            - next_states: (num_entries, 8) - Next states
            - rewards: (num_entries,) - Rewards
            - masks: (num_entries,) - Masks
            - actions: (num_entries, action_horizon, action_dim) - Ground truth actions
            - metadata: Dictionary with cache metadata
    """
    logging.info(f"Loading VLM cache from {cache_path}")
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    
    cache = filter_cache(cache)
    # breakpoint()
    
    # Normalize inputs #
    # breakpoint()
    mean_vlm_outputs = np.mean(cache['current_vlm_outputs'], axis=0)
    std_vlm_outputs = np.std(cache['current_vlm_outputs'], axis=0)
    # cache['current_vlm_outputs'] = (cache['current_vlm_outputs'] - mean_vlm_outputs) / (std_vlm_outputs + 1e-8)
    # cache['next_vlm_outputs'] = (cache['next_vlm_outputs'] - mean_vlm_outputs) / (std_vlm_outputs + 1e-8)

    # Log cache statistics
    num_entries = len(cache['episode_ids'])
    logging.info(f"Cache statistics:")
    logging.info(f"  Total entries: {num_entries}")
    logging.info(f"  Current VLM outputs shape: {cache['current_vlm_outputs'].shape}")
    logging.info(f"  Next VLM outputs shape: {cache['next_vlm_outputs'].shape}")
    logging.info(f"  Current actions shape: {cache['current_actions'].shape}")
    logging.info(f"  Next actions shape: {cache['next_actions'].shape}")
    logging.info(f"  Rewards shape: {cache['rewards'].shape}")
    logging.info(f"  Metadata: {cache['metadata']}")

    # breakpoint()
    
    return cache, mean_vlm_outputs, std_vlm_outputs


def sample_batch_from_cache(
    cache: Dict,
    batch_size: int,
    rng: jax.random.PRNGKey,
) -> Dict:
    """Sample a batch of data from the VLM cache.
    
    Args:
        cache: VLM cache dictionary
        batch_size: Number of samples to draw
        rng: JAX random key
        
    Returns:
        Batch dictionary with sampled data
    """
    num_entries = len(cache['episode_ids'])
    
    # Sample random indices
    key, rng = jax.random.split(rng)
    indices = jax.random.choice(
        key,
        num_entries,
        shape=(batch_size,),
        replace=False if batch_size <= num_entries else True,
    )
    
    # Convert to numpy for indexing (JAX arrays can be used directly)
    indices_np = np.array(indices)

    if 'mc_returns' not in cache:
        cache['mc_returns'] = np.zeros((num_entries,))
    
    # Create batch by indexing into cache
    batch = {
        'current_vlm_outputs': cache['current_vlm_outputs'][indices_np],  # (batch_size, N, vlm_dim)
        'next_vlm_outputs': cache['next_vlm_outputs'][indices_np],  # (batch_size, N, vlm_dim)
        'current_actions': cache['current_actions'][indices_np],  # (batch_size, N, action_horizon, action_dim)
        'next_actions': cache['next_actions'][indices_np],  # (batch_size, N, action_horizon, action_dim)
        'current_states': cache['current_states'][indices_np],  # (batch_size, 8)
        'next_states': cache['next_states'][indices_np],  # (batch_size, 8)
        'rewards': cache['rewards'][indices_np],  # (batch_size,)
        'masks': cache['masks'][indices_np],  # (batch_size,)
        'actions': cache['actions'][indices_np],  # (batch_size, action_horizon, action_dim)
        'mc_returns': cache['mc_returns'][indices_np],  # (batch_size,)
        'terminals': cache['terminals'][indices_np],  # (batch_size,)
        'truncates': cache['truncates'][indices_np],  # (batch_size,)
        # 'metadata': cache['metadata'],  # Dictionary with cache metadata
        "next_actions_sampled": cache['next_actions_sampled'][indices_np],  # (batch_size, N, action_horizon, action_dim)
    }
    
    # Convert to JAX arrays
    batch = jax.tree_map(lambda x: jnp.array(x), batch)
    
    return batch


def train_critic(_):
    """Main training loop for critic using warm-start cache."""
    
    if FLAGS.debug:
        # Disabling jit might be useful for debugging
        # jax.config.update("jax_disable_jit", True)
        pass
    
    # Prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")
    
    # Setup WandB
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    os.environ["WANDB_INIT_TIMEOUT"] = "120"
    wandb.require("core")
    
    devices = jax.local_devices()
    num_devices = len(devices)
    logging.info(f"Found {num_devices} JAX devices")
    rng = jax.random.PRNGKey(FLAGS.seed)
    # Create timer for performance tracking
    timer = Timer()
    
    # Get PI config
    pi_config = get_config("pi05_libero_custom_low_mem")
    pi_config.fsdp_devices = 1
    pi_config.exp_name = FLAGS.wandb_experiment_name
    pi_config.overwrite = True
    
    # Setup WandB logging
    if FLAGS.wandb_project_name is not None:
        wandb_config = WandBLogger.get_default_config()
        wandb_config.update(
            {
                "project": FLAGS.wandb_project_name,
                "exp_descriptor": FLAGS.wandb_experiment_name,
                "tag": None,
                "group": FLAGS.wandb_group,
            }
        )
        wandb_logger = WandBLogger(
            wandb_config=wandb_config,
            variant=FLAGS.config.to_dict(),
            debug=FLAGS.debug,
            allow_val_change=True,
        )
        save_dir = os.path.join(
            os.path.abspath(FLAGS.config.save_dir) if FLAGS.config.save_dir else "./checkpoints",
            f"seed_{FLAGS.seed}",
        )
    else:
        wandb_logger = None
        save_dir = os.path.abspath(FLAGS.config.save_dir) if FLAGS.config.save_dir else "./checkpoints"
    
    os.makedirs(save_dir, exist_ok=True)
    logging.info(f"Save directory: {save_dir}")
    
    # Load VLM cache
    assert FLAGS.vlm_cache_path is not None, "Must provide --vlm_cache_path"
    cache, mean_vlm_outputs, std_vlm_outputs = load_vlm_cache(FLAGS.vlm_cache_path)
    # Each key is (num_data_points, ...) based on corresponding array shapes
    # breakpoint()
    
    # Create a dummy batch for agent initialization
    # Get shapes from cache
    vlm_dim = cache['current_vlm_outputs'].shape[1]  # N, vlm_dim -> vlm_dim
    action_horizon = cache['current_actions'].shape[2]  # N, action_horizon, action_dim -> action_horizon
    action_dim = cache['current_actions'].shape[3]  # N, action_horizon, action_dim -> action_dim
    
    logging.info(f"Inferred shapes: vlm_dim={vlm_dim}, action_horizon={action_horizon}, action_dim={action_dim}")
    
    # Create dummy observation for agent initialization
    dummy_batch = {
        'current_vlm_outputs': jnp.zeros((FLAGS.config.agent_kwargs.batch_size, vlm_dim)),
        'next_vlm_outputs': jnp.zeros((FLAGS.config.agent_kwargs.batch_size, vlm_dim)),
        'current_actions': jnp.zeros((FLAGS.config.agent_kwargs.batch_size, 1, action_horizon, action_dim)),
        'next_actions': jnp.zeros((FLAGS.config.agent_kwargs.batch_size, FLAGS.num_actions_to_sample, action_horizon, action_dim)),
        'current_states': jnp.zeros((FLAGS.config.agent_kwargs.batch_size, 8)),
        'next_states': jnp.zeros((FLAGS.config.agent_kwargs.batch_size, 8)),
        'rewards': jnp.zeros((FLAGS.config.agent_kwargs.batch_size,)),
        'masks': jnp.ones((FLAGS.config.agent_kwargs.batch_size,)),
        'mc_returns': jnp.zeros((FLAGS.config.agent_kwargs.batch_size,)),
        'terminals': jnp.zeros((FLAGS.config.agent_kwargs.batch_size,)),
        'truncates': jnp.zeros((FLAGS.config.agent_kwargs.batch_size,)),
        'next_actions_sampled': jnp.zeros((FLAGS.config.agent_kwargs.batch_size, FLAGS.num_actions_to_sample, action_horizon, action_dim)),
        # 'metadata': {},
    }
    
    # Create EXPO agent
    logging.info("Creating EXPO agent...")
    rng, construct_rng = jax.random.split(rng)
    
    agent = ExpoPiLearner.create(
        config=pi_config,
        seed=FLAGS.seed,
        observations=dummy_batch,  # dummy batch for initialization
        rng=construct_rng,
        N=FLAGS.num_actions_to_sample,
        n_edit_samples=FLAGS.num_edit_samples,
        batch_size_dict_key='current_actions',
        q_clip_low=FLAGS.config.q_clip_low,
        q_clip_high=FLAGS.config.q_clip_high,
    )

    # breakpoint()

    libero_config = get_libero_config()
    eval_env = get_libero_env(cfg=libero_config, task_name=FLAGS.task_name, is_pi=True)
    ### Evaluation Setup ###
    env_data_collection_policy_fn = None  # Will get set later
    rng, eval_policy_fn_key = jax.random.split(rng)
    eval_policy_fn = get_policy_fn(
        agent=agent,
        rng=eval_policy_fn_key,
        timer=timer,
    )
    #########################################################

    loader = CacheBatchLoader(
        cache=cache,
        batch_size=FLAGS.config.agent_kwargs.batch_size,
        rng=rng,                # will be updated internally per epoch
        shuffle=True,
        drop_last=True,
        # keys=None -> auto
        to_jax=True,
        device_put=False,
    )

    log_interval = 1
    save_interval = 10000

    for step in tqdm(range(FLAGS.num_train_steps), desc="Training critic"):
        timer.tick("total_step_time")

        timer.tick("sample_batch_time")
        batch = next(loader)     # sequential batches within an epoch; reshuffles on exhaustion
        timer.tock("sample_batch_time")

        # Optional action slicing you do
        batch["actions"] = batch["actions"][:, :, :7]

        timer.tick("update_critic_time")
        rng, update_rng = jax.random.split(loader.rng)  # keep RNG usage consistent if you want
        loader.rng = rng
        agent, info = agent.update_critic_ws(batch=batch, seed=update_rng, timer=timer)

        # if step % 100 == 0:
        #     breakpoint()
        
        timer.tock("update_critic_time")

        timer.tock("total_step_time")
    
    # # TODO: Later move these to flags #

    # logging.info("Agent created successfully")
    
    # # Training loop
    # logging.info(f"Starting critic training for {FLAGS.num_train_steps} steps")
    # logging.info(f"Batch size: {FLAGS.config.agent_kwargs.batch_size}")
    
    # for step in tqdm(range(FLAGS.num_train_steps), desc="Training critic"):
    #     timer.tick("total_step_time")
        
    #     # Sample batch from cache
    #     timer.tick("sample_batch_time")
    #     rng, sample_rng = jax.random.split(rng)
    #     batch = sample_batch_from_cache(cache, FLAGS.config.agent_kwargs.batch_size, sample_rng)
    #     # breakpoint()
    #     batch['actions'] = batch['actions'][:, :, :7]
    #     timer.tock("sample_batch_time")
        
        # Update critic
        # timer.tick("update_critic_time")
        # rng, update_rng = jax.random.split(rng)
        # agent, info = agent.update_critic_ws(
        #     batch=batch,
        #     seed=update_rng,
        #     timer=timer,
        # )
        # timer.tock("update_critic_time")
        
        # timer.tock("total_step_time")
        
        # Logging
        if (step + 1) % log_interval == 0:
            # Log training metrics
            # log_dict = {
            #     "step": step + 1,
            #     "critic_loss": float(info["critic_loss"]),
            #     "q_mean": float(info["q_mean"]),
            #     "q_std": float(info["q_std"]),
            #     "target_q_mean": float(info["target_q_mean"]),
            #     "target_q_std": float(info["target_q_std"]),
            #     "next_qs_mean": float(info["next_qs_mean"]),
            #     "next_qs_std": float(info["next_qs_std"]),
            # }
            log_dict = info

            # Log batch statsitics
            batch_stats = {
                # "batch_stats/batch_size": len(batch),
                "batch_stats/batch_size": int(batch["rewards"].shape[0]),
                "batch_stats/epoch": int(loader.epoch),
                "batch_stats/masks_mean": np.mean(batch["masks"]),
                "batch_stats/mc_returns_mean": np.mean(batch["mc_returns"]),
                "batch_stats/mc_returns_min": np.min(batch["mc_returns"]),
                "batch_stats/mc_returns_max": np.max(batch["mc_returns"]),
                "batch_stats/rewards_mean": np.mean(batch["rewards"]),
                "batch_stats/rewards_min": np.min(batch["rewards"]),
                "batch_stats/rewards_max": np.max(batch["rewards"]),
                "batch_stats/terminals_mean": np.mean(batch["terminals"]),
                "batch_stats/truncations_mean": np.mean(batch["truncates"]),
            }
            
            if wandb_logger is not None:
                wandb_logger.log(batch_stats, step=step)
            
            # Add timing info
            timing_info = timer.get_average_times()
            log_dict.update({f"timing/{k}": v for k, v in timing_info.items()})
            
            # Log to wandb
            if wandb_logger is not None:
                wandb_logger.log(log_dict, step=step)
            
            # Print to console
            logging.info(f"Step {step + 1}: " + 
                        f"loss={log_dict['critic_loss']:.4f}, " +
                        f"q_mean={log_dict['q_mean']:.4f}, " +
                        f"target_q_mean={log_dict['target_q_mean']:.4f}")
        
        # Save checkpoint
        if FLAGS.config.save_dir and (step + 1) % save_interval == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_{step + 1}.pkl")
            logging.info(f"Saving checkpoint to {checkpoint_path}")
            # Save critic parameters
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({
                    'critic_params': agent.critic.params,
                    'target_critic_params': agent.target_critic.params,
                    'step': step + 1,
                }, f)
        
        ### Evaluation ###
        if (
            (step + 1) % FLAGS.config.eval_interval == 0
        ) and eval_env is not None:
            # breakpoint()
            """eval"""
            logging.info("Evaluating...")
            timer.tick("evaluation/total")

            if FLAGS.config.save_video:
                try:
                    eval_env.start_recording(
                        FLAGS.config.num_episodes_per_video,
                        FLAGS.config.num_episodes_per_row,
                    )
                except Exception as e:
                    pass
            
            if FLAGS.config.num_eval_episodes > 0:
                print("Evaluating...")
                q_vs_mc_returns_vals = None
                eval_policy_fn = get_policy_fn(
                    agent=agent,
                    rng=eval_policy_fn_key,
                    timer=timer,
                    debug_mode=True,
                )
                trajectories, q_vs_mc_returns_vals = evaluate_with_trajectories_libero(
                    eval_policy_fn,
                    eval_env,
                    FLAGS.config.num_eval_episodes,
                    action_horizon=pi_config.model.action_horizon,
                    # action_horizon=1
                    use_full_horizon_for_refill=True, # For proper Q vs MC Returns calculation #
                )
                
                # breakpoint()

                if (FLAGS.environment_name == "calvin" or FLAGS.environment_name =='libero') and FLAGS.config.save_video:
                    trajectories_to_save = trajectories[
                        : FLAGS.config.num_episodes_per_video
                    ]
                    frames = []
                    ind_traj = []
                    for j, traj in enumerate(trajectories_to_save):
                        trajectory_return = 0
                        for transition, reward in zip(
                            traj["observation"], traj["reward"]
                        ):
                            assert transition["image"].shape[-1] == 3
                            if len(transition["image"].shape) == 4:
                                transition["image"] = transition["image"][0]
                            image = transition["image"]  # .transpose(2, 0, 1)
                            # Add text for reward and return so far
                            trajectory_return += reward
                            # image = np.flipud(image)
                            image = np.ascontiguousarray(image) 
                            frame = cv2.putText(
                                image,
                                f"reward: {reward}. return: {trajectory_return}",
                                (10, 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.3,
                                (0, 0, 0),
                                1,
                            )
                            ind_traj.append(frame)
                            frame = frame.transpose(2, 0, 1)
                            frames.append(frame)
                        
                        save_rollout_gif(ind_traj, save_dir, step_i=step, rollout_j=j)
                        ind_traj = []
                    
                    del ind_traj, frames
                    import gc; gc.collect()

                eval_metrics = {
                    "eval/average_return": np.mean(
                        [np.sum(t["reward"]) for t in trajectories]
                    ),
                    "eval/average_episode_length": np.mean(
                        [len(t["reward"]) for t in trajectories]
                    ),
                    **(
                        {
                            "eval/average_normalized_return": np.mean(
                                [
                                    eval_env.get_normalized_score(np.sum(t["reward"]))
                                    for t in trajectories
                                ]
                            ),
                            "eval/min_normalized_return": np.min(
                                [
                                    eval_env.get_normalized_score(np.sum(t["reward"]))
                                    for t in trajectories
                                ]
                            ),
                            "eval/max_normalized_return": np.max(
                                [
                                    eval_env.get_normalized_score(np.sum(t["reward"]))
                                    for t in trajectories
                                ]
                            ),
                        }
                        if hasattr(eval_env, "get_normalized_score")
                        else {}
                    ),
                    "eval/average_max_reward": np.mean(
                        [np.max(t["reward"]) for t in trajectories]
                    ),
                }

                if wandb_logger is not None:
                    wandb_logger.log(eval_metrics, step=step)
                
                # Log Q vs MC Returns #
                if q_vs_mc_returns_vals is not None:
                    qs_across_trajectories = []
                    for q_vs_mc_return_val in q_vs_mc_returns_vals: # Per trajectory
                        qs = []
                        for idx in range(len(q_vs_mc_return_val)):
                            vlm_output, action_sequence = q_vs_mc_return_val[idx]
                            params = agent.critic.params
                            # Prepare input for critic #
                            vlm_output = vlm_output.reshape(1, -1)
                            vlm_only = vlm_output[:, :2048]
                            # vlm_only = (vlm_only - mean_vlm_outputs) / (std_vlm_outputs + 1e-8)
                            state_only = vlm_output[:, 2048:]
                            vlm_output = jnp.concatenate([vlm_only, state_only], axis=1)
                            # vlm_output = state_only
                            action_sequence = action_sequence.reshape(1, -1)
                            q_vals = compute_q_all(agent.critic.apply_fn, params, vlm_output, action_sequence)
                            # breakpoint()
                            qs.append(q_vals[0]) # Look at q1 specifically #
                        qs_across_trajectories.append(qs)
                    
                    H = pi_config.model.action_horizon  # chunk size
                    gamma_step = FLAGS.config.agent_kwargs.discount

                    gamma_chunk = gamma_step # ** H

                    mc_returns = []
                    for traj_i, t in enumerate(trajectories):
                        # per-step rewards (same scaling/biasing you already do)
                        r = np.asarray(t["reward"], dtype=np.float32) * FLAGS.reward_scale + FLAGS.reward_bias
                        d = np.asarray(t["done"], dtype=np.bool_)  # per-step done flags

                        # chunk starts: 0, H, 2H, ...
                        starts = np.arange(0, len(r), H)

                        # sum rewards inside each chunk; last chunk can be shorter automatically
                        r_chunks = np.add.reduceat(r, starts)

                        # chunk done/mask: mark chunk terminal if ANY step inside the chunk is done
                        done_chunks = np.array([d[s : min(s + H, len(d))].any() for s in starts], dtype=np.float32)
                        masks_chunks = 1.0 - done_chunks

                        # assert alignment with the Q-values you logged per chunk
                        assert len(r_chunks) == len(q_vs_mc_returns_vals[traj_i]), (
                            f"traj {traj_i}: #reward_chunks={len(r_chunks)} != "
                            f"#q_vs_mc_points={len(q_vs_mc_returns_vals[traj_i])} (H={H}, T={len(r)})"
                        )

                        mc = calc_return_to_go(
                            rewards=r_chunks,
                            masks=masks_chunks,
                            gamma=gamma_chunk,
                            push_failed_to_min=("maze" in FLAGS.environment_name) or (FLAGS.environment_name == "real_robot"),
                            min_reward=FLAGS.reward_bias,
                        )

                        # breakpoint()
                        mc_returns.append(mc)

                    initial_mc_returns = [mc[0] for mc in mc_returns]

                    # 1. Logic to create flattened arrays for plotting
                    all_q_values = []
                    all_mc_returns = []

                    for idx in range(len(trajectories)):
                        mc_return = mc_returns[idx]
                        q_vals = qs_across_trajectories[idx]
                        
                        # ... [Your slicing logic] ...
                        # Ensure this logic matches your specific horizon/stride needs
                        # mc_return_slice = mc_return[pi_config.model.action_horizon::pi_config.model.action_horizon]
                        mc_return_slice = mc_return
                        assert len(mc_return_slice) == len(q_vals)
                        # Safety clip to matching length
                        min_len = min(len(mc_return_slice), len(q_vals))
                        
                        # Flatten just in case, though they should already be 1D
                        q_vals_segment = np.array(q_vals[:min_len]).reshape(-1)
                        mc_return_segment = np.array(mc_return_slice[:min_len]).reshape(-1)
                        
                        all_q_values.append(q_vals_segment)
                        all_mc_returns.append(mc_return_segment)

                    # Concatenate for plotting
                    if len(all_q_values) > 0:
                        flat_q_values = np.concatenate(all_q_values, axis=0)
                        flat_mc_returns = np.concatenate(all_mc_returns, axis=0)
                        np.save(f"flat_q_values_iter_{step}.npy", flat_q_values)
                        np.save(f"flat_mc_returns_iter_{step}.npy", flat_mc_returns)

                        # Setup simple plot
                        plt.figure(figsize=(8, 6))
                        
                        # Basic scatter plot
                        plt.scatter(flat_q_values, flat_mc_returns, alpha=0.6, s=20)

                        # Simple labels and grid
                        plt.xlabel("Q-Values (Predicted)")
                        plt.ylabel("Monte Carlo Returns (Actual)")
                        plt.title(f"Q-Values vs Returns (Step {step})")
                        plt.grid(True, alpha=0.3)

                        # Save to WandB
                        buf = io.BytesIO()
                        plt.savefig(buf, format="png", bbox_inches='tight')
                        buf.seek(0)
                        image = Image.open(buf)

                        if wandb_logger is not None:
                            wandb_logger.log({
                                "eval/q_vs_mc_scatter": wandb.Image(image, caption="Q vs MC Scatter Plot")
                            }, step=step)

                        plt.close()
                        buf.close()
                    else:
                        print("Warning: No data available for Q vs MC plot.")
                
                del trajectories
                import gc; gc.collect()

            timer.tock("evaluation/total")
    
    logging.info("Training complete!")
    
    # Final checkpoint
    if FLAGS.config.save_dir:
        final_checkpoint_path = os.path.join(save_dir, "checkpoint_final.pkl")
        logging.info(f"Saving final checkpoint to {final_checkpoint_path}")
        with open(final_checkpoint_path, 'wb') as f:
            pickle.dump({
                'critic_params': agent.critic.params,
                'target_critic_params': agent.target_critic.params,
                'step': FLAGS.num_train_steps,
            }, f)
    
    # Print final timing statistics
    logging.info("=" * 80)
    logging.info("Final timing statistics:")
    timing_stats = timer.get_average_times()
    for key, value in timing_stats.items():
        logging.info(f"  {key}: {value:.4f}s")
    logging.info("=" * 80)


if __name__ == "__main__":
    app.run(train_critic)

