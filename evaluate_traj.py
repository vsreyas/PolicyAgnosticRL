"""Script for offline to online RL."""

import os
import time
from typing import Any, Callable, Dict, List, Optional, Union

import cv2
import flax
import gym
import jax
import jax.numpy as jnp
import numpy as np
import seaborn as sns
import tensorflow as tf
import io
from PIL import Image
import pickle
from PIL import Image, ImageDraw, ImageFont

import wandb
from absl import app, flags, logging
from matplotlib import pyplot as plt
from ml_collections import config_flags
from tqdm import tqdm
import functools

from jaxrl_m.agents import agents
from jaxrl_m.agents.continuous.action_optimization import (
    LocalOptimizationState,
    action_optimization_sample_actions,
    add_base_policy_actions_to_batch,
    local_optimization_steps as take_local_optimization_steps,
)
from jaxrl_m.agents.continuous.auto_regressive_transformer import (
    AutoRegressiveTransformerAgent,
)
from jaxrl_m.agents.continuous.base_policy import BasePolicy, BasePolicyTypes
from jaxrl_m.agents.continuous.ddpm_bc import DDPMBCAgent
from jaxrl_m.agents.continuous.openvla import OpenVLAAgent
from jaxrl_m.agents.continuous.pi_0 import PiPolicy
from jaxrl_m.common.common import JaxRLTrainState
from jaxrl_m.common.evaluation import evaluate_with_trajectories_vectorized, supply_rng, evaluate_with_trajectories_libero, save_rollout_gif
from jaxrl_m.common.traj import TrajSampler, calc_return_to_go
from jaxrl_m.common.typing import Batch, Data
from jaxrl_m.common.wandb import WandBLogger
from jaxrl_m.data.bridge_dataset import (
    BridgeDataset,
    get_task_to_initial_eep,
    glob_to_path_list,
)
from jaxrl_m.data.image_replay_buffer_pi import (
    ImageReplayBufferPi,
    save_trajectory_as_tfrecord,
)
from jaxrl_m.data.replay_buffer import ReplayBuffer
# D4RL is not needed for Libero EXPO; keep optional to avoid Mujoco dependency errors.
# from jaxrl_m.envs.d4rl import TruncationWrapper, get_d4rl_dataset_with_mc_calculation
from jaxrl_m.utils.timer_utils import Timer
from jaxrl_m.utils.train_utils import concatenate_batches, load_recorded_video
from jaxrl_m.vision import encoders
from jaxrl_m.utils.train_utils import preprocess_action, repack_action
from jaxrl_m.agents.continuous.expo_pi import ExpoPiLearner, compute_q, compute_q_all
from jaxrl_m.agents.continuous.expo_pi_cache import ExpoPiLearnerCache
from jaxrl_m.utils.expo_utils import calc_mc_return_fn
from jaxrl_m.agents.continuous.pi_vlm_cached import (
    create_agent,
    PiResidualTD3Cache,
    PiResidualPPOCache,
)

try:
    from jax_smi import initialise_tracking  # type: ignore

    initialise_tracking()
except ImportError:
    pass

from openpi.training.config import get_config

FLAGS = flags.FLAGS

flags.DEFINE_string("environment_name", "", "Environment name.")
flags.DEFINE_string("wandb_project_name", "PI-0.5-finetuning", "WandB project name.") #"PA-RL""debug"
flags.DEFINE_string("wandb_experiment_name", "", "WandB experiment name.")
flags.DEFINE_string("wandb_group", "", "WandB group.")
flags.DEFINE_string("agent_name", "pi_residual_td3", "Agent name (pi_residual_td3 or pi_residual_ppo).")
config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)
config_flags.DEFINE_config_file(
    "bridgedata_config",
    None,
    "File path to the bridgedata configuration.",
    lock_config=False,
)
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("num_offline_epochs", 100, "Number of epochs for pre-training.")
flags.DEFINE_integer(
    "num_online_epochs", 500, "Number of epochs for online fine-tuning."
)
flags.DEFINE_integer(
    "num_train_steps_per_offline_epoch", 50, "Number of training steps per epoch."
)
flags.DEFINE_float("reward_scale", 1.0, "Reward scale.")
flags.DEFINE_float("reward_bias", 0.0, "Reward bias.")
flags.DEFINE_float("clip_action", 200.0, "Clip action.")
flags.DEFINE_integer("num_parallel_envs", 1, "Number of parallel environments.")
flags.DEFINE_bool("debug", False, "Debug config")
flags.DEFINE_string("resume_path", None, "Resume training from checkpoint.")
flags.DEFINE_integer("max_episode_steps", 1000, "Maximum episode steps.")
flags.DEFINE_string(
    "replay_buffer_path", "", "Path to replay buffer to load (Optional)."
)
flags.DEFINE_string(
    "train_on_separate_computer_mode",
    "single_computer",  # "env_steps_only", "agent_training_only"
    "Training on separate computer mode.",
)

# Q-diffusion pre-processing flags
flags.DEFINE_integer(
    "num_online_trajectories_per_epoch",
    1,
    "Number of trajectories collected from interaction per online epoch.",
)
flags.DEFINE_integer(
    "num_trajectories_to_collect",
    None,
    "Optional override for number of evaluation trajectories.",
)
flags.DEFINE_integer(
    "num_warmup_trajectories",
    1,
    "Number of trajectories collected from interaction before training.",
)
flags.DEFINE_integer("critic_utd", 1, "update-to-data ratio of the critic")
flags.DEFINE_float(
    "base_policy_utd",
    1,
    "Update-to-data ratio of the base policy for distillation.",
)
flags.DEFINE_string(
    "base_policy_offline_cache_path",
    None,
    "Path to pre-computed base policy actions to use for pre-training.",
)
flags.DEFINE_string(
    "task_name",
    "put both moka pots on the stove", 
    "Name of fixed task"
)
flags.DEFINE_bool(
    "plot_q_values_over_trajectory_figure",
    False,
    "Plot Q-values over trajectory time step.",
)
flags.DEFINE_bool(
    "use_lang",
    True,
    "Use language conditioning."
)
flags.DEFINE_integer(
    "num_edit_samples",
    4,
    "Number of edit samples to use for editing the actions.",
)
flags.DEFINE_integer(
    "num_actions_to_sample",
    4,
    "Number of actions to sample for the policy.",
)
flags.DEFINE_integer(
    "online_trajectory_collection_frequency",
    10, # Every 10 update steps, collect 1 trajectory from environment #
    "Frequency of online trajectory collection.",
)
flags.DEFINE_integer(
    "critic_warmup_steps",
    100, # Warmup critic for 100 update steps #
    "Number of steps to warmup critic.",
)
flags.DEFINE_bool(
    "final_step_sparse_reward",
    False,
    "Use sparse reward for the final step.",
)
flags.DEFINE_bool(
    "use_wrist_view",
    True,
    "Use Wrist view camera."
)
flags.DEFINE_string(
    "params_path",
    None,
    "Path to the critic parameters to load.",
)
flags.DEFINE_bool(
    "filter_successful_trajectories",
    False,
    "Filter successful trajectories.",
)
flags.DEFINE_bool(
    "base_actions",
    False,
    "Use base policy actions instead of edited actions.",
)
flags.DEFINE_string(
    "pi_config_name",
    "pi05_libero_custom_low_mem_ep5",
    "Name of the PI config to use.",
)
flags.DEFINE_float(
    "perturbation_epsilon",
    -1.0,
    "Exploration noise epsilon for action perturbation. If > 0, adds Gaussian noise to actions.",
)

# 2: 07 2 13
BASE_POLICY_TYPE_TO_CLASS = {
    BasePolicyTypes.OpenVLA: OpenVLAAgent,
    BasePolicyTypes.DDPM: DDPMBCAgent,
    BasePolicyTypes.AutoRegressiveTransformer: AutoRegressiveTransformerAgent,
    BasePolicyTypes.Pi0: PiPolicy
}

devices = jax.local_devices()
# def shard_batch(batch, sharding):
#     return jax.tree_map(lambda x: jax.device_put(x, sharding), batch)
def shard_batch(batch, base_sharding):
    def shard_array(x):
        # Build a sharding spec matching the array's rank
        sharding_shape = (len(devices),) + (1,) * (x.ndim - 1)
        sharding = base_sharding.reshape(sharding_shape)
        return jax.device_put(x, sharding)

    return jax.tree_map(shard_array, batch)

def sanitize_obs(obs):
    out = {}
    for k, v in obs.items():
        if isinstance(v, (np.ndarray, jnp.ndarray)):
            out[k] = v
        # skip strings, lists, python objects
    return out


def get_policy_fn(
    agent,
    rng: jax.random.PRNGKey,
    timer: Timer | None = None,
    debug_mode: bool = False,
    use_base_actions: bool = False,
    perturbation_epsilon: float = -1.0,
) -> Callable[[Data], np.ndarray]:
    # Validate perturbation_epsilon
    if perturbation_epsilon > 0:
        assert perturbation_epsilon < 1.0, f"perturbation_epsilon must be < 1.0, got {perturbation_epsilon}"
    
    def policy_fn(observations: Data, *args, **kwargs) -> np.ndarray:
        if not isinstance(observations, dict):
            observations = {"state": observations}
        if "state" in observations:
            obs_ndim = observations["state"].ndim
        else:
            assert "proprio" in observations
            obs_ndim = observations["proprio"].ndim
        
        if use_base_actions:
            out_dict = jax.device_get(
                agent.sample_base_actions(
                    observations, *args, **kwargs, timer=timer, output_action_chunk=True
                )
            )
        else:
            out_dict = jax.device_get(
                agent.sample_actions(
                    observations, *args, **kwargs, timer=timer, output_action_chunk=True, use_deterministic_actions=True
                )
            )
        
        # Apply action perturbation if perturbation_epsilon > 0
        if perturbation_epsilon > 0 and 'action' in out_dict:
            actions = out_dict['action']
            # Get rng from kwargs (supplied by supply_rng wrapper)
            rng_key = kwargs.get('rng', None)
            if rng_key is not None:
                rng_key, rng_exploration = jax.random.split(rng_key)
                kwargs['rng'] = rng_key  # Update rng for next call
                
                # Normalize actions using agent's normalization (expects batch dimension)
                # Actions come as (action_horizon, action_dim), add batch dim
                actions_normalized = agent.actor.norm_actions(actions[None, ...])
                actions_normalized = actions_normalized[0]  # Remove batch dim
                
                # Generate exploration noise
                exploration_noise = jax.random.normal(
                    rng_exploration, actions_normalized.shape) * perturbation_epsilon
                exploration_noise = jnp.clip(exploration_noise, -1.0, 1.0)
                
                # Add noise and clip to [0, 1]
                actions_normalized = actions_normalized + exploration_noise
                actions_normalized = jnp.clip(actions_normalized, -1.0, 1.0)
                
                # Unnormalize using agent's unnormalization (expects batch dimension)
                actions = agent.actor.unnorm_actions(actions_normalized[None, ...])
                actions = actions[0]  # Remove batch dim
                
                # Update actions in output dict
                out_dict['action'] = jax.device_get(actions)

        return out_dict

    policy_fn = supply_rng(policy_fn, rng=rng)

    return policy_fn

def set_batch_masks(
    batch: Batch, environment_name: str, reward_bias: float, reward_scale: float
) -> Batch:
    """Environment-specific mask setting."""
    if "maze" in environment_name or environment_name == "real_robot" or "libero" in environment_name:
        # Assumes sparse rewards, mask should be 0 only at success
        success_reward = 1.0 * reward_scale + reward_bias
    elif "kitchen" in environment_name or "calvin" in environment_name:
        # Assumes 0-4 rewards, mask should be 0 only at 4
        success_reward = 4.0 * reward_scale + reward_bias
    else:
        raise NotImplementedError
    batch["masks"] = (batch["rewards"] != success_reward).astype(np.float32)
    return batch


@jax.jit
def resize_images_to_100x100(images):
    batch_size = images.shape[0]
    return jax.image.resize(images, (batch_size, 100, 100, 3), method="cubic")

def create_gif_from_images(images: List[np.ndarray], output_path: str, duration: int = 100):
    """
    Create a GIF from a list of images.

    Args:
        images: List of numpy arrays representing images (H, W, 3) in uint8 format
        output_path: Path to save the GIF file
        duration: Duration of each frame in milliseconds
    """
    if not images:
        raise ValueError("No images provided to create GIF")

    # Convert numpy arrays to PIL Images
    pil_images = []
    for timestep, img in enumerate(images):
        # Ensure the image is in the correct format
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255.0).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

        # Handle different image dimensions
        if img.ndim == 4:  # Batch dimension present
            img = img[0]

        # Convert to PIL Image
        pil_img = Image.fromarray(img)

        # Add timestep text overlay
        draw = ImageDraw.Draw(pil_img)
        text = f"Timestep: {timestep}"

        # Try to use a truetype font, fall back to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 20)
            except:
                font = ImageFont.load_default()

        # Get text bounding box for background
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Draw semi-transparent background for text
        padding = 5
        background_box = [
            5,
            5,
            5 + text_width + 2 * padding,
            5 + text_height + 2 * padding
        ]
        draw.rectangle(background_box, fill=(0, 0, 0, 180))

        # Draw text in white
        draw.text((5 + padding, 5 + padding), text, fill=(255, 255, 255), font=font)

        pil_images.append(pil_img)

    # Save as GIF
    pil_images[0].save(
        output_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=duration,
        loop=0,
    )
    print(f"✅ GIF saved to {output_path}")

def train_agent(_):
    # breakpoint()
    
    if FLAGS.debug:
        breakpoint()
        # Disabling jit might be useful for debugging
        # jax.config.update("jax_disable_jit", True)

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    os.environ["WANDB__SERVICE_WAIT"] = "300"
    os.environ["WANDB_INIT_TIMEOUT"] = "120"
    wandb.require("core")
    devices = jax.local_devices()
    num_devices = len(devices)
    assert FLAGS.config.batch_size % num_devices == 0

    # Get PI config #
    pi_config = get_config(FLAGS.pi_config_name)
    # breakpoint()
    pi_config.fsdp_devices = 1 # Try out with model parallel
    pi_config.exp_name = FLAGS.wandb_experiment_name if FLAGS.wandb_experiment_name else "evaluate_traj"
    pi_config.overwrite = True

    if FLAGS.num_trajectories_to_collect is not None:
        FLAGS.config.num_eval_episodes = FLAGS.num_trajectories_to_collect

    wandb_logger = None
    save_dir = tf.io.gfile.join(
        os.path.abspath(FLAGS.config.save_dir),
    )
    os.makedirs(save_dir, exist_ok=True)
    FLAGS.config.wandb_enabled = False

    # Create environment and dataset
    action_space = None
    offline_dataset_size = None

    # breakpoint()
    
    ####### Dataset and evironment setup ###########
    if FLAGS.environment_name=="libero":
        assert FLAGS.config.image_observations
        from jaxrl_m.envs.libero import (
            get_libero_config,
            get_libero_env,
            get_libero_tfrecord_dataset,
        )

        # LOG: Data stored in hf_cache on babel, can access on common path; Loads for example, 'libero_10' path as tf_records #
        # LOG: `dataset` will store the offline dataset to train on #
        # Iterates over to yield a dict with bunch of keys which can include observations, actions, rewards, masks, next_observations, etc. #
        # dataset = get_libero_tfrecord_dataset(
        #     tfrecord_regexp=FLAGS.config.libero_tfrecord_regexp, use_wrist_view=FLAGS.use_wrist_view, 
        #     use_language=FLAGS.use_lang, config=pi_config, is_pi=True, **FLAGS.config.dataset_kwargs,
        #     task_name=FLAGS.task_name, 
        #     final_step_sparse_reward=FLAGS.final_step_sparse_reward,
        #     filter_successful_trajectories=FLAGS.filter_successful_trajectories,
        # )
        # breakpoint()
        libero_config = get_libero_config()

        train_env = get_libero_env(cfg=libero_config, task_name=FLAGS.task_name, is_pi=True)
        if FLAGS.num_parallel_envs > 1:
            num_parallel_envs = FLAGS.num_parallel_envs
            task_name = FLAGS.task_name
            eval_env = gym.vector.AsyncVectorEnv(
                [
                    lambda: get_libero_env(
                        cfg=libero_config, task_id = ind*num_parallel_envs, task_name=task_name, is_pi=True,
                    )
                    for ind in range(num_parallel_envs)
                ],
                context="forkserver", shared_memory=False, # the default "fork" is incompatible with JAX
            )
        else:
            eval_env = get_libero_env(cfg=libero_config, task_name=FLAGS.task_name, is_pi=True)
    else:
       raise NotImplementedError

    if action_space is None:
        action_space = train_env.action_space
    assert action_space.high.ndim == 1, action_space.shape
    # Create replay buffer
    # LOG: Libero comes under this for now #
    if FLAGS.config.image_observations:
        # tf.io.gfile.makedirs(tf.io.gfile.join(save_dir, "image_replay_buffer"))
        assert not tf.io.gfile.exists(
            tf.io.gfile.join(save_dir, "image_replay_buffer", "episode_0.tfrecord")
        ), f"Image replay buffer already exists! ({tf.io.gfile.join(save_dir, 'image_replay_buffer', 'episode_0.tfrecord')})"
        state_replay_buffer = None

    rng = jax.random.PRNGKey(FLAGS.seed)
    # we shard the leading dimension (batch dimension) accross all devices evenly
    # sharding = jax.sharding.PositionalSharding(devices)
    sharding = jax.sharding.PositionalSharding(devices)
    # Create data iterators
    # LOG: Offline dataset #
    # offline_train_iterator = dataset.iterator(
    #     batch_size=FLAGS.config.agent_kwargs.batch_size
    # )
    # Online dataset/buffer #
    # Online iterators will be set when switching to online training.
    online_train_iterator = None
    #########################################################
    # Dataset created now in `dataset` and environment created now in `train_env` #
    # breakpoint()

    ### Sharding Data ###
    # example_batch = next(offline_train_iterator)
    # example_batch = shard_batch(example_batch, sharding) # DO NOT shard here, will be handled in the expo agent forward passes
    
    ### Create trajectory sampler ###
    data_collection_trajectory_sampler = TrajSampler(
        train_env,
        clip_action=FLAGS.clip_action,
        reward_scale=FLAGS.reward_scale,
        reward_bias=FLAGS.reward_bias,
        max_traj_length=FLAGS.config.get("max_episode_steps", 1000),
        action_horizon=pi_config.model.action_horizon,
        # action_horizon=1,
    )

    ### Create agent #
    rng, construct_rng = jax.random.split(rng)
    
    # Build agent kwargs from config and flags
    # Remove keys that are already passed explicitly to avoid conflicts
    agent_kwargs = dict(FLAGS.config.agent_kwargs)
    for key in ['config', 'seed', 'batch_size', 'rng', 'params_path']:
        agent_kwargs.pop(key, None)
    
    agent = create_agent(
        agent_name=FLAGS.agent_name,
        config=pi_config,
        seed=FLAGS.seed,
        batch_size=FLAGS.config.batch_size,
        rng=construct_rng,
        params_path=FLAGS.params_path,
        **agent_kwargs,
    )
    # breakpoint()

    timer = Timer()

    ### Evaluation Setup ###
    env_data_collection_policy_fn = None  # Will get set later
    rng, eval_policy_fn_key = jax.random.split(rng)
    eval_policy_fn = get_policy_fn(
        agent=agent,
        rng=eval_policy_fn_key,
        timer=timer,
        use_base_actions=FLAGS.base_actions,
        perturbation_epsilon=FLAGS.perturbation_epsilon,
    )
    #########################################################

    for i in range(1):
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
            if "libero" not in FLAGS.environment_name: 
                trajectories = evaluate_with_trajectories_vectorized(
                    eval_policy_fn,
                    eval_env,
                    FLAGS.config.num_eval_episodes,
                )
            else:
                if FLAGS.num_parallel_envs != 1:
                    trajectories = evaluate_with_trajectories_vectorized(
                    eval_policy_fn,
                    eval_env,
                    FLAGS.config.num_eval_episodes,
                )
                else:
                    eval_policy_fn = get_policy_fn(
                        agent=agent,
                        rng=eval_policy_fn_key,
                        timer=timer,
                        debug_mode=False,
                        use_base_actions=FLAGS.base_actions,
                        perturbation_epsilon=FLAGS.perturbation_epsilon,
                    )
                    trajectories, _ = evaluate_with_trajectories_libero(
                    eval_policy_fn,
                    eval_env,
                    FLAGS.config.num_eval_episodes,
                    action_horizon=pi_config.model.action_horizon,
                    # action_horizon=1,
                    use_full_horizon_for_refill=False,
                )
            
            # breakpoint()

            if (FLAGS.environment_name == "calvin" or FLAGS.environment_name =='libero') and FLAGS.config.save_video:
                # Create success/failure folders
                success_dir = os.path.join(FLAGS.config.save_dir, "success")
                failure_dir = os.path.join(FLAGS.config.save_dir, "failure")
                os.makedirs(success_dir, exist_ok=True)
                os.makedirs(failure_dir, exist_ok=True)
                
                success_episode_ids = []
                failure_episode_ids = []
                
                for j, traj in enumerate(trajectories):
                    trajectory_return = np.sum(traj["reward"])
                    # Determine success based on reward (success typically means reward == 1.0 in Libero)
                    is_success = trajectory_return >= 1.0
                    
                    if is_success:
                        success_episode_ids.append(j)
                        save_dir_traj = success_dir
                    else:
                        failure_episode_ids.append(j)
                        save_dir_traj = failure_dir
                    
                    ind_traj = []
                    traj_return_accum = 0
                    for transition, reward in zip(
                        traj["observation"], traj["reward"]
                    ):
                        assert transition["image"].shape[-1] == 3
                        if len(transition["image"].shape) == 4:
                            transition["image"] = transition["image"][0]
                        image = transition["image"]
                        traj_return_accum += reward
                        image = np.ascontiguousarray(image) 
                        frame = cv2.putText(
                            image,
                            f"reward: {reward}. return: {traj_return_accum}",
                            (10, 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.3,
                            (0, 0, 0),
                            1,
                        )
                        ind_traj.append(frame)
                    
                    create_gif_from_images(ind_traj, os.path.join(save_dir_traj, f'rollout_{j}.gif'))
                
                # Save CSV with evaluation metrics
                import csv
                csv_path = os.path.join(FLAGS.config.save_dir, "evaluation_results.csv")
                with open(csv_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['metric', 'value'])
                    writer.writerow(['num_eval_episodes', FLAGS.config.num_eval_episodes])
                    writer.writerow(['success_rate', len(success_episode_ids) / FLAGS.config.num_eval_episodes])
                    writer.writerow(['success_episode_ids', ','.join(map(str, success_episode_ids))])
                    writer.writerow(['failure_episode_ids', ','.join(map(str, failure_episode_ids))])
                print(f"✅ Evaluation results saved to {csv_path}")
                
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

            # breakpoint()

            print("Eval metrics: ", eval_metrics)


if __name__ == "__main__":
    app.run(train_agent)