"""Script for offline to online RL."""

# Try increasing the number of open files limit
import resource
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(65535, hard), hard))

import os
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import pickle
from absl import app, flags

from ml_collections import config_flags
import functools

from jaxrl_m.agents import agents
from jaxrl_m.agents.continuous.auto_regressive_transformer import (
    AutoRegressiveTransformerAgent,
)
from jaxrl_m.agents.continuous.base_policy import BasePolicyTypes
from jaxrl_m.agents.continuous.ddpm_bc import DDPMBCAgent
from jaxrl_m.agents.continuous.openvla import OpenVLAAgent
from jaxrl_m.agents.continuous.pi_0 import PiPolicy
from jaxrl_m.common.evaluation import supply_rng
from jaxrl_m.common.traj import TrajSampler
from jaxrl_m.common.typing import Data
from jaxrl_m.data.image_replay_buffer_pi import (
    save_trajectory_as_tfrecord,
)
from jaxrl_m.utils.timer_utils import Timer
from jaxrl_m.agents.continuous.pi_vlm_cached import (
    create_agent,
    PiResidualTD3Cache,
    PiResidualPPOCache,
)
from jaxrl_m.utils.expo_utils import calc_mc_return_fn

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
    "Path to the parameters to load.",
)
flags.DEFINE_bool(
    "filter_successful_trajectories",
    False,
    "Filter successful trajectories.",
)
flags.DEFINE_integer(
    "num_diffusion_samples",
    1,
    "Number of diffusion samples to use to output `diffusion_actions` from base policy, useful for umap visualization/analysis; for actual training, use 1",
)
flags.DEFINE_bool(
    "normalize_diffusion_actions",
    False,
    "Normalize diffusion actions before returning.; For actual training, use False, use True for umap visualization/analysis",
)
flags.DEFINE_integer(
    "num_trajectories_to_collect",
    200,
    "Number of trajectories to collect from environment.",
)
flags.DEFINE_string(
    "pi_config_name",
    "pi05_libero_custom_low_mem",
    "Name of the PI config to use.",
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
    num_diffusion_samples: int = 1,
    normalize_diffusion_actions: bool = False,
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
            agent.sample_base_actions(
                observations, *args, **kwargs, timer=timer, output_action_chunk=True, num_diffusion_samples=num_diffusion_samples, normalize_diffusion_actions=normalize_diffusion_actions,
            )
        )
        # breakpoint()

        return out_dict

    policy_fn = supply_rng(policy_fn, rng=rng)

    return policy_fn

def get_vlm_output_fn(
    agent,
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
            agent.get_vlm_output(
                observations, *args, **kwargs, timer=timer, output_action_chunk=True, debug_mode=debug_mode
            )
        )
        # breakpoint()

        return out_dict

    policy_fn = supply_rng(policy_fn, rng=rng)

    return policy_fn

def train_agent(_):

    if FLAGS.debug:
        breakpoint()

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    devices = jax.local_devices()
    num_devices = len(devices)
    assert FLAGS.config.batch_size % num_devices == 0

    # Get PI config #
    pi_config = get_config(FLAGS.pi_config_name)
    # breakpoint()
    pi_config.fsdp_devices = 1 # Try out with model parallel
    pi_config.exp_name = "sample_base_traj"
    pi_config.overwrite = True

    wandb_logger = None
    save_dir = tf.io.gfile.join(
        os.path.abspath(FLAGS.config.save_dir),
    )
    FLAGS.config.wandb_enabled = False

    # Create environment and dataset
    action_space = None
    offline_dataset_size = None
    
    ####### Dataset and evironment setup ###########
    if FLAGS.environment_name=="libero":
        assert FLAGS.config.image_observations
        from jaxrl_m.envs.libero import (
            get_libero_config,
            get_libero_env,
            get_libero_tfrecord_dataset,
        )
        libero_config = get_libero_config()

        train_env = get_libero_env(cfg=libero_config, task_name=FLAGS.task_name, is_pi=True)
    else:
       raise NotImplementedError

    if action_space is None:
        action_space = train_env.action_space
    assert action_space.high.ndim == 1, action_space.shape

    if not tf.io.gfile.exists(
            tf.io.gfile.join(save_dir, "image_replay_buffer", "episode_0.tfrecord")
        ):
        tf.io.gfile.makedirs(tf.io.gfile.join(save_dir, "image_replay_buffer"))
    
    state_replay_buffer = None

    rng = jax.random.PRNGKey(FLAGS.seed)
    #########################################################
    
    ### Create trajectory sampler ###
    data_collection_trajectory_sampler = TrajSampler(
        train_env,
        clip_action=FLAGS.clip_action,
        reward_scale=FLAGS.reward_scale,
        reward_bias=FLAGS.reward_bias,
        max_traj_length=FLAGS.config.get("max_episode_steps", 1000),
        action_horizon=pi_config.model.action_horizon,
    )

    ### Create EXPO agent #
    # LOG: sharded batch is used to calibrate batch size in `create` method of `ExpoPiLearner` class #
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
    #########################################################

    


    # TODO: Remove hardcode and init with flags appropriately #
    num_trajectories_to_collect = FLAGS.num_trajectories_to_collect
    online_trajectories_in_buffer = len(os.listdir(os.path.join(save_dir, "image_replay_buffer")))

    for i in range(1):

        ### Collect Trajectories ###
        if i % FLAGS.online_trajectory_collection_frequency == 0:
            print("Collecting trajectories...")
            trajectories = []
            q_vs_mc_returns_vals = []
            for traj_index in range(num_trajectories_to_collect):
                print("Traj Index: ", traj_index)

                # Fresh RNG keys per trajectory, with independent keys for policy and VLM
                data_collection_rng_key, rng = jax.random.split(rng)
                policy_rng, vlm_rng = jax.random.split(data_collection_rng_key)

                env_data_collection_policy_fn = get_policy_fn(
                    agent=agent,
                    rng=policy_rng,
                    timer=timer,
                    debug_mode=True,
                    num_diffusion_samples=FLAGS.num_diffusion_samples,
                    normalize_diffusion_actions=FLAGS.normalize_diffusion_actions,
                )
                vlm_output_fn = get_vlm_output_fn(
                    agent=agent,
                    rng=vlm_rng,
                    timer=timer,
                )

                fns_dict = {
                    "policy_fn": env_data_collection_policy_fn,
                    "vlm_output_fn": vlm_output_fn,
                }
                timer.tick("trajectory_sampling_time")
                trajs, _q_vs_mc_returns_vals = data_collection_trajectory_sampler.sample(
                    fns_dict=fns_dict,
                    num_episodes=1,
                    replay_buffer=state_replay_buffer,
                    calc_mc_return_fn=functools.partial(calc_mc_return_fn, discount=FLAGS.config.agent_kwargs.discount, reward_bias=FLAGS.reward_bias),
                    store_max_trajectory_reward=True,
                    terminate_on_success=FLAGS.config.get(
                        "early_terminate_on_success", False
                    ),
                )
                traj = trajs[0]
                # breakpoint()
                _q_vs_mc_returns_vals = _q_vs_mc_returns_vals[0]
                timer.tock("trajectory_sampling_time")
                print(timer.get_total_times(reset=False))
                # LOG: `traj` statistics #
                # breakpoint()
                trajectories.append(traj)
                q_vs_mc_returns_vals.append(_q_vs_mc_returns_vals)
                # breakpoint()

                if FLAGS.config.image_observations:
                    # Save trajectory as tfrecord
                    save_trajectory_as_tfrecord(
                        trajectory=traj,
                        path=tf.io.gfile.join(
                            save_dir,
                            "image_replay_buffer",
                            f"episode_{online_trajectories_in_buffer}.tfrecord",
                        ),
                    )
                online_trajectories_in_buffer += 1

if __name__ == "__main__":
    app.run(train_agent)