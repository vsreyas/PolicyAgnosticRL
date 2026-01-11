"""Script for offline to online RL."""

# Try increasing the number of open files limit
import resource
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(65535, hard), hard))


import os
import time
from typing import Any, Callable, Dict, List, Optional, Union

import cv2
import flax
import gym
import jax
# jax.config.update("jax_log_compiles", True)
# jax.config.update("jax_explain_cache_misses", True)
# jax.config.update("jax_traceback_filtering", "off")  # more context in logs

import jax.numpy as jnp
import numpy as np
import seaborn as sns
import tensorflow as tf
import io
from PIL import Image
import pickle
import gc

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
from jaxrl_m.agents.continuous.expo_pi_cache import ExpoPiLearnerCache, compute_q, compute_q_all
from jaxrl_m.utils.expo_utils import calc_mc_return_fn
from jaxrl_m.envs.libero import StepTimeout, time_limit, STEP_TIME_LIMIT

try:
    from jax_smi import initialise_tracking  # type: ignore

    initialise_tracking()
except ImportError:
    pass

from openpi.training.config import get_config


print("\n\n\n IMPORTS DONE \n\n\n")

FLAGS = flags.FLAGS

flags.DEFINE_string("environment_name", "", "Environment name.")
flags.DEFINE_string("wandb_project_name", "PI-0.5-finetuning", "WandB project name.") #"PA-RL""debug"
flags.DEFINE_string("wandb_experiment_name", "", "WandB experiment name.")
flags.DEFINE_string("wandb_group", "", "WandB group.")
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
    "critic_params_path",
    None,
    "Path to the critic parameters to load.",
)
flags.DEFINE_bool(
    "filter_successful_trajectories",
    False,
    "Filter successful trajectories.",
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
    agent: ExpoPiLearnerCache,
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

def get_vlm_output_fn(
    agent: ExpoPiLearnerCache,
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
        vlm_output = jax.device_get(
            agent.get_vlm_output(
                observations, *args, **kwargs, timer=timer, output_action_chunk=True, debug_mode=debug_mode
            )
        )
        # breakpoint()

        return vlm_output

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

def plot_q_values_over_trajectory_time_step(
    trajectories: List[Dict[str, List[Union[np.ndarray, Dict[str, np.ndarray]]]]],
    critic_agent,
    sharding: jax.sharding.Sharding,
):
    trajectories = [trajectories[0]]  # only plot the first trajectory
    if isinstance(trajectories[0]["observation"][0], dict):
        trajectories[0]['observation'][0] = sanitize_obs(trajectories[0]['observation'][0])
        observations = [
            {
                key: np.array([obs[key] for obs in trajectory["observation"]])
                for key in trajectory["observation"][0].keys()
            }
            for trajectory in trajectories
        ]
    else:
        observations = [
            shard_batch(jnp.array(trajectory["observation"]), sharding)
            for trajectory in trajectories
        ]

    actions = [
        shard_batch(jnp.array(trajectory["action"]), sharding)
        for trajectory in trajectories
    ]

    q_values = []
    for trajectory_index in range(len(trajectories)):
        q_values.append(
            critic_agent.forward_critic(
                observations[trajectory_index],
                actions[trajectory_index],
                jax.random.PRNGKey(0),
            ).mean(axis=0)
        )
    q_values = jnp.stack(q_values, axis=0).mean(axis=0)
    assert q_values.shape == (len(trajectories[0]["observation"]),)

    # Plot the q-values over the trajectory time step using seaborn, make it look nice
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    plot = sns.lineplot(
        x=np.arange(len(q_values)),
        y=q_values,
        color="blue",
        linewidth=2.5,
    )
    plot.set_title("Q-values over trajectory time step")
    plot.set_xlabel("Time step")
    plot.set_ylabel("Q-value")

    return plot


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
    pi_config = get_config("pi05_libero_custom_low_mem")
    # breakpoint()
    pi_config.fsdp_devices = 1 # Try out with model parallel
    pi_config.exp_name = FLAGS.wandb_experiment_name
    pi_config.overwrite = True

    # LOG: WANDB setup #
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
        )
        save_dir = tf.io.gfile.join(
            (
                os.path.abspath(FLAGS.config.save_dir)
                if "gs://" not in FLAGS.config.save_dir
                else FLAGS.config.save_dir
            ),
            # wandb_logger.config.project,
            # wandb_logger.config.exp_descriptor,
            f"seed_{FLAGS.seed}",
        )
    else:
        wandb_logger = None
        save_dir = tf.io.gfile.join(
            os.path.abspath(FLAGS.config.save_dir),
        )
        FLAGS.config.wandb_enabled = False

    # breakpoint()

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
        dataset = get_libero_tfrecord_dataset(
            tfrecord_regexp=FLAGS.config.libero_tfrecord_regexp, use_wrist_view=FLAGS.use_wrist_view, 
            use_language=FLAGS.use_lang, config=pi_config, is_pi=True, **FLAGS.config.dataset_kwargs,
            task_name=FLAGS.task_name, 
            final_step_sparse_reward=FLAGS.final_step_sparse_reward,
            # filter_successful_trajectories=FLAGS.filter_successful_trajectories,
            filter_successful_trajectories=False,
            use_reverse_data_paths=False,
            alpha=1.0,
            scale_success_reward=True,
            drop_images_from_output=True, # Do not need it as we are caching things are trajectory generation time #
        )
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
        # assert not tf.io.gfile.exists(
        #     tf.io.gfile.join(save_dir, "image_replay_buffer", "episode_0.tfrecord")
        # ), f"Image replay buffer already exists! ({tf.io.gfile.join(save_dir, 'image_replay_buffer', 'episode_0.tfrecord')})"
        image_replay_buffer = None  # Will be created when switching to online training.
        state_replay_buffer = None

    rng = jax.random.PRNGKey(FLAGS.seed)
    # we shard the leading dimension (batch dimension) accross all devices evenly
    # sharding = jax.sharding.PositionalSharding(devices)
    sharding = jax.sharding.PositionalSharding(devices)
    # Create data iterators
    # LOG: Offline dataset #
    offline_train_iterator = dataset.iterator(
        batch_size=FLAGS.config.agent_kwargs.batch_size
    )
    # Online dataset/buffer #
    # Online iterators will be set when switching to online training.
    online_train_iterator = None
    #########################################################
    # Dataset created now in `dataset` and environment created now in `train_env` #
    # breakpoint()

    ### Sharding Data ###
    example_batch = next(offline_train_iterator)
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

    ### Create EXPO agent #
    # LOG: sharded batch is used to calibrate batch size in `create` method of `ExpoPiLearner` class #
    rng, construct_rng = jax.random.split(rng)
    
    if FLAGS.critic_params_path is not None:
        critic_params = pickle.load(open(FLAGS.critic_params_path, 'rb'))['critic_params']
    else:
        critic_params = None
    
    agent = ExpoPiLearnerCache.create(
        config=pi_config,
        seed=FLAGS.seed,
        # observations=example_batch,
        batch_size=FLAGS.config.batch_size,
        rng=construct_rng,
        N=FLAGS.num_actions_to_sample,
        n_edit_samples=FLAGS.num_edit_samples,
        critic_params=critic_params,
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
    )
    #########################################################

    


    # TODO: Remove hardcode and init with flags appropriately #
    num_trajectories_to_collect = 5
    online_env_steps = 0
    online_trajectories_added = 0
    env_recreation_frequency = 3
    env_recreation_count = 0

    ### EXPO agent training ###
    ### Online training ###
    for i in range(FLAGS.num_offline_epochs + FLAGS.num_online_epochs + 1):
        online_env_steps_this_epoch = 0

        if i >= FLAGS.num_offline_epochs and FLAGS.num_online_epochs > 0:
            timer.tick("online_iter_total")
            # logging.info("Switching to online training...")

            ### Collect Trajectories ###
            if i % FLAGS.online_trajectory_collection_frequency == 0:
                env_recreation_count += 1
                print("Collecting trajectories...")
                data_collection_rng_key, rng = jax.random.split(rng)
                
                env_data_collection_policy_fn = get_policy_fn(
                    agent=agent,
                    rng=data_collection_rng_key,
                    timer=timer,
                    # debug_mode=debug_mode,
                    debug_mode=False,
                )
                vlm_output_fn = get_vlm_output_fn(
                    agent=agent,
                    rng=data_collection_rng_key,
                    timer=timer,
                )

                if env_recreation_count % env_recreation_frequency == 0:
                    train_env.env.close()
                    print("Recreating environment...")
                    del train_env
                    import gc; gc.collect()
                    train_env = get_libero_env(cfg=libero_config, task_name=FLAGS.task_name, is_pi=True)
                    data_collection_trajectory_sampler = TrajSampler(
                        train_env,
                        clip_action=FLAGS.clip_action,
                        reward_scale=FLAGS.reward_scale,
                        reward_bias=FLAGS.reward_bias,
                        max_traj_length=FLAGS.config.get("max_episode_steps", 1000),
                        action_horizon=pi_config.model.action_horizon,
                        # action_horizon=1,
                    )
                    env_recreation_count = 0

                trajectories = []
                q_vs_mc_returns_vals = []
                for traj_index in range(num_trajectories_to_collect):
                    timer.tick("trajectory_sampling_time")

                    sampled_trajectories_successfully = False
                    while not sampled_trajectories_successfully:
                        # try:
                        #     with time_limit(STEP_TIME_LIMIT):
                        #         obs, reward, done, info = self.env.step(action)# may STILL not interrupt if stuck in native code
                        # except StepTimeout:
                        #     raise StepTimeout("env.step() timed out")
                        # try:

                        try:
                            with time_limit(STEP_TIME_LIMIT):
                                trajs, _q_vs_mc_returns_vals = data_collection_trajectory_sampler.sample(
                                    env_data_collection_policy_fn,
                                    vlm_output_fn,
                                    num_episodes=1,
                                    replay_buffer=state_replay_buffer,
                                    calc_mc_return_fn=functools.partial(calc_mc_return_fn, discount=FLAGS.config.agent_kwargs.discount, reward_bias=FLAGS.reward_bias),
                                    store_max_trajectory_reward=True,
                                    terminate_on_success=False,
                                )
                                sampled_trajectories_successfully = True
                                break
                        except:
                            print("Trajectory sampling timed out")
                            del train_env
                            import gc; gc.collect()
                            train_env = get_libero_env(cfg=libero_config, task_name=FLAGS.task_name, is_pi=True)
                            data_collection_trajectory_sampler = TrajSampler(
                                train_env,
                                clip_action=FLAGS.clip_action,
                                reward_scale=FLAGS.reward_scale,
                                reward_bias=FLAGS.reward_bias,
                                max_traj_length=FLAGS.config.get("max_episode_steps", 1000),
                                action_horizon=pi_config.model.action_horizon,
                                # action_horizon=1,
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
                                f"episode_{online_trajectories_added}.tfrecord",
                            ),
                        )
                    online_trajectories_added += 1
                    online_env_steps_this_epoch += len(traj["rewards"])
                    
                    # breakpoint()
                
                # Get trajectory statistics
                # LOG: Log some statistics for the collected trajectories #
                mean_trajectory_return = np.mean(
                    [np.sum(t["rewards"]) for t in trajectories]
                )
                mean_trajectory_length = np.mean([len(t["rewards"]) for t in trajectories])
                mean_max_reward = np.mean([np.max(t["rewards"]) for t in trajectories])
                if wandb_logger is not None:
                    wandb_logger.log(
                        {
                            "train_env": {
                                "mean_trajectory_return": mean_trajectory_return,
                                "mean_trajectory_length": mean_trajectory_length,
                                "mean_max_reward": mean_max_reward,
                            },
                            "online_env_steps": online_env_steps,
                            "online_trajectories_added": online_trajectories_added,
                        },
                        step=i,
                    )
                
                del trajectories, q_vs_mc_returns_vals
                import gc; gc.collect()
                
                # breakpoint()

                # Finished collecting trajectories
                # LOG: Construct buffers using the trajectories #
                # LOG: Looks like two iterators are constructed, one for online trajectories and `dataset` from previous definition, for offline dataset #
                online_env_steps += online_env_steps_this_epoch
                # Recreate the image replay buffer iterator to include the new trajectories
                timer.tick("recreate_image_replay_buffer_iterator")
                data_paths = glob_to_path_list(
                    tf.io.gfile.join(save_dir, "image_replay_buffer", "*.tfrecord")
                )
                #########################
                
                # Do some cleanups to avoid hangs #
                online_train_iterator = None
                image_replay_buffer = None
                import gc; gc.collect()

                #########################################################
                # breakpoint()
                image_replay_buffer = ImageReplayBufferPi(
                    data_paths=data_paths,
                    seed=FLAGS.seed,
                    train=True,
                    task_name=FLAGS.task_name,
                    use_wrist_view=FLAGS.use_wrist_view, 
                    use_language=FLAGS.use_lang, config=pi_config,
                    final_step_sparse_reward=False, # Use rewards from environment and DO NOT override with sparse 0/1 rewards at final step #
                    filter_successful_trajectories=False, # Use success buffer #
                    use_reverse_data_paths=True,
                    alpha=1.0,
                    scale_success_reward=True,
                    drop_images_from_output=True,
                    **FLAGS.config.image_replay_buffer_kwargs,
                )
                timer.tock("recreate_image_replay_buffer_iterator")
                #########################################################

                # Update online iterator #
                online_train_iterator = image_replay_buffer.iterator(
                    batch_size=FLAGS.config.agent_kwargs.batch_size
                )
            
            # breakpoint()
            # offline_batch['diffusion_actions'] = offline_batch['actions']
            # offline_batch['next_diffusion_actions'] = offline_batch['next_actions']

            rng, rng_update = jax.random.split(rng)

            # Critic warmup on only offline data as these are reliable actions #
            if i < FLAGS.critic_warmup_steps:
                print("Critic warmup...Updating only critic")
                # batch = offline_batch
                batch = next(online_train_iterator)
                # breakpoint()
                # batch = set_batch_masks(
                #     batch, FLAGS.environment_name, FLAGS.reward_bias, FLAGS.reward_scale
                # )
                agent, info = agent.update(batch, utd_ratio=FLAGS.config.utd_ratio, timer=timer, update_only_critic=True, output_only_base_actions=True, seed=rng_update)
            else:
                # try:
                # Sample a batch from online and do update #
                # RLPD style online + offline update #
                timer.tick("batch_sampling_time")
                offline_batch = next(offline_train_iterator)
                online_batch = next(online_train_iterator)
                timer.tock("batch_sampling_time")
                # except StopIteration:
                #     # No successful trajectories in online buffer, construct full buffer #
                #     image_replay_buffer = ImageReplayBufferPi(
                #         data_paths=data_paths,
                #         seed=FLAGS.seed,
                #         train=True,
                #         task_name=FLAGS.task_name,
                #         use_wrist_view=FLAGS.use_wrist_view, 
                #         use_language=FLAGS.use_lang, config=pi_config,
                #         final_step_sparse_reward=False, # Use rewards from environment and DO NOT override with sparse 0/1 rewards at final step #
                #         filter_successful_trajectories=False, # Use success buffer #
                #         use_reverse_data_paths=True,
                #         **FLAGS.config.image_replay_buffer_kwargs,
                #     )
                #     online_train_iterator = image_replay_buffer.iterator(
                #         batch_size=FLAGS.config.agent_kwargs.batch_size
                #     )
                #     online_batch = next(online_train_iterator)
                
                # timer.tick("concatenate_batches_time")
                # batch = concatenate_batches([offline_batch, online_batch])
                # timer.tock("concatenate_batches_time")
                batch = online_batch
                # batch = online_batch
                # Do this as it cleanly handles termination/truncation for bootstrapping during critic update #
                # The function effectively sets mask as 0.0 only where reward == 1.0, so for unsuccessful trajectory, it will have 'dones' as 0.0 at end #
                # batch = set_batch_masks(
                #     batch, FLAGS.environment_name, FLAGS.reward_bias, FLAGS.reward_scale
                # )
                agent, info = agent.update(batch, utd_ratio=FLAGS.config.utd_ratio, timer=timer, seed=rng_update)
            
            # Log batch statistics #
            # breakpoint()
            batch_stats = {
                "batch_stats/batch_size": len(batch),
                "batch_stats/masks_mean": np.mean(batch["masks"]),
                "batch_stats/mc_returns_mean": np.mean(batch["mc_returns"]),
                "batch_stats/rewards_mean": np.mean(batch["rewards"]),
                "batch_stats/rewards_min": np.min(batch["rewards"]),
                "batch_stats/rewards_max": np.max(batch["rewards"]),
                "batch_stats/terminals_mean": np.mean(batch["terminals"]),
                "batch_stats/truncations_mean": np.mean(batch["truncates"]),
            }
            if wandb_logger is not None:
                wandb_logger.log(batch_stats, step=i)
            

            # Log training metrics #
            if wandb_logger is not None:
                wandb_logger.log(info, step=i)
            
            # breakpoint()
            
            timer.tock("online_iter_total")
            print(timer.get_total_times(reset=True))

            ### Evaluation ###
            if (
                (i + 1) % FLAGS.config.eval_interval == 0
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
                            )
                            trajectories, q_vs_mc_returns_vals = evaluate_with_trajectories_libero(
                            eval_policy_fn,
                            eval_env,
                            FLAGS.config.num_eval_episodes,
                            action_horizon=pi_config.model.action_horizon,
                            # action_horizon=1,
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
                            
                            save_rollout_gif(ind_traj, save_dir, step_i=i, rollout_j=j)
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
                        wandb_logger.log(eval_metrics, step=i)
                    
                    # # Log Q vs MC Returns #
                    # if q_vs_mc_returns_vals is not None:
                    #     qs_across_trajectories = []
                    #     for q_vs_mc_return_val in q_vs_mc_returns_vals: # Per trajectory
                    #         qs = []
                    #         for idx in range(len(q_vs_mc_return_val)):
                    #             vlm_output, action_sequence = q_vs_mc_return_val[idx]
                    #             params = agent.critic.params
                    #             # Prepare input for critic #
                    #             vlm_output = vlm_output.reshape(1, -1)
                    #             action_sequence = action_sequence.reshape(1, -1)
                    #             q_vals = compute_q_all(agent.critic.apply_fn, params, vlm_output, action_sequence)
                    #             # breakpoint()
                    #             qs.append(q_vals[0]) # Look at q1 specifically #
                    #         qs_across_trajectories.append(qs)
                    
                    #     # mc_returns = jax.tree_map(
                    #     #     lambda t: calc_return_to_go(
                    #     #         rewards=np.array(t["reward"]), # * FLAGS.reward_scale
                    #     #         # + FLAGS.reward_bias,
                    #     #         masks=1 - np.array(t["done"]),
                    #     #         gamma=FLAGS.config.agent_kwargs.discount,
                    #     #         push_failed_to_min="maze" in FLAGS.environment_name
                    #     #         or FLAGS.environment_name == "real_robot",
                    #     #         min_reward=FLAGS.reward_bias,
                    #     #     ),
                    #     #     trajectories,
                    #     #     is_leaf=lambda x: isinstance(
                    #     #         x, dict
                    #     #     ),  # only map over traj in trajs
                    #     # )
                    #     # initial_mc_returns = jax.tree_map(lambda t: t[0], mc_returns)

                    #     H = pi_config.model.action_horizon  # chunk size
                    #     gamma_step = FLAGS.config.agent_kwargs.discount

                    #     gamma_chunk = gamma_step # ** H

                    #     mc_returns = []
                    #     for traj_i, t in enumerate(trajectories):
                    #         # per-step rewards (same scaling/biasing you already do)
                    #         r = np.asarray(t["reward"], dtype=np.float32) * FLAGS.reward_scale + FLAGS.reward_bias
                    #         d = np.asarray(t["done"], dtype=np.bool_)  # per-step done flags

                    #         # chunk starts: 0, H, 2H, ...
                    #         starts = np.arange(0, len(r), H)

                    #         # sum rewards inside each chunk; last chunk can be shorter automatically
                    #         r_chunks = np.add.reduceat(r, starts)

                    #         # chunk done/mask: mark chunk terminal if ANY step inside the chunk is done
                    #         done_chunks = np.array([d[s : min(s + H, len(d))].any() for s in starts], dtype=np.float32)
                    #         masks_chunks = 1.0 - done_chunks

                    #         # assert alignment with the Q-values you logged per chunk
                    #         assert len(r_chunks) == len(q_vs_mc_returns_vals[traj_i]), (
                    #             f"traj {traj_i}: #reward_chunks={len(r_chunks)} != "
                    #             f"#q_vs_mc_points={len(q_vs_mc_returns_vals[traj_i])} (H={H}, T={len(r)})"
                    #         )

                    #         mc = calc_return_to_go(
                    #             rewards=r_chunks,
                    #             masks=masks_chunks,
                    #             gamma=gamma_chunk,
                    #             push_failed_to_min=("maze" in FLAGS.environment_name) or (FLAGS.environment_name == "real_robot"),
                    #             min_reward=FLAGS.reward_bias,
                    #         )
                    #         mc_returns.append(mc)

                    #     initial_mc_returns = [mc[0] for mc in mc_returns]
                    #     # breakpoint()

                    #     # q_vs_mc_data_list = []
                    #     # for idx in range(len(trajectories)):
                    #     #     mc_return = mc_returns[idx]
                    #     #     q_vals = qs_across_trajectories[idx]
                    #     #     num_qs = len(q_vals)
                    #     #     num_points = mc_return.shape[0] // num_qs
                    #     #     mc_return_slice = mc_return[pi_config.model.action_horizon::pi_config.model.action_horizon]
                    #     #     assert len(mc_return_slice) <= len(q_vals)
                    #     #     q_vals = np.array(q_vals[:len(mc_return_slice)]).reshape(-1,)
                    #     #     q_vs_mc_data_list.append([q_vals, mc_return_slice]) # Both are of shape (T,); T varies across trajectories

                    #     # q_vs_mc_data_list = np.array(q_vs_mc_data_list)

                    #     # 1. Logic to create flattened arrays for plotting
                    #     all_q_values = []
                    #     all_mc_returns = []

                    #     for idx in range(len(trajectories)):
                    #         mc_return = mc_returns[idx]
                    #         q_vals = qs_across_trajectories[idx]
                            
                    #         # ... [Your slicing logic] ...
                    #         # Ensure this logic matches your specific horizon/stride needs
                    #         # mc_return_slice = mc_return[pi_config.model.action_horizon::pi_config.model.action_horizon]
                    #         mc_return_slice = mc_return
                    #         assert len(mc_return_slice) == len(q_vals)
                    #         # Safety clip to matching length
                    #         min_len = min(len(mc_return_slice), len(q_vals))
                            
                    #         # Flatten just in case, though they should already be 1D
                    #         q_vals_segment = np.array(q_vals[:min_len]).reshape(-1)
                    #         mc_return_segment = np.array(mc_return_slice[:min_len]).reshape(-1)
                            
                    #         all_q_values.append(q_vals_segment)
                    #         all_mc_returns.append(mc_return_segment)

                    #     # Concatenate for plotting
                    #     if len(all_q_values) > 0:
                    #         flat_q_values = np.concatenate(all_q_values, axis=0)
                    #         flat_mc_returns = np.concatenate(all_mc_returns, axis=0)
                    #         np.save(f"flat_q_values_iter_{i}.npy", flat_q_values)
                    #         np.save(f"flat_mc_returns_iter_{i}.npy", flat_mc_returns)

                    #         # Setup simple plot
                    #         plt.figure(figsize=(8, 6))
                            
                    #         # Basic scatter plot
                    #         plt.scatter(flat_q_values, flat_mc_returns, alpha=0.6, s=20)

                    #         # Simple labels and grid
                    #         plt.xlabel("Q-Values (Predicted)")
                    #         plt.ylabel("Monte Carlo Returns (Actual)")
                    #         plt.title(f"Q-Values vs Returns (Step {i})")
                    #         plt.grid(True, alpha=0.3)

                    #         # Save to WandB
                    #         buf = io.BytesIO()
                    #         plt.savefig(buf, format="png", bbox_inches='tight')
                    #         buf.seek(0)
                    #         image = Image.open(buf)

                    #         if wandb_logger is not None:
                    #             wandb_logger.log({
                    #                 "eval/q_vs_mc_scatter": wandb.Image(image, caption="Q vs MC Scatter Plot")
                    #             }, step=i)

                    #         plt.close()
                    #         buf.close()
                    #     else:
                    #         print("Warning: No data available for Q vs MC plot.")
                    

                    # debug_metrics = agent.get_debug_metrics(batch=batch, seed=eval_policy_fn_key)
                    # if wandb_logger is not None:
                    #     wandb_logger.log(eval_metrics, step=i)
                    #     wandb_logger.log(
                    #         {f"debug/{k}": float(v) for k, v in debug_metrics.items()},
                    #         step=i,
                    #     )
                    
                    del trajectories
                    import gc; gc.collect()
                # if FLAGS.config.save_video:
                #     try:
                #         eval_video = load_recorded_video(
                #             video_path=eval_env.current_save_path
                #         )
                #         if wandb_logger is not None:
                #             wandb_logger.log({"evaluation/video": eval_video}, step=i)
                #     except Exception as e:
                #         pass
                timer.tock("evaluation/total")

                if FLAGS.config.save_dir:
                    os.makedirs(FLAGS.config.save_dir, exist_ok=True)
                    final_checkpoint_path = os.path.join(FLAGS.config.save_dir, f"checkpoint_{i}.pkl")
                    logging.info(f"Saving final checkpoint to {final_checkpoint_path}")
                    with open(final_checkpoint_path, 'wb') as f:
                        pickle.dump({
                            'critic_params': agent.critic.params,
                            'target_critic_params': agent.target_critic.params,
                            'edit_actor_params': agent.edit_actor.params,
                            'temp_params': agent.temp.params,
                            'step': i,
                        }, f)
        
        ### Offline training ###
        # Not really expo style as of now, but keep it here in case need to do this paradigm later #
        else:
            # Sample an offline batch and do an update #
            batch = next(offline_train_iterator)
            # breakpoint()
            # batch = shard_batch(batch, sharding)
            batch = set_batch_masks(
                batch, FLAGS.environment_name, FLAGS.reward_bias, FLAGS.reward_scale
            )
            agent, info = agent.update(batch, utd_ratio=FLAGS.config.utd_ratio, timer=timer)
            print(timer.get_total_times(reset=True))

            

        # breakpoint()

if __name__ == "__main__":
    app.run(train_agent)