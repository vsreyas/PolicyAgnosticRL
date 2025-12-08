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
from jaxrl_m.agents.continuous.expo_pi import ExpoPiLearner
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
flags.DEFINE_float("clip_action", 0.99999, "Clip action.")
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

def add_empty_observation_history_axis_to_batch(batch: Batch) -> Batch:
    """
    Add empty chunking dimension to observations, next_observations, and actions.

    This is used for DDPM, because it assumes observation history.

    Args:
        batch: Training batch.

    Returns:
        Batch with empty observation history axis.
    """
    for key in ["observations", "next_observations", "actions"]:
        # First dimension is batch size, second dimension is chunking dimension
        batch[key] = jax.tree_map(lambda x: x[:, None], batch[key])
    return batch


def unbatch_observation_history_axis(batch: Batch) -> Batch:
    """
    Remove the observation history axis from the batch.

    Args:
        batch: Training batch.

    Returns:
        Batch without observation history axis.
    """
    for key in ["observations", "next_observations", "actions"]:
        batch[key] = jax.tree_map(lambda x: x[:, 0], batch[key])
    return batch


def preprocess_batch_with_action_optimization(
    batch: Batch,
    critic_agent,
    local_optimization_steps: int,
    local_optimization_step_size: float,
    optimize_critic_ensemble_min: bool,
    action_space_low: gym.Space,
    action_space_high: gym.Space,
    improve_actions_with_global_optimization: bool,
    base_policy_agent: Optional[flax.struct.PyTreeNode] = None,
    num_base_policy_actions: int = 32,
    num_actions_to_keep: int = 10,
    distill_argmax: bool = True,
    rng: Optional[jax.random.PRNGKey] = None,
) -> Dict[str, jnp.ndarray]:
    """
    Preprocess the batch with Action Optimization.

    Args:
        batch: Training batch.
        critic_agent: Critic agent (e.g. Cal-QL).
        local_optimization_steps: Number of gradient steps.
        local_optimization_step_size: Step size.
        optimize_critic_ensemble_min: Whether gradient steps are taken w.r.t. minimum or mean.
        action_space_low: Low bound of the action space.
        action_space_high: High bound of the action space.
        improve_actions_with_global_search: Whether to use global optimization.
        base_policy_agent: Base policy agent.
        num_base_policy_actions: Number of actions to sample for global optimization.
        num_actions_to_keep: Number of actions to keep for global optimization.
        distill_argmax: Whether to only keep the best action from the candidate set. If False, will
            sample from a softmax over action candidates.
        rng: Random seed.

    Returns:
        Batch after applying action optimization to the actions.
    """

    # assert (
    #     len(batch["actions"].shape) == 3 and batch["actions"].shape[1] == 1
    # ), f"This function assumes an empty action chunking axis. Found actions with shape {batch['actions'].shape}"

    # if isinstance(batch["observations"], dict):
    #     if "image" in batch["observations"]:
    #         assert (
    #             len(batch["observations"]["image"].shape) == 5
    #             and batch["observations"]["image"].shape[1] == 1
    #         ), f"This function assumes an empty observation history axis. Found images with shape {batch['observations']['image'].shape}"
    #     else:
    #         assert (
    #             batch["observations"]["state"].ndim == 3
    #             and batch["observations"]["state"].shape[1] == 1
    #         ), f"This function assumes an empty observation history axis. Found states with shape {batch['observations']['state'].shape}"
    # else:
    #     assert (
    #         len(batch["observations"].shape) == 3
    #         and batch["observations"].shape[1] == 1
    #     ), f"This function assumes an empty observation history axis. Found observations with shape {batch['observations'].shape}"

    # Unbatch the dataset
    # batch = unbatch_observation_history_axis(batch)
    observations = batch["observations"]

    if improve_actions_with_global_optimization:
        assert base_policy_agent is not None
        assert rng is not None
        rng, key = jax.random.split(rng)
        action_distribution, info = action_optimization_sample_actions(
            observations,
            critic_agent,
            critic_state=critic_agent.state,
            num_base_policy_actions=num_base_policy_actions,
            num_actions_to_keep=num_actions_to_keep,
            num_steps=local_optimization_steps,
            step_size=local_optimization_step_size,
            optimize_critic_ensemble_min=optimize_critic_ensemble_min,
            rng=key,
            action_space_low=action_space_low,
            action_space_high=action_space_high,
            argmax=distill_argmax,
            dataset_actions_to_consider=batch["actions"],
        )
        batch["actions"] = action_distribution.sample(seed=rng)
    else:
        if isinstance(observations, dict) and "ddpm_actions" in observations:
            observations = observations["state"]
        local_optimization_results: LocalOptimizationState = (
            take_local_optimization_steps(
                observations,
                batch["actions"],
                critic=critic_agent,
                critic_state=critic_agent.state,
                num_steps=local_optimization_steps,
                step_size=local_optimization_step_size,
                optimize_critic_ensemble_min=optimize_critic_ensemble_min,
                action_space_low=action_space_low,
                action_space_high=action_space_high,
            )
        )
        batch["actions"] = local_optimization_results.actions
    # batch = add_empty_observation_history_axis_to_batch(batch)
    return batch


def get_robot_action_space(
    normalization_action_mean: np.ndarray,
    normalization_action_std: np.ndarray,
    dof: int,
) -> gym.spaces.Box:
    original_action_space_low = np.array([-0.05, -0.05, -0.05, -0.25, -0.25, -0.25])[
        :dof
    ]
    original_action_space_high = np.array([0.05, 0.05, 0.05, 0.25, 0.25, 0.25])[:dof]
    return gym.spaces.Box(
        np.concatenate(
            [
                (original_action_space_low - normalization_action_mean[:dof])
                / normalization_action_std[:dof],
                [0],
            ]
        ),
        np.concatenate(
            [
                (original_action_space_high - normalization_action_mean[:dof])
                / normalization_action_std[:dof],
                [1],
            ]
        ),
    )


def get_base_policy_agent_pi(
    base_policy_type: BasePolicyTypes,
    rng: jax.random.PRNGKey,
    config,
    base_policy_path: Optional[str] = None,
) -> BasePolicy:
    base_policy_class = BASE_POLICY_TYPE_TO_CLASS[base_policy_type]
    # TODO change this definition according to Pi 
    base_policy_agent = base_policy_class(
        rng=rng,
        config=config,
    )

    if base_policy_path is not None:
        base_policy_agent = base_policy_agent.restore_checkpoint(
            base_policy_path
        )

    return base_policy_agent

def sanitize_obs(obs):
    out = {}
    for k, v in obs.items():
        if isinstance(v, (np.ndarray, jnp.ndarray)):
            out[k] = v
        # skip strings, lists, python objects
    return out


def get_policy_fn(
    agent: ExpoPiLearner,
    rng: jax.random.PRNGKey,
    timer: Timer | None = None,
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
        actions = jax.device_get(
            agent.sample_actions(
                observations, *args, **kwargs, timer=timer,
            )
        )
        # breakpoint()

        return actions

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


def restart_agent_optimizer_state(agent):
    assert hasattr(agent, "state") and isinstance(agent.state, JaxRLTrainState)
    agent.state = agent.state.replace(
        opt_states=JaxRLTrainState._tx_tree_map(
            lambda tx: tx.init(agent.state.params), agent.state.txs
        ),
        step=0,
    )
    return agent


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
    pi_config.exp_name = FLAGS.wandb_experiment_name
    pi_config.overwrite = True

    # # LOG: WANDB setup #
    # if FLAGS.wandb_project_name is not None:
    #     wandb_config = WandBLogger.get_default_config()
    #     wandb_config.update(
    #         {
    #             "project": FLAGS.wandb_project_name,
    #             "exp_descriptor": FLAGS.wandb_experiment_name,
    #             "tag": None,
    #             "group": FLAGS.wandb_group,
    #         }
    #     )
    #     wandb_logger = WandBLogger(
    #         wandb_config=wandb_config,
    #         variant=FLAGS.config.to_dict(),
    #         debug=FLAGS.debug,
    #     )
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
    # else:
    #     wandb_logger = None
    #     save_dir = tf.io.gfile.join(
    #         os.path.abspath(FLAGS.config.save_dir),
    #     )
    #     config.wandb_enabled = False

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
        tf.io.gfile.makedirs(tf.io.gfile.join(save_dir, "image_replay_buffer"))
        assert not tf.io.gfile.exists(
            tf.io.gfile.join(save_dir, "image_replay_buffer", "episode_0.tfrecord")
        ), f"Image replay buffer already exists! ({tf.io.gfile.join(save_dir, 'image_replay_buffer', 'episode_0.tfrecord')})"
        image_replay_buffer = None  # Will be created when switching to online training.
        state_replay_buffer = None

    rng = jax.random.PRNGKey(FLAGS.seed)
    # we shard the leading dimension (batch dimension) accross all devices evenly
    # sharding = jax.sharding.PositionalSharding(devices)
    sharding = jax.sharding.PositionalSharding(devices)
    # Create data iterators
    # LOG: Offline dataset #
    offline_train_iterator_for_critic = dataset.iterator(
        batch_size=FLAGS.config.agent_kwargs.batch_size
    )
    offline_train_iterator_for_base_policy = None
    # Online dataset/buffer #
    # Online iterators will be set when switching to online training.
    online_train_iterator_for_critic = None
    #########################################################
    # Dataset created now in `dataset` and environment created now in `train_env` #
    # breakpoint()

    ### Sharding Data ###
    example_batch = next(offline_train_iterator_for_critic)
    example_batch = shard_batch(example_batch, sharding)
    
    ### Create trajectory sampler ###
    data_collection_trajectory_sampler = TrajSampler(
        train_env,
        clip_action=FLAGS.clip_action,
        reward_scale=FLAGS.reward_scale,
        reward_bias=FLAGS.reward_bias,
        max_traj_length=FLAGS.config.get("max_episode_steps", 1000),
    )

    ### Create EXPO agent #
    rng, construct_rng = jax.random.split(rng)
    agent = ExpoPiLearner.create(
        config=pi_config,
        seed=FLAGS.seed,
        observations=example_batch,
        rng=construct_rng,
        N=FLAGS.num_actions_to_sample,
        n_edit_samples=FLAGS.num_edit_samples,
    )
    # breakpoint()

    # TODO: Remove hardcode and init with flags appropriately #
    num_trajectories_to_collect = 1

    timer = Timer()

    ### EXPO agent training ###
    ### Online training ###
    for i in range(FLAGS.num_offline_epochs + FLAGS.num_online_epochs + 1):
        if i >= FLAGS.num_offline_epochs and FLAGS.num_online_epochs > 0:
            logging.info("Switching to online training...")
            data_collection_rng_key, rng = jax.random.split(rng)
            env_data_collection_policy_fn = get_policy_fn(
                agent=agent,
                rng=data_collection_rng_key,
                timer=timer,
            )

            trajectories = []
            for traj_index in range(num_trajectories_to_collect):
                timer.tick("trajectory_sampling_time")
                traj = data_collection_trajectory_sampler.sample(
                    env_data_collection_policy_fn,
                    num_episodes=1,
                    replay_buffer=state_replay_buffer,
                    calc_mc_return_fn=functools.partial(calc_mc_return_fn, discount=FLAGS.config.agent_kwargs.discount, reward_bias=FLAGS.reward_bias),
                    store_max_trajectory_reward=True,
                    terminate_on_success=FLAGS.config.get(
                        "early_terminate_on_success", False
                    ),
                )[0]
                timer.tock("trajectory_sampling_time")
                print(timer.get_total_times(reset=False))
                # LOG: `traj` statistics #
                # breakpoint()
                trajectories.append(traj)

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

            # Finished collecting trajectories
            # LOG: Construct buffers using the trajectories #
            # LOG: Looks like two iterators are constructed, one for online trajectories and `dataset` from previous definition, for offline dataset #
            online_env_steps += online_env_steps_this_epoch
            # Recreate the image replay buffer iterator to include the new trajectories
            timer.tick("recreate_image_replay_buffer_iterator")
            data_paths = glob_to_path_list(
                tf.io.gfile.join(save_dir, "image_replay_buffer", "*.tfrecord")
            )
            image_replay_buffer = ImageReplayBufferPi(
                data_paths=data_paths,
                seed=FLAGS.seed,
                train=True,
                task_name=FLAGS.task_name,
                use_wrist_view=FLAGS.use_wrist_view, 
                use_language=FLAGS.use_lang, config=pi_config,
                **FLAGS.config.image_replay_buffer_kwargs,
            )
        
        ### Offline training ###
        else:
            # Sample an offline batch and do an update #
            batch = next(offline_train_iterator_for_critic)
            # breakpoint()
            batch = shard_batch(batch, sharding)
            batch = set_batch_masks(
                batch, FLAGS.environment_name, FLAGS.reward_bias, FLAGS.reward_scale
            )
            agent, info = agent.update(batch, utd_ratio=FLAGS.config.utd_ratio, timer=timer)

            

        breakpoint()

if __name__ == "__main__":
    app.run(train_agent)