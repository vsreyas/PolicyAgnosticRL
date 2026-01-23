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
    q_chunking_best_of_n,
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
from jaxrl_m.envs.d4rl import TruncationWrapper, get_d4rl_dataset_with_mc_calculation
from jaxrl_m.utils.timer_utils import Timer
from jaxrl_m.utils.train_utils import concatenate_batches, load_recorded_video
from jaxrl_m.vision import encoders
from jaxrl_m.utils.train_utils import preprocess_action, repack_action

try:
    from jax_smi import initialise_tracking  # type: ignore

    initialise_tracking()
except ImportError:
    pass

from openpi.training.config import get_config

FLAGS = flags.FLAGS

flags.DEFINE_string("environment_name", "", "Environment name.")
flags.DEFINE_string("wandb_project_name", None , "WandB project name.") # "PI-0.5-finetuning-filtered-BC"
flags.DEFINE_string("wandb_experiment_name", "PI-0.5-finetuning-filtered-BC", "WandB experiment name.")
flags.DEFINE_string("wandb_group", "", "WandB group.")
config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)
flags.DEFINE_string("PI_config", "pi05_libero_custom_low_mem", "Which PI config to use.")
config_flags.DEFINE_config_file(
    "bridgedata_config",
    None,
    "File path to the bridgedata configuration.",
    lock_config=False,
)
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("num_offline_epochs", 0, "Number of epochs for pre-training.")
flags.DEFINE_integer(
    "num_online_epochs", 100, "Number of epochs for online fine-tuning."
)
flags.DEFINE_integer(
    "num_train_steps_per_offline_epoch", 10, "Number of training steps per epoch."
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
flags.DEFINE_integer(
    "max_attempts_to_collect_successful_trajectory",
    10,
    "Maximum attempts to collect a successful trajectory."
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
flags.DEFINE_integer("critic_utd", 3, "update-to-data ratio of the critic")
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
# Is every GPU -- Sanity check if every GPU has different data TODO
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
    q_chunking: bool = False,
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
    elif q_chunking:
        assert base_policy_agent is not None
        assert rng is not None
        rng, key = jax.random.split(rng)
        batch["actions"] = q_chunking_best_of_n(
            observations,
            critic_agent,
            critic_state=critic_agent.state,
            num_base_policy_actions=num_base_policy_actions,
            rng=key,
            dataset_actions_to_consider=batch["actions"],
        )
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
    # breakpoint()
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
    agent: BasePolicy,
    argmax: bool,
    num_samples_from_base_policy: int,
    timer: Timer,
    rng: jax.random.PRNGKey,
    base_policy: Optional[BasePolicy] = None,
) -> Callable[[Data], np.ndarray]:
    def policy_fn(observations: Data, *args, **kwargs) -> np.ndarray:
        if not isinstance(observations, dict):
            observations = {"state": observations}
        if "state" in observations:
            obs_ndim = observations["state"].ndim
        else:
            assert "proprio" in observations
            obs_ndim = observations["proprio"].ndim

        if base_policy is not None:
            # Get samples from the base policy, put them in the observation dict

            # if obs_ndim == 2:
            #     batched_observations = jax.tree_map(lambda x: x[:, None], observations)
            # elif obs_ndim == 1:
            #     batched_observations = jax.tree_map(
            #         lambda x: x[None, None], observations
            #     )
            observations["base_policy_actions"], obs_processed = base_policy.sample_actions(
                observations,
                repeat=num_samples_from_base_policy,
                timer=timer,
                argmax=False,
                normalized=True,
                return_obs=True,
                **kwargs,
            )
            actions = observations["base_policy_actions"]
            observations['proprio'] = obs_processed['state'][0]
            observations['image'] = obs_processed['image']['base_0_rgb'][0]
            observations['wrist_image'] = obs_processed['image']['left_wrist_0_rgb'][0]
            B, R, H, D = actions.shape
            actions = actions.reshape(B*R, H, D)
            actions = preprocess_action(actions)
            actions = actions.reshape(B,R,actions.shape[-1])
            observations["base_policy_actions"] = actions
            observations = sanitize_obs(observations)
            if obs_ndim == 1:
                observations = jax.tree_map(lambda x: x[None,: ], observations)
                obs_ndim += 1
            observations["image"] = resize_images_to_100x100(
                observations["image"]
            )
            if FLAGS.use_wrist_view:
                observations["wrist_image"] = resize_images_to_100x100(
                    observations["wrist_image"]
                )

        if "ddpm" in FLAGS.config.agent:

            if obs_ndim == 2:
                observations = jax.tree_map(lambda x: x[:, None], observations)
            elif obs_ndim == 1:
                observations = jax.tree_map(lambda x: x[None, None], observations)
        
        
 
        # print("Observations: ", observations['image'].shape)
        #TODO this is not right, since the observation preprocessing happens inside the dataloader
        actions = jax.device_get(
            agent.sample_actions(
                observations, *args, **kwargs, argmax=argmax, timer=timer
            )
        )
        if "parl" in FLAGS.config.agent:
            print("ono the config has parl")
            actions = repack_action(actions)
            output ={}
            output["actions"] = actions
            output['state'] = observations["proprio"]
            actions = base_policy.unnormalize(output)
            actions = actions['actions']
        # if actions.ndim == 3:
        #     assert actions.shape[1] == 1, actions.shape
        #     actions = actions[:, 0]
        return actions

    policy_fn = supply_rng(policy_fn, rng=rng)

    return policy_fn


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

### Try subprocenv ###
import multiprocessing as mp
from multiprocessing.connection import wait

def env_worker(conn, make_env_fn):
    # Import robosuite / libero INSIDE the subprocess
    env = make_env_fn()
    try:
        while True:
            msg = conn.recv()
            cmd = msg[0]
            if cmd == "reset":
                conn.send(env.reset())
            elif cmd == "get_attr":
                conn.send(getattr(env, "max_steps"))
            elif cmd == "step":
                action = msg[1]
                conn.send(env.step(action))
            elif cmd == "close":
                try:
                    env.close()
                except Exception:
                    pass
                conn.send(None)
                break
            else:
                raise RuntimeError(f"Unknown cmd: {cmd}")
    finally:
        try:
            env.close()
        except Exception:
            pass

class SubprocEnv:
    def __init__(self, make_env_fn):
        self.make_env_fn = make_env_fn
        self._ctx = mp.get_context("spawn")
        self._start()

    def _start(self):
        self.parent_conn, child_conn = self._ctx.Pipe()
        self.proc = self._ctx.Process(target=env_worker, args=(child_conn, self.make_env_fn), daemon=True)
        self.proc.start()

    def _restart(self):
        self.kill()
        self._start()

    def kill(self):
        try:
            if self.proc.is_alive():
                self.proc.terminate()
                self.proc.join(timeout=5)
        finally:
            try:
                self.parent_conn.close()
            except Exception:
                pass

    def call(self, cmd, *args, timeout_s=300):
        self.parent_conn.send((cmd, *args))
        ready = wait([self.parent_conn], timeout=timeout_s)
        if not ready:
            # hung in native code -> hard reset
            self._restart()
            raise TimeoutError(f"{cmd} timed out after {timeout_s}s")
        return self.parent_conn.recv()

    def reset(self, timeout_s=300):
        return self.call("reset", timeout_s=timeout_s)

    def step(self, action, timeout_s=300):
        return self.call("step", action, timeout_s=timeout_s)

    def close(self):
        try:
            self.call("close", timeout_s=5)
        except Exception:
            pass
        self.kill()
    
    def get_attr(self, name: str):
        return self.call("get_attr", name)

    @property
    def max_steps(self):
        # assumes underlying env exposes `max_steps`
        return self.get_attr("max_steps")

def make_env(task_name):
    from jaxrl_m.envs.libero import get_libero_env, get_libero_config
    cfg = get_libero_config()
    return get_libero_env(cfg=cfg, task_name=task_name, is_pi=True)


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
    
    if FLAGS.wandb_project_name is not None:
        run_id = f"{FLAGS.wandb_experiment_name}+seed{FLAGS.seed}+{FLAGS.wandb_project_name}"
        wandb_config = WandBLogger.get_default_config()
        wandb_config.update(
            {
                "project": FLAGS.wandb_project_name,
                "exp_descriptor": FLAGS.wandb_experiment_name,
                "tag": None,
                "group": FLAGS.wandb_group,
                "id": run_id,
                "resume": "allow",
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
            wandb_logger.config.project,
            wandb_logger.config.exp_descriptor,
            f"seed_{FLAGS.seed}",
        )
    else:
        wandb_logger = None
        save_dir = tf.io.gfile.join(
            os.path.abspath(FLAGS.config.save_dir),
            FLAGS.wandb_experiment_name,
            f"seed_{FLAGS.seed}",
        )
        

    # Create environment and dataset
    action_space = None
    offline_dataset_size = None
    config = get_config(FLAGS.PI_config)
    config.exp_name = FLAGS.wandb_experiment_name
    config.overwrite = False
    config.resume = True
    config.checkpoint_base_dir = tf.io.gfile.join(save_dir, "agent_checkpoints")
    config.assets_base_dir = tf.io.gfile.join(save_dir, "assets")
    config.wandb_enabled = FLAGS.wandb_project_name is not None

    
    if FLAGS.environment_name == "real_robot":
        assert FLAGS.reward_bias == -1.0
        assert FLAGS.reward_scale == 1.0
        from jaxrl_m.envs.robot_env import (
            RobotEnv,
            get_reward_function,
            get_robot_action_space,
        )

        assert isinstance(FLAGS.bridgedata_config.include[0], list)
        task_paths = [
            glob_to_path_list(
                path,
                prefix=FLAGS.config.data_path,
                exclude=FLAGS.bridgedata_config.exclude,
            )
            for path in FLAGS.bridgedata_config.include
        ]

        train_paths = [sub_list for sub_list in task_paths]

        dataset = BridgeDataset(
            train_paths,
            FLAGS.seed,
            train=True,
            action_proprio_metadata=FLAGS.bridgedata_config.action_proprio_metadata,
            **FLAGS.config.dataset_kwargs,
        )
        task_to_initial_eep = get_task_to_initial_eep(train_paths[0])
        if FLAGS.config.general_params.env_params.action_mode == "3trans1rot":
            dof = 4
        elif FLAGS.config.general_params.env_params.action_mode == "3trans":
            dof = 3
        elif FLAGS.config.general_params.env_params.action_mode == "3trans3rot":
            dof = 6
        else:
            raise NotImplementedError

        reward_function = get_reward_function(
            task_name=FLAGS.real_robot_task_name_for_reward,
            task_encoding=np.zeros(512),
            dof=dof,
        )

        action_space = get_robot_action_space(
            normalization_action_mean=FLAGS.bridgedata_config.action_proprio_metadata[
                "action"
            ]["mean"],
            normalization_action_std=FLAGS.bridgedata_config.action_proprio_metadata[
                "action"
            ]["std"],
            dof=dof,
        )

        if FLAGS.train_on_separate_computer_mode != "agent_training_only":
            train_env = RobotEnv(
                env_params=FLAGS.config.general_params.env_params,
                ip=FLAGS.config.general_params.ip,
                port=FLAGS.config.general_params.port,
                train_config=FLAGS.config,
                reward_function=reward_function,
                initial_eep=(
                    task_to_initial_eep[FLAGS.real_robot_task_name_for_initial_eep]
                    if FLAGS.initial_eep_pos is None
                    else np.array(FLAGS.initial_eep_pos)
                ),
                # This script supports single-task fine-tuning only, so language conditioning is not
                # used. Note that OpenVLA handles language conditioning on its own.
                language_conditioning_encoding=np.zeros(512),
                action_mean=FLAGS.bridgedata_config.action_proprio_metadata["action"][
                    "mean"
                ],
                action_std=FLAGS.bridgedata_config.action_proprio_metadata["action"][
                    "std"
                ],
                image_size=FLAGS.config.general_params.image_size,
                # termination_reward_function_name=termination_reward_function_name,
            )
        else:
            train_env = None
        eval_env = None
    elif FLAGS.environment_name == "calvin":
        assert FLAGS.config.image_observations
        from jaxrl_m.envs.calvin import (
            get_calvin_config,
            get_calvin_env,
            get_calvin_tfrecord_dataset,
        )

        dataset = get_calvin_tfrecord_dataset(
            tfrecord_regexp=FLAGS.config.calvin_tfrecord_regexp,
            use_lang=FLAGS.use_lang, env_name=FLAGS.environment_name, **FLAGS.config.dataset_kwargs,
        )
        calvin_config = get_calvin_config()
        use_lang=FLAGS.use_lang
        train_env = get_calvin_env(cfg=calvin_config, use_lang=use_lang,)
        if FLAGS.num_parallel_envs > 1:
            num_parallel_envs = FLAGS.num_parallel_envs
            eval_env = gym.vector.AsyncVectorEnv(
                [
                    lambda: get_calvin_env(
                        cfg=calvin_config, use_lang=use_lang,
                    )
                    for _ in range(num_parallel_envs)
                ],
                context="forkserver",  # the default "fork" is incompatible with JAX
            )
        else:
            eval_env = gym.vector.SyncVectorEnv(
                [
                    lambda: get_calvin_env(
                        cfg=calvin_config, use_lang=use_lang,
                    )
                ]
            )
    elif FLAGS.environment_name=="libero":
        assert FLAGS.config.image_observations
        from jaxrl_m.envs.libero import (
            get_libero_config,
            get_libero_env,
            get_libero_tfrecord_dataset,
        )

        dataset = get_libero_tfrecord_dataset(
            tfrecord_regexp=FLAGS.config.libero_tfrecord_regexp, use_wrist_view=FLAGS.use_wrist_view, 
            use_language=FLAGS.use_lang, config=config, is_pi=True, **FLAGS.config.dataset_kwargs,
            task_name=FLAGS.task_name,
        )
        libero_config = get_libero_config()

        train_env = SubprocEnv(make_env_fn=functools.partial(make_env, task_name=FLAGS.task_name))
        # if FLAGS.num_parallel_envs > 1:
        #     num_parallel_envs = FLAGS.num_parallel_envs
        #     task_name = FLAGS.task_name
        #     eval_env = gym.vector.AsyncVectorEnv(
        #         [
        #             lambda: get_libero_env(
        #                 cfg=libero_config, task_id = ind*num_parallel_envs, task_name=task_name, is_pi=True,
        #             )
        #             for ind in range(num_parallel_envs)
        #         ],
        #         context="forkserver", shared_memory=False, # the default "fork" is incompatible with JAX
        #     )
        # else:
        #     eval_env = get_libero_env(cfg=libero_config, task_name=FLAGS.task_name, is_pi=True)
        eval_env = SubprocEnv(make_env_fn=functools.partial(make_env, task_name=FLAGS.task_name))
        dummy_env = get_libero_env(cfg=libero_config, task_name=FLAGS.task_name, is_pi=True)

    else:
       raise NotImplementedError

    if action_space is None:
        action_space = dummy_env.action_space
    assert action_space.high.ndim == 1, action_space.shape
    # Create replay buffer
    if FLAGS.config.image_observations:
        # tf.io.gfile.makedirs(tf.io.gfile.join(save_dir, "image_replay_buffer"))
        # if FLAGS.train_on_separate_computer_mode != "agent_training_only":
        #     assert not tf.io.gfile.exists(
        #         tf.io.gfile.join(save_dir, "image_replay_buffer", "episode_0.tfrecord")
        #     ), f"Image replay buffer already exists! ({tf.io.gfile.join(save_dir, 'image_replay_buffer', 'episode_0.tfrecord')})"
        image_replay_buffer = None  # Will be created when switching to online training.
        state_replay_buffer = None
    else:
        state_replay_buffer = ReplayBuffer(
            dummy_env.observation_space,
            action_space,
            capacity=FLAGS.config.get("replay_buffer_capacity", int(1e6)),
            # goal_space=finetune_env.observation_space if goal_conditioned else None,
            store_mc_return=True,
            store_max_trajectory_reward=True,
            seed=FLAGS.seed,
        )
        image_replay_buffer = None

    rng = jax.random.PRNGKey(FLAGS.seed)
    # we shard the leading dimension (batch dimension) accross all devices evenly
    # sharding = jax.sharding.PositionalSharding(devices)
    sharding = jax.sharding.PositionalSharding(devices)
    # Create data iterators
    offline_train_iterator_for_critic = dataset.iterator(
        batch_size=FLAGS.config.agent_kwargs.batch_size
    )
    offline_train_iterator_for_base_policy = None
    # Online iterators will be set when switching to online training.
    online_train_iterator_for_critic = None
    online_train_iterator_for_base_policy = None

    # Optionally create base policy agent
    base_policy_path_components = FLAGS.config.get("base_policy_path", "").split(":")
    if len(base_policy_path_components) == 1 and base_policy_path_components[0] == "":
        # This means the agent we're training doesn't use a base policy
        # E.g. we could be pre-training the base policy itself.
        base_policy_agent = None
        base_policy_type = None
    else:
        offline_train_iterator_for_base_policy = dataset.iterator(
            batch_size=int(
                FLAGS.config.base_policy_agent_kwargs.batch_size
                * FLAGS.config.mixing_ratio
            )
        )
        assert len(base_policy_path_components) in [1, 2]
        try:
            base_policy_type = BasePolicyTypes(base_policy_path_components[0])
        except ValueError:
            raise ValueError(
                f"Did you forget to specify the base policy type in the base_policy_path? E.g. ddpm:./results/...\nGot {base_policy_path_components[0]}"
            )
        rng, construct_rng = jax.random.split(rng)
        base_policy_agent = get_base_policy_agent_pi(
            base_policy_type=base_policy_type,
            rng=construct_rng,
            config=config
        )

    example_batch = next(offline_train_iterator_for_critic)
    # if "ddpm" in FLAGS.config.agent:
    #     example_batch = add_empty_observation_history_axis_to_batch(example_batch)

    logging.info(f"Number of devices: {num_devices}")
    if FLAGS.config.image_observations:
        logging.info(f"Batch size: {example_batch['observations']['proprio'].shape[0]}")
        logging.info(
            f"Batch size per device: {example_batch['observations']['proprio'].shape[0] // num_devices}"
        )
    else:
        logging.info(f"Batch size: {example_batch['observations'].shape[0]}")
        logging.info(
            f"Batch size per device: {example_batch['observations'].shape[0] // num_devices}"
        )
    print("parsed tensor keys:", example_batch.keys())

    example_batch = shard_batch(example_batch, sharding)
    if base_policy_agent is not None:
        example_batch["observations"]["image"] = resize_images_to_100x100(
            example_batch["observations"]["image"]
        )
        example_batch["next_observations"]["image"] = resize_images_to_100x100(
            example_batch["next_observations"]["image"]
        )
        if FLAGS.use_wrist_view:
            example_batch["observations"]["wrist_image"] = resize_images_to_100x100(
                example_batch["observations"]["wrist_image"]
            )
            example_batch["next_observations"]["wrist_image"] = resize_images_to_100x100(
                example_batch["next_observations"]["wrist_image"]
            )
    # print("shape of images:" , example_batch["observations"]['image'].shape, 
    #       "\nwrist_view cam: ", example_batch["observations"]['wrist_image'].shape , 
    #       "\n languages shape: ", example_batch["observations"]["language"].shape )
    # # define encoder
 
    # initialize agent
    rng, construct_rng = jax.random.split(rng)

    is_transformer_agent = FLAGS.config.agent in ["auto_regressive_transformer"]

    

    if 'parl' in FLAGS.config.agent:
        # print("line 822: example -- batch keys :", example_batch.keys())
        observations = example_batch["observations"]

        if not isinstance(observations, dict):
            observations = {"state": example_batch["observations"]}
        if FLAGS.config.image_observations:
            encoder_def = encoders[FLAGS.config.encoder](**FLAGS.config.encoder_kwargs)
            # if FLAGS.use_wrist_view:
            #     encoder_def = [encoders[FLAGS.config.encoder](**FLAGS.config.encoder_kwargs),
            #                    encoders[FLAGS.config.encoder](**FLAGS.config.encoder_kwargs)]
        else:
            def encoder_def(x, **kwargs):
                if isinstance(x, dict):
                    x = x["state"]
                if x.ndim == 3:
                    assert x.shape[1] == 1, x.shape
                    return x[:, 0]
                return x
        actions = preprocess_action(example_batch['actions'])

        H = config.model.action_horizon
        FLAGS.config.agent_kwargs.discount = (FLAGS.config.agent_kwargs.discount)**H
        low = jnp.tile(action_space.low, (H,))
        high = jnp.tile(action_space.high, (H,))
        agent = agents[FLAGS.config.agent](
                rng=construct_rng,
                observations=observations,
                actions=actions,
                encoder_def=encoder_def,
                action_space_low=low,
                action_space_high=high,
                num_base_policy_actions=FLAGS.config.parl_config.num_base_policy_actions,
                # encoder_use_lang=FLAGS.use_lang,
                # encoder_use_wrist_view=FLAGS.use_wrist_view,
                **FLAGS.config.agent_kwargs,
            )
    else:
        agent = agents[FLAGS.config.agent](
            rng=construct_rng,
            config=config
        )
    # print("example batch processing done: -----------")
    del example_batch
    

    # Replicate agent across devices
    # The transformer agent handles this internally.
    # if not is_transformer_agent:
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    # agent = jax.device_put(jax.tree_map(jnp.array, agent), sharding.replicate())
    # agent = jax.device_put(agent, sharding.replicate())
    agent.to_device(sharding)
    if base_policy_agent is not None:
        base_policy_agent.to_device(sharding)

    timer = Timer()

    env_data_collection_policy_fn = None  # Will get set later
    rng, eval_policy_fn_key = jax.random.split(rng)
    argmax_eval = (
        base_policy_agent is None
        or FLAGS.config.evaluation_particle_choosing_strategy == "max_q_value"
    )
    eval_policy_fn = get_policy_fn(
        agent=agent,
        argmax=argmax_eval,
        num_samples_from_base_policy=(
            1
            if base_policy_agent is None
            else FLAGS.config.parl_config.num_base_policy_actions
        ),
        timer=timer,
        rng=eval_policy_fn_key,
        base_policy=base_policy_agent,
    )
    # print("eval function instantiated ----")
    data_collection_trajectory_sampler = TrajSampler(
        train_env,
        clip_action=FLAGS.clip_action,
        reward_scale=FLAGS.reward_scale,
        reward_bias=FLAGS.reward_bias,
        max_traj_length=FLAGS.config.get("max_episode_steps", 1000),
    )
    # print("traj sampler set up ---- ")
    def calc_mc_return_fn(rewards, masks):
        return calc_return_to_go(
            rewards,
            masks,
            FLAGS.config.agent_kwargs.discount,
            push_failed_to_min=(
                True if FLAGS.environment_name == "real_robot" else False
            ),
            min_reward=FLAGS.reward_bias,
        )

    online_env_steps = 0
    online_trajectories_added = 0
    online_env_steps_this_epoch = 0
    
    dummy_env.env.close()
    del dummy_env
    print("Starting evaluation...")
    assert tf.io.gfile.join(save_dir, "agent_checkpoints") == config.checkpoint_base_dir

    
    # agent, step = agent.restore_checkpoint("/data/user_data/sreyasv/pi-0-ckpts/bc-finetuning-2500 steps")
    # if step is None:
    #     step = 0
    #     logging.info("No checkpoint found, starting training from scratch...")
    # else:
    #     logging.info(f"Resumed agent from checkpoint {step}: {config.checkpoint_base_dir}")
    trajectories = evaluate_with_trajectories_libero(
                        eval_policy_fn,
                        eval_env,
                        FLAGS.config.num_eval_episodes,
                        action_horizon=config.model.action_horizon
                    )

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
        
        save_rollout_gif(ind_traj, save_dir, step_i=-1, rollout_j=j)
        ind_traj = []
    print("Evaluation rollout gifs saved.")
    print("Save directory: ", save_dir)
      
    del ind_traj, frames
    import gc; gc.collect()
    exit()

    agent, step = agent.restore_checkpoint(config.checkpoint_base_dir)
    if step is None:
        step = 0
        logging.info("No checkpoint found, starting training from scratch...")
    else:
        logging.info(f"Resumed agent from checkpoint {step}: {config.checkpoint_base_dir}")
    
    for i in tqdm(
        range(FLAGS.num_offline_epochs + FLAGS.num_online_epochs + 1), desc="Epoch"
    ):
        timer.tick("total")

        if i < step:
            logging.info(f"Setting epoch {i} to {step} since resuming from checkpoint...")
            i = step

        if i >= FLAGS.num_offline_epochs and FLAGS.num_online_epochs > 0:
            num_trajectories_to_collect = FLAGS.num_online_trajectories_per_epoch
            if i == FLAGS.num_offline_epochs:
                logging.info("Switching to online training...")
                data_collection_rng_key, rng = jax.random.split(rng)
                env_data_collection_policy_fn = get_policy_fn(
                    agent=agent,
                    argmax=False,
                    num_samples_from_base_policy=1,
                    timer=timer,
                    rng=data_collection_rng_key,
                    base_policy=base_policy_agent,
                )
                offline_train_iterator_for_critic = dataset.iterator(
                    batch_size=int(
                        FLAGS.config.agent_kwargs.batch_size * FLAGS.config.mixing_ratio
                    )
                )
                if not FLAGS.config.image_observations:
                    online_train_iterator_for_critic = state_replay_buffer.iterator(
                        batch_size=FLAGS.config.batch_size
                        - int(
                            FLAGS.config.agent_kwargs.batch_size
                            * FLAGS.config.mixing_ratio
                        )
                    )
                    if offline_dataset_size is None:
                        online_train_iterator_for_base_policy = state_replay_buffer.iterator(
                            batch_size=FLAGS.config.base_policy_agent_kwargs.batch_size
                            - int(
                                FLAGS.config.base_policy_agent_kwargs.batch_size
                                * FLAGS.config.mixing_ratio
                            )
                        )

                num_trajectories_to_collect = FLAGS.num_warmup_trajectories

            if FLAGS.train_on_separate_computer_mode == "agent_training_only":
                # Don't collect if mode is agent_training_only
                num_trajectories_to_collect = 0

            trajectories = []
            atleast_one_successful_trajectory_collected = False
            for traj_index in range(num_trajectories_to_collect):
                traj = data_collection_trajectory_sampler.sample(
                    env_data_collection_policy_fn,
                    num_episodes=1,
                    replay_buffer=state_replay_buffer,
                    calc_mc_return_fn=calc_mc_return_fn,
                    store_max_trajectory_reward=True,
                    terminate_on_success=FLAGS.config.get(
                        "early_terminate_on_success", False
                    ),
                )[0]
                traj_success = (True if traj['rewards'][-1] > 0 else False)
                if traj_success:
                    trajectories.append(traj)
                    atleast_one_successful_trajectory_collected = True

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
                # online_env_steps_this_epoch=2
            
            if not atleast_one_successful_trajectory_collected:
                for traj_index in range(FLAGS.max_attempts_to_collect_successful_trajectory):
                    traj = data_collection_trajectory_sampler.sample(
                        env_data_collection_policy_fn,
                        num_episodes=1,
                        replay_buffer=state_replay_buffer,
                        calc_mc_return_fn=calc_mc_return_fn,
                        store_max_trajectory_reward=True,
                        terminate_on_success=False,
                    )[0]
                    traj_success = (True if traj['rewards'][-1] > 0 else False)
                    if traj_success:
                        trajectories.append(traj)
                        atleast_one_successful_trajectory_collected = True
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
                        logging.info(f"Successfully collected a successful trajectory on retry {traj_index+1}!")
                        break
            if not atleast_one_successful_trajectory_collected:
                logging.info("Failed to collect a successful trajectory this epoch.")
                    # online_env_steps_this_epoch=2

            # Finished collecting trajectories
            online_env_steps += online_env_steps_this_epoch
            if FLAGS.config.image_observations:
                # Recreate the image replay buffer iterator to include the new trajectories
                timer.tick("recreate_image_replay_buffer_iterator")
                if atleast_one_successful_trajectory_collected:
                    data_paths = glob_to_path_list(
                    tf.io.gfile.join(save_dir, "image_replay_buffer", "*.tfrecord")
                    )
                    image_replay_buffer = ImageReplayBufferPi(
                        data_paths=data_paths,
                        seed=FLAGS.seed,
                        train=True,
                        task_name=FLAGS.task_name,
                        use_wrist_view=FLAGS.use_wrist_view, 
                        use_language=FLAGS.use_lang, config=config,
                        filter_success=True,
                        **FLAGS.config.image_replay_buffer_kwargs,
                        
                    )
                else:
                    image_replay_buffer = dataset  # use offline dataset only
                    online_trajectories_added = 0
                    online_env_steps_this_epoch = 500
                    online_env_steps += online_env_steps_this_epoch
                    logging.info("Using offline dataset only for image replay buffer since no successful online trajectories were collected.")
                
                online_train_iterator_for_critic = image_replay_buffer.iterator(
                    batch_size=FLAGS.config.batch_size
                    - int(FLAGS.config.batch_size * FLAGS.config.mixing_ratio),
                )
                if base_policy_agent is not None and offline_dataset_size is None:
                    online_train_iterator_for_base_policy = image_replay_buffer.iterator(
                        batch_size=FLAGS.config.base_policy_agent_kwargs.batch_size
                        - int(
                            FLAGS.config.base_policy_agent_kwargs.batch_size
                            * FLAGS.config.mixing_ratio
                        ),
                    )
                # print("batxh size: _--", FLAGS.config.batch_size
                    # - int(FLAGS.config.batch_size * FLAGS.config.mixing_ratio),)

                timer.tock("recreate_image_replay_buffer_iterator")
                # print("image buffer recreated")

            # Get trajectory statistics
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

        """Base policy distillation"""
        if (
            base_policy_agent is not None
            and FLAGS.train_on_separate_computer_mode != "env_steps_only"
            # Distillation only happens during online training
            and i >= FLAGS.num_offline_epochs
            and FLAGS.num_online_epochs > 0
        ):
            # print("base policy distallation --")
            # print("type: ", type(online_train_iterator_for_base_policy))
            if offline_dataset_size is not None:
                # The ratio of offline to online data changes after every epoch, so we need to
                # update the iterator.
                total_data_size = offline_dataset_size + online_env_steps
                offline_train_iterator_for_base_policy = dataset.iterator(
                    batch_size=int(
                        (offline_dataset_size / total_data_size)
                        * FLAGS.config.base_policy_agent_kwargs.batch_size
                    )
                )

                online_train_iterator_for_base_policy = state_replay_buffer.iterator(
                    batch_size=FLAGS.config.base_policy_agent_kwargs.batch_size
                    - int(
                        (offline_dataset_size / total_data_size)
                        * FLAGS.config.base_policy_agent_kwargs.batch_size
                    )
                )
                # print("type: ", type(online_train_iterator_for_base_policy))
            num_base_policy_distillation_steps = int(
                online_env_steps_this_epoch * FLAGS.base_policy_utd
            )
            print(
                f"Distilling base policy for {num_base_policy_distillation_steps} steps"
            )
            timer.tick("base_policy_distillation/total")
            for update_index in tqdm(
                range(num_base_policy_distillation_steps),
                desc="Policy distillation",
                leave=False,
            ):
                timer.tick("base_policy_distillation/get_batch")
                if FLAGS.config.mixing_ratio > 0:
                    offline_batch = next(offline_train_iterator_for_base_policy)
                else:
                    offline_batch = None
                # print("Update index for base_policy distillation:" , update_index)
                online_batch = next(online_train_iterator_for_base_policy)
                if offline_batch is not None:
                    batch = concatenate_batches([offline_batch, online_batch])
                else:
                    batch = online_batch

                assert batch["rewards"].shape == (
                    FLAGS.config.base_policy_agent_kwargs.batch_size,
                )

                batch = shard_batch(batch, sharding)
                # print("adding base policy actions in base policy distallation: ---")
                # print(batch.keys())
                if FLAGS.config.improve_base_policy_actions_with_global_search or FLAGS.config.parl_config.q_chunking:
                    # pre-compute actions from base policy, add them to the batch
                    rng, key = jax.random.split(rng)
                    batch = add_base_policy_actions_to_batch(
                        batch,
                        base_policy_agent,
                        FLAGS.config.parl_config.num_base_policy_actions,
                        seed=key,
                        base_policy_type=base_policy_type,
                        save_dir=save_dir,
                        epoch=i,
                        timer=timer,
                    )

                base_policy_agent.prepare_for_finetuning()
                # batch = add_empty_observation_history_axis_to_batch(batch)
                timer.tock("base_policy_distillation/get_batch")

                timer.tick(
                    "base_policy_distillation/preprocess_batch_with_action_optimization"
                )
                rng, key = jax.random.split(rng)
                batch['actions'] = preprocess_action(batch['actions'])
                batch["actions"] = np.clip(
                    batch["actions"], low, high
                )
                batch['observations']['next_on_policy_actions'] = preprocess_action(batch['observations']['next_on_policy_actions'])
                batch["observations"]["next_on_policy_actions"] = np.clip(
                    batch["observations"]["next_on_policy_actions"], low, high
                )
                # print("preprocessing bathc with action optimisation")
                # print(batch.keys())

                batch = preprocess_batch_with_action_optimization(
                    batch=batch,
                    critic_agent=agent,
                    local_optimization_steps=FLAGS.config.parl_config.num_steps,
                    local_optimization_step_size=FLAGS.config.parl_config.step_size,
                    optimize_critic_ensemble_min=FLAGS.config.parl_config.optimize_critic_ensemble_min,
                    action_space_low=low,
                    action_space_high=high,
                    improve_actions_with_global_optimization=FLAGS.config.improve_base_policy_actions_with_global_search,
                    base_policy_agent=base_policy_agent,
                    num_base_policy_actions=FLAGS.config.parl_config.num_base_policy_actions,
                    num_actions_to_keep=FLAGS.config.parl_config.num_actions_to_keep,
                    distill_argmax=FLAGS.config.distill_argmax,
                    rng=key,
                    q_chunking=FLAGS.config.parl_config.q_chunking,
                )
                batch['actions'] = repack_action(batch['actions'], pad=True)
                batch['observations']['next_on_policy_actions'] = repack_action(batch['observations']['next_on_policy_actions'], pad=True)
                timer.tock(
                    "base_policy_distillation/preprocess_batch_with_action_optimization"
                )

                timer.tick("base_policy_distillation/update")
                base_policy_update_info = base_policy_agent.update(batch, timer=timer)
                timer.tock("base_policy_distillation/update")
                if update_index == 0:
                    base_policy_update_info = jax.device_get(base_policy_update_info)
                    if wandb_logger is not None:
                        wandb_logger.log(
                            {"base_policy_distillation": base_policy_update_info}, step=i
                        )
                base_policy_agent.prepare_for_inference()

            timer.tock("base_policy_distillation/total")

            if FLAGS.train_on_separate_computer_mode == "agent_training_only":
                # Save base policy checkpoint for this epoch, so caching workers can already use it
                timer.tick("base_policy_save_checkpoint")
                base_policy_agent.save_checkpoint(
                    tf.io.gfile.join(
                        save_dir,
                        "base_policy_checkpoints_from_agent_trainer",
                        f"checkpoint_{i}",
                    )
                )
                timer.tock("base_policy_save_checkpoint")

        """Critic update"""
        # print("critic update -----")
        timer.tick("critic_training/total")
        num_train_steps = (
            FLAGS.num_train_steps_per_offline_epoch
            if i < FLAGS.num_offline_epochs
            else online_env_steps_this_epoch * FLAGS.critic_utd
        )
        for batch_idx in tqdm(
            range(num_train_steps), desc="Agent training", leave=False
        ):
            timer.tick("critic_training/get_batch")

            if i < FLAGS.num_offline_epochs or FLAGS.config.mixing_ratio > 0:
                timer.tick("critic_training/get_batch/offline_iterator")
                offline_batch = next(offline_train_iterator_for_critic)
                timer.tock("critic_training/get_batch/offline_iterator")
            else:
                offline_batch = None

            if i >= FLAGS.num_offline_epochs:
                timer.tick("critic_training/get_batch/online_iterator")
                online_batch = next(online_train_iterator_for_critic)

                if offline_batch is not None:
                    batch = concatenate_batches([offline_batch, online_batch])
                else:
                    batch = online_batch
                timer.tock("critic_training/get_batch/online_iterator")
            else:
                batch = offline_batch
            assert batch["rewards"].shape[0] == FLAGS.config.batch_size

            timer.tick("critic_training/batch_processing")
            # batch = set_batch_masks(
            #     batch, FLAGS.environment_name, FLAGS.reward_bias, FLAGS.reward_scale
            # )
            # print("batch in critic traiining-----")
            # print(batch.keys())
            # if "ddpm" in FLAGS.config.agent:
            #     batch = add_empty_observation_history_axis_to_batch(batch)

            # batch["actions"] = np.clip(
            #     batch["actions"], action_space.low, action_space.high
            # )
            timer.tock("critic_training/batch_processing")

            timer.tick("critic_training/shard_batch")
            batch = shard_batch(batch, sharding)
            timer.tock("critic_training/shard_batch")

            # Pre-compute actions from base policy (if not training base policy)
            if base_policy_agent is not None:
                timer.tick("critic_training/add_base_policy_actions_to_batch")
                manual_cache_dir = (
                    FLAGS.base_policy_offline_cache_path
                    if i < FLAGS.num_offline_epochs
                    else None
                )
                rng, key = jax.random.split(rng)

                batch = add_base_policy_actions_to_batch(
                    batch,
                    base_policy_agent=base_policy_agent,
                    base_policy_type=base_policy_type,
                    num_base_policy_actions=FLAGS.config.parl_config.num_base_policy_actions,
                    save_dir=save_dir,
                    epoch=i,
                    timer=timer,
                    # Critic update requires policy next actions.
                    add_to_next_observations=True,
                    manual_cache_dir=manual_cache_dir,
                    seed=key,
                )

                # print("after adding base policy actions to batch")
                timer.tock("critic_training/add_base_policy_actions_to_batch")

            if (
                base_policy_agent is not None
            ):
                # OpenVLA requires 224x224 images. For a single-task critic, 100x100 is likely
                # enough and speeds up training.
                timer.tick("critic_image_resize")
                batch["observations"]["image"] = resize_images_to_100x100(
                    batch["observations"]["image"]
                )
                batch["next_observations"]["image"] = resize_images_to_100x100(
                    batch["next_observations"]["image"]
                )
                if FLAGS.use_wrist_view:
                    batch["observations"]["wrist_image"] = resize_images_to_100x100(
                        batch["observations"]["wrist_image"]
                    )
                    batch["next_observations"]["wrist_image"] = resize_images_to_100x100(
                        batch["next_observations"]["wrist_image"]
                    )
                # print("image resize --completed")
                # print(batch.keys())
                # print(batch['observations']['image'].shape)
                timer.tock("critic_image_resize")

            timer.tock("critic_training/get_batch")
            timer.tick("agent.update")
            # batch['actions'] = preprocess_action(batch['actions'])
            # batch["actions"] = np.clip(
            #     batch["actions"], low, high
            # )
            # batch['observations']['next_on_policy_actions'] = preprocess_action(batch['observations']['next_on_policy_actions'])
            # batch["observations"]["next_on_policy_actions"] = np.clip(
            #     batch["observations"]["next_on_policy_actions"], low, high
            # )
            print("Pi agent update ---- ")
            # print(batch.keys())
            update_return_values = agent.update(
                batch,
            )

            # print("PARL update done")
            if len(update_return_values) == 2:
                agent, critic_update_info = update_return_values
            else:
                critic_update_info = update_return_values
            timer.tock("agent.update")

            timer.tick("wandb_logging")
            if batch_idx == 0:
                critic_update_info = jax.device_get(critic_update_info)
                debug_metrics = agent.get_debug_metrics(batch=batch, seed=eval_policy_fn_key)
                batch_info = {
                    "rewards_mean": np.mean(batch["rewards"]),
                    "rewards_std": np.std(batch["rewards"]),
                    "rewards_max": np.max(batch["rewards"]),
                    "rewards_min": np.min(batch["rewards"]),
                    "masks_mean": np.mean(batch["masks"]),
                    "masks_std": np.std(batch["masks"]),
                    "masks_max": np.max(batch["masks"]),
                    "masks_min": np.min(batch["masks"]),
                    "actions_mean": np.mean(batch["actions"]),
                    "actions_std": np.std(batch["actions"]),
                    "actions_max": np.max(batch["actions"]),
                    "actions_min": np.min(batch["actions"]),
                }
                if "mc_returns" in batch:
                    batch_info.update(
                        {
                            "mc_returns_mean": np.mean(batch["mc_returns"]),
                            "mc_returns_std": np.std(batch["mc_returns"]),
                            "mc_returns_max": np.max(batch["mc_returns"]),
                            "mc_returns_min": np.min(batch["mc_returns"]),
                        }
                    )
                if (
                    wandb_logger is not None
                    and (i + 1) % FLAGS.config.log_interval == 0
                ):
                    wandb_logger.log(
                        {
                            "training": critic_update_info,
                            "batch_info": batch_info,
                        },
                        step=i,
                    )
                    wandb_logger.log(
                        {f"debug/{k}": float(v) for k, v in debug_metrics.items()},
                        step=i,
                    )
            timer.tock("wandb_logging")

        timer.tock("critic_training/total")
        # print("critic trianing done ---- ")

        if (
            (i + 1) % FLAGS.config.eval_interval == 0 or i == FLAGS.num_offline_epochs
        ) and eval_env is not None:
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
                        trajectories = evaluate_with_trajectories_libero(
                        eval_policy_fn,
                        eval_env,
                        FLAGS.config.num_eval_episodes,
                        action_horizon=config.model.action_horizon
                    )

                # # log Q - MC
                # if hasattr(agent, "forward_critic"):
                #     timer.tick("q-mc calculation")
                #     initial_states = []
                #     for t in trajectories:
                #         observations = sanitize_obs(t["observation"][0])
                #         observations["image"] = resize_images_to_100x100(
                #             observations["image"]
                #         )
                #         if FLAGS.use_wrist_view:
                #             observations["wrist_image"] = resize_images_to_100x100(
                #                 observations["wrist_image"]
                #             )
                #         initial_states.append(observations)
                        
                #     breakpoint()
                #     initial_states = jax.tree_map(
                #         lambda *x: jnp.stack(x), *initial_states
                #     )
                #     initial_actions = [t["action"][0] for t in trajectories]
                #     initial_actions = jax.tree_map(
                #         lambda *x: jnp.stack(x), *initial_actions
                #     )
                #     initial_qs = agent.forward_critic(
                #         initial_states, initial_actions, rng=None, train=False
                #     ).mean(axis=0)
                #     mc_returns = jax.tree_map(
                #         lambda t: calc_return_to_go(
                #             rewards=np.array(t["reward"]) * FLAGS.reward_scale
                #             + FLAGS.reward_bias,
                #             masks=1 - np.array(t["done"]),
                #             gamma=FLAGS.config.agent_kwargs.discount,
                #             push_failed_to_min="maze" in FLAGS.environment_name
                #             or FLAGS.environment_name == "real_robot",
                #             min_reward=FLAGS.reward_bias,
                #         ),
                #         trajectories,
                #         is_leaf=lambda x: isinstance(
                #             x, dict
                #         ),  # only map over traj in trajs
                #     )
                #     initial_mc_returns = jax.tree_map(lambda t: t[0], mc_returns)

                #     timer.tock("q-mc calculation")
                #     if FLAGS.plot_q_values_over_trajectory_figure:
                #         timer.tick("q_values_over_trajectory")
                #         q_values_over_trajectory_time_step_figure = (
                #             plot_q_values_over_trajectory_time_step(
                #                 trajectories=trajectories,
                #                 critic_agent=agent,
                #                 sharding=sharding,
                #             )
                #         )
                #     else:
                #         q_values_over_trajectory_time_step_figure = None
                #     timer.tock("q_values_over_trajectory")
                #     if wandb_logger is not None:
                #         wandb.log(
                #             {
                #                 "eval/initial state Q": wandb.Histogram(initial_qs),
                #                 "eval/initial state MC": wandb.Histogram(
                #                     initial_mc_returns
                #                 ),
                #                 "eval/Q - MC": wandb.Histogram(
                #                     np.array(initial_qs) - np.array(initial_mc_returns)
                #                 ),
                #                 "eval/q_values_over_trajectory_time_step": q_values_over_trajectory_time_step_figure,
                #             },
                #             step=i,
                #         )

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
                        
                    # frames = np.array(frames)
                    # wandb.log(
                    #     {
                    #         "video": wandb.Video(
                    #             frames,
                    #             fps=24,
                    #             format="mp4",
                    #         )
                    #     },
                    #     step=i,
                    # )
                    # print("video logged")
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

        if i % FLAGS.config.save_interval == 0:
            logging.info("Saving checkpoint...")
            checkpoint_path = tf.io.gfile.join(save_dir, "agent_checkpoints")
            agent.save_checkpoint(checkpoint_path, step=i)

            logging.info("Saved checkpoint to %s", checkpoint_path)
            if i >= FLAGS.num_offline_epochs and FLAGS.num_online_epochs > 0 and base_policy_agent is not None:
                # Save base policy
                base_policy_save_dir = tf.io.gfile.join(
                    save_dir,
                    "base_policy_checkpoints",
                )
                base_policy_agent.save_checkpoint(base_policy_save_dir, step=i)

                logging.info("Saved ddpm checkpoint to %s", base_policy_save_dir)

        timer.tock("total")

        if wandb_logger is not None and i % FLAGS.config.log_interval == 0:
            wandb_logger.log(
                {"timer/total_times": timer.get_total_times(reset=False)}, step=i
            )
            wandb_logger.log({"timer/average_times": timer.get_average_times()}, step=i)

        if FLAGS.train_on_separate_computer_mode == "single_computer":
            online_env_steps_this_epoch = 0
        else:
            # Wait until peer is done with their part.
            if FLAGS.train_on_separate_computer_mode == "env_steps_only":
                # Save steps_elapsed to a file.
                assert online_env_steps_this_epoch > 0
                with tf.io.gfile.GFile(
                    os.path.join(save_dir, "latest_steps_elapsed.txt"), "w"
                ) as f:
                    f.write(str(online_env_steps_this_epoch))
                online_env_steps_this_epoch = 0
                # Wait until checkpoint for this epoch is available, and load it.
                restored_checkpoint = False
                while not restored_checkpoint:
                    # "checkpoint_{epoch}"
                    critic_path = tf.io.gfile.join(
                        save_dir,
                        "critic_checkpoints_from_agent_trainer",
                        f"checkpoint_{i}",
                    )
                    base_policy_path = tf.io.gfile.join(
                        save_dir,
                        "base_policy_checkpoints_from_agent_trainer",
                        f"checkpoint_{i}",
                    )

                    if tf.io.gfile.exists(critic_path) and tf.io.gfile.exists(
                        base_policy_path
                    ):
                        # move critic and agent into temporary local file
                        agent = agent.restore_checkpoint(critic_path)
                        agent.to_device(sharding)

                        base_policy_agent = base_policy_agent.restore_checkpoint(
                            base_policy_path
                        )

                        restored_checkpoint = True

                    else:
                        print(
                            f"Waiting for the checkpoint for epoch {i} to be available."
                        )
                        time.sleep(0.5)
                        continue
            elif FLAGS.train_on_separate_computer_mode == "agent_training_only":
                # Save the checkpoint for this epoch.
                critic_path = tf.io.gfile.join(
                    save_dir,
                    "critic_checkpoints_from_agent_trainer",
                )
                agent.save_checkpoint(critic_path, step=i)

                # Wait until newest episode appears on the replay buffer.
                newest_episode_exists = False
                while not newest_episode_exists:
                    if tf.io.gfile.exists(
                        os.path.join(
                            save_dir,
                            "image_replay_buffer",
                            f"episode_{online_trajectories_added - 1}.tfrecord",
                        )
                    ):
                        newest_episode_exists = True
                        online_trajectories_added += (
                            FLAGS.num_online_trajectories_per_epoch
                        )
                    else:
                        print(
                            f"Waiting for episode_{online_trajectories_added-1}.tfrecord to be available."
                        )
                        time.sleep(0.5)

                # Wait until the file exists
                steps_elapsed_file_path = os.path.join(
                    save_dir, "latest_steps_elapsed.txt"
                )
                while not tf.io.gfile.exists(steps_elapsed_file_path):
                    print(f"Waiting for {steps_elapsed_file_path} to be created...")
                    time.sleep(1)  # Wait for 1 second before checking again

                print(
                    f"{steps_elapsed_file_path} has been created. Reading the file..."
                )

                # Load steps_elapsed from file, add it to online_env_steps.
                with tf.io.gfile.GFile(
                    os.path.join(save_dir, "latest_steps_elapsed.txt"), "r"
                ) as f:
                    online_env_steps_this_epoch = int(f.read())
                online_env_steps += online_env_steps_this_epoch

            else:
                raise ValueError(
                    f"Invalid train_on_separate_computer_mode: {FLAGS.train_on_separate_computer_mode}"
                )
    base_policy_agent.checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    app.run(train_agent)
