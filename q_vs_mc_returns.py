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
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

FLAGS = flags.FLAGS

flags.DEFINE_string("environment_name", "", "Environment name.")
flags.DEFINE_string("wandb_project_name", 'debug-plots-q-vs-mc', "WandB project name.") #"PA-RL""debug" "PI-0.5-finetuning"
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
flags.DEFINE_integer("num_offline_epochs", 1, "Number of epochs for pre-training.")
flags.DEFINE_integer(
    "num_online_epochs", 1, "Number of epochs for online fine-tuning."
)
flags.DEFINE_integer(
    "num_train_steps_per_offline_epoch", 500, "Number of training steps per epoch."
)
flags.DEFINE_float("reward_scale", 1.0, "Reward scale.")
flags.DEFINE_float("reward_bias", 0.0, "Reward bias.")
flags.DEFINE_float("clip_action", 0.99999, "Clip action.")
flags.DEFINE_integer("num_parallel_envs", 1, "Number of parallel environments.")
flags.DEFINE_bool("debug", False, "Debug config")
flags.DEFINE_string("ckpt_folder", "/data/user_data/sreyasv/q_chunking_pi_no_-1_10_critic_ws_debug/libero_10_pi05_put_the_two_mocha_pots_on_the_stove/debug_plots/results/debug-runs/pi0_5_finetuning_parl_qchunking_-1_10_critic_ws_debug/seed_0/agent_checkpoints", "Resume training from checkpoint.")
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

        if "ddpm" in FLAGS.config.agent:

            if obs_ndim == 2:
                observations = jax.tree_map(lambda x: x[:, None], observations)
            elif obs_ndim == 1:
                observations = jax.tree_map(lambda x: x[None, None], observations)
        
        
        observations["image"] = resize_images_to_100x100(
            observations["image"]
        )
        if FLAGS.use_wrist_view:
            observations["wrist_image"] = resize_images_to_100x100(
                observations["wrist_image"]
            )
        # print("Observations: ", observations['image'].shape)
        #TODO this is not right, since the observation preprocessing happens inside the dataloader
        actions = jax.device_get(
            agent.sample_actions(
                observations, *args, **kwargs, argmax=argmax, timer=timer
            )
        )
        if "parl" in FLAGS.config.agent:
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




def visualize_q_diagnostics_trajectory(
    q_values,
    mc_returns,
    ckpt_num: int,
    mode: str,  # "offline" or "online"
    trajectory_id: int = 0,
    save_dir: str = "./plots",
):
    """
    Visualize Q-function diagnostics for a single trajectory.

    Args:
        q_values (np.ndarray): Shape (T,), predicted Q-values
        mc_returns (np.ndarray): Shape (T,), Monte Carlo returns
        ckpt_num (int): Checkpoint number
        mode (str): "offline" or "online"
        trajectory_id (int): Trajectory index (for title)
        save_dir (str): Base directory for saving plots
    """
    assert mode in {"offline", "online"}, "mode must be 'offline' or 'online'"
    assert len(q_values) == len(mc_returns), "Q and MC must have same length"

    q_values = np.asarray(q_values)
    mc_returns = np.asarray(mc_returns)

    save_dir = os.path.join(save_dir, "q_diagnostics_plots", mode)
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(24, 7), constrained_layout=True)
    fig.suptitle(
        f"Q-Function Diagnostics | {mode.upper()} Trajectory\n"
        f"Checkpoint {ckpt_num} | Trajectory {trajectory_id}",
        fontsize=16,
    )

    # -------------------------------------------------------
    # Subplot 1: Q vs MC scatter
    # -------------------------------------------------------
    ax1 = axes[0]
    sc1 = ax1.scatter(
        q_values,
        mc_returns,
        c=mc_returns,
        cmap="plasma",
        alpha=0.6,
        s=20,
        edgecolors="none",
    )
    fig.colorbar(sc1, ax=ax1, label="MC Return")

    min_val = min(q_values.min(), mc_returns.min())
    max_val = max(q_values.max(), mc_returns.max())
    ax1.plot(
        [min_val, max_val],
        [min_val, max_val],
        "--",
        color="red",
        linewidth=2,
        label="y = x",
    )

    ax1.set_title("Predicted Q vs MC Return")
    ax1.set_xlabel("Q(s, a)")
    ax1.set_ylabel("MC Return")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal", "box")

    # -------------------------------------------------------
    # Subplot 2: Temporal value drop consistency
    # -------------------------------------------------------
    ax2 = axes[1]
    q_diffs = q_values[:-1] - q_values[1:]
    mc_diffs = mc_returns[:-1] - mc_returns[1:]

    ax2.scatter(
        q_diffs,
        mc_diffs,
        alpha=0.4,
        s=20,
        edgecolors="none",
        color="royalblue",
    )

    min_val = min(q_diffs.min(), mc_diffs.min())
    max_val = max(q_diffs.max(), mc_diffs.max())
    ax2.plot(
        [min_val, max_val],
        [min_val, max_val],
        "--",
        color="red",
        linewidth=2,
        label="y = x",
    )

    ax2.set_title("Temporal Value Drop Consistency")
    ax2.set_xlabel(r"$Q_t - Q_{t+1}$")
    ax2.set_ylabel(r"$MC_t - MC_{t+1}$")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal", "box")

    # -------------------------------------------------------
    # Subplot 3: t-SNE of (Q, MC)
    # -------------------------------------------------------
    
    ax3 = axes[2]

    try:
        X = np.stack([q_values, mc_returns], axis=1)
        X_scaled = StandardScaler().fit_transform(X)

        tsne = TSNE(
            n_components=2,
            perplexity=min(30, len(X) - 1),
            learning_rate="auto",
            init="pca",
            max_iter=500,
        )
        embeddings = tsne.fit_transform(X_scaled)

        sc3 = ax3.scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            c=mc_returns,
            cmap="viridis",
            alpha=0.7,
            s=20,
            edgecolors="none",
        )
        fig.colorbar(sc3, ax=ax3, label="MC Return")

        ax3.set_title("t-SNE of (Q, MC) Pairs")
        ax3.set_xlabel("t-SNE Dim 1")
        ax3.set_ylabel("t-SNE Dim 2")
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect("equal", "box")

    except Exception as e:
        # Clear axis and show empty placeholder
        ax3.cla()
        ax3.set_title("t-SNE of (Q, MC) Pairs (failed)")
        ax3.text(
            0.5, 0.5,
            "t-SNE failed",
            ha="center",
            va="center",
            transform=ax3.transAxes,
            fontsize=10,
            alpha=0.7,
        )
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.grid(False)


    # -------------------------------------------------------
    # Save
    save_dir_diags = os.path.join(save_dir, "q_mc_diagnostics")
    os.makedirs(save_dir_diags, exist_ok=True)
    save_path = os.path.join(
        save_dir_diags,
        f"ckpt_{ckpt_num}_traj_{trajectory_id}_{mode}.png",
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")

    print(f"[Saved] Q diagnostics ({mode}) → {save_path}")
    return fig

def visualize_q_over_time(
    q_values,
    ckpt_num: int,
    mode: str,  # "offline" or "online"
    trajectory_id: int = 0,
    save_dir: str = "./plots",
):
    """
    Visualize Q-values over timestep for a single trajectory.

    Args:
        q_values (np.ndarray): Shape (T,), predicted Q-values
        ckpt_num (int): Checkpoint number
        mode (str): "offline" or "online"
        trajectory_id (int): Trajectory index (for title)
        save_dir (str): Base directory for saving plots
    """
    assert mode in {"offline", "online"}, "mode must be 'offline' or 'online'"

    q_values = np.asarray(q_values)
    assert q_values.ndim == 1, "q_values must be 1D"

    timesteps = np.arange(len(q_values))

    # -------------------------------------------------------
    # Directory setup
    # -------------------------------------------------------
    save_dir = os.path.join(save_dir, "q_value_plots", mode)
    os.makedirs(save_dir, exist_ok=True)

    # -------------------------------------------------------
    # Plot
    # -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        f"Q-Value Trajectory | {mode.upper()}\n"
        f"Checkpoint {ckpt_num} | Trajectory {trajectory_id}",
        fontsize=14,
    )

    ax.plot(
        timesteps,
        q_values,
        linewidth=2,
    )

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Q(s, a)")
    # ax.set_title("Predicted Q-values over Time")
    ax.grid(True, alpha=0.3)

    # Optional reference line at zero (often useful for diagnostics)
    ax.axhline(0.0, linestyle="--", linewidth=1, alpha=0.5)

    # -------------------------------------------------------
    # Save
    # -------------------------------------------------------
    save_dir_q_values = os.path.join(save_dir, "q_values_over_time")
    os.makedirs(save_dir_q_values, exist_ok=True)
    save_path = os.path.join(
        save_dir_q_values,
        f"ckpt_{ckpt_num}_traj_{trajectory_id}_{mode}.png",
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")

    print(f"[Saved] Q-values over time ({mode}) → {save_path}")
    return fig




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
    config = get_config("pi05_libero_custom_low_mem")
    config.exp_name = FLAGS.wandb_experiment_name
    config.overwrite = True

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
            wandb_logger.config.project,
            wandb_logger.config.exp_descriptor,
            f"seed_{FLAGS.seed}",
        )
    else:
        wandb_logger = None
        save_dir = tf.io.gfile.join(
            os.path.abspath(FLAGS.config.save_dir),
        )
        config.wandb_enabled = False

    # Create environment and dataset
    action_space = None
    
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
        FLAGS.config.dataset_kwargs['traj_sampling'] = True
        traj_dataset = get_libero_tfrecord_dataset(
            tfrecord_regexp=FLAGS.config.libero_tfrecord_regexp, use_wrist_view=FLAGS.use_wrist_view, 
            use_language=FLAGS.use_lang, config=config, is_pi=True, **FLAGS.config.dataset_kwargs,
            task_name=FLAGS.task_name,
        )
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

    rng = jax.random.PRNGKey(FLAGS.seed)
    # we shard the leading dimension (batch dimension) accross all devices evenly
    # sharding = jax.sharding.PositionalSharding(devices)
    sharding = jax.sharding.PositionalSharding(devices)
    # Create data iterators
    offline_train_iterator_for_critic = dataset.iterator(
        batch_size=FLAGS.config.batch_size)

    # Optionally create base policy agent
    base_policy_path_components = FLAGS.config.get("base_policy_path", "").split(":")
    if len(base_policy_path_components) == 1 and base_policy_path_components[0] == "":
        # This means the agent we're training doesn't use a base policy
        # E.g. we could be pre-training the base policy itself.
        base_policy_agent = None
        base_policy_type = None
    else:
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
    print("data loader construction done: -----------"  )
    print("base policy agent loaded: ", base_policy_agent)
    example_batch = next(offline_train_iterator_for_critic)


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
 
    # initialize agent
    rng, construct_rng = jax.random.split(rng)

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
    
    if base_policy_agent is not None:
        base_policy_agent.to_device(sharding)
    print("PARL agent construction done: -----------")
    timer = Timer()
    
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
    data_collection_trajectory_sampler = TrajSampler(
        train_env,
        clip_action=FLAGS.clip_action,
        reward_scale=FLAGS.reward_scale,
        reward_bias=FLAGS.reward_bias,
        max_traj_length=FLAGS.config.get("max_episode_steps", 1000),
    )

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
   
    traj = data_collection_trajectory_sampler.sample(
                    eval_policy_fn,
                    num_episodes=1,
                    replay_buffer=None,
                    calc_mc_return_fn=calc_mc_return_fn,
                    store_max_trajectory_reward=True,
                    terminate_on_success=FLAGS.config.get(
                        "early_terminate_on_success", False
                    ),
                )[0]
    save_trajectory_as_tfrecord(
                        trajectory=traj, 
                        path=tf.io.gfile.join(
                            save_dir,
                            "image_replay_buffer",
                            f"episode_test.tfrecord",
                        ),
                    )
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
        traj_sampling=True,
        **FLAGS.config.image_replay_buffer_kwargs,
    )
    online_eval_iterator = image_replay_buffer.iterator(
                    batch_size=1)
    offline_eval_iterator = traj_dataset.iterator(batch_size=1)
    
    offline_trajectory =  next(offline_eval_iterator)
    online_trajectory = next(online_eval_iterator)

    offline_trajectory = jax.tree_map(lambda x: x[0], offline_trajectory)
    online_trajectory = jax.tree_map(lambda x: x[0], online_trajectory)

    # breakpoint()
    offline_trajectory = shard_batch(offline_trajectory, sharding)
    online_trajectory = shard_batch(online_trajectory, sharding)

    offline_trajectory['observations']["image"] = resize_images_to_100x100(
                    offline_trajectory["observations"]["image"]
                )
    online_trajectory["observations"]["image"] = resize_images_to_100x100(
                    online_trajectory["observations"]["image"]
                )
    if FLAGS.use_wrist_view:
        offline_trajectory["observations"]["wrist_image"] = resize_images_to_100x100(
                        offline_trajectory["observations"]["wrist_image"]
                    )
        online_trajectory["observations"]["wrist_image"] = resize_images_to_100x100(
                        online_trajectory["observations"]["wrist_image"]
                    )
    offline_trajectory['actions'] = preprocess_action(offline_trajectory['actions'])
    online_trajectory['actions'] = preprocess_action(online_trajectory['actions'])

    print("Online and offline trajectories obtained.")
    print("Length of offline trajectory observations: ", len(offline_trajectory['observations']['proprio']))
    print("Length of online trajectory observations: ", len(online_trajectory['observations']['proprio']))

    print("Checkpoint folder: ", FLAGS.ckpt_folder)
    # List all directories in FLAGS.ckpt_folder
    ckpt_folders = [
        os.path.join(FLAGS.ckpt_folder, folder) for folder in os.listdir(FLAGS.ckpt_folder)
        if os.path.isdir(os.path.join(FLAGS.ckpt_folder, folder))
    ]
    
    plots_save_dir = os.path.join(save_dir, "plots")
    os.makedirs(plots_save_dir, exist_ok=True)
    # Iterate over each checkpoint folder
    for ckpt_path in ckpt_folders:
        # Extract checkpoint number (folder name)
        checkpoint_number = os.path.basename(ckpt_path)
        checkpoint_number = int(checkpoint_number.rsplit("_", 1)[1])
        
        # Restore the agent from the checkpoint
        agent = agent.restore_checkpoint(ckpt_path)
        agent.to_device(sharding)

        # log Q - MC
        if hasattr(agent, "forward_critic"):
            timer.tick("q-mc calculation")
            # offline_qs = agent.forward_critic(
            #     offline_trajectory["observations"], offline_trajectory["actions"], rng=None, train=False
            # ).mean(axis=0)
            obs = offline_trajectory["observations"]
            act = offline_trajectory["actions"]

            T = jax.tree_util.tree_leaves(obs)[0].shape[0]

            offline_qs = []

            for t in range(T):
                obs_t = jax.tree_map(lambda x: x[t:t+1], obs)
                act_t = jax.tree_map(lambda x: x[t:t+1], act)

                q_t = agent.forward_critic(
                    obs_t,
                    act_t,
                    rng=None,
                    train=False
                ).mean(axis=0)   # (1,)

                offline_qs.append(q_t[0])  
            offline_qs = jnp.stack(offline_qs)  

            # online_qs = agent.forward_critic(
            #     online_trajectory["observations"], online_trajectory["actions"], rng=None, train=False
            # ).mean(axis=0)
            obs = online_trajectory["observations"]
            act = online_trajectory["actions"]

            T_online = jax.tree_util.tree_leaves(obs)[0].shape[0]
            online_qs = []

            for t in range(T_online):
                obs_t = jax.tree_map(lambda x: x[t:t+1], obs)
                act_t = jax.tree_map(lambda x: x[t:t+1], act)

                q_t = agent.forward_critic(
                    obs_t,
                    act_t,
                    rng=None,
                    train=False
                ).mean(axis=0)   # (1,)

                online_qs.append(q_t[0])  
            online_qs = jnp.stack(online_qs)

            offline_mc_returns = offline_trajectory['mc_returns']
            online_mc_returns = online_trajectory['mc_returns']
            timer.tock("q-mc calculation")

            offline_fig = visualize_q_diagnostics_trajectory(
                q_values=offline_qs,
                mc_returns=offline_mc_returns,
                ckpt_num=checkpoint_number,
                mode="offline",
                trajectory_id=0,
                save_dir=plots_save_dir,
            )

            online_fig = visualize_q_diagnostics_trajectory(
                q_values=online_qs,
                mc_returns=online_mc_returns,
                ckpt_num=checkpoint_number,
                mode="online",
                trajectory_id=0,
                save_dir=plots_save_dir,
            )

            offline_q_time_fig = visualize_q_over_time(
                q_values=offline_qs,
                ckpt_num=checkpoint_number,
                mode="offline",
                trajectory_id=0,
                save_dir=plots_save_dir,
            )

            online_q_time_fig = visualize_q_over_time(
                q_values=online_qs,
                ckpt_num=checkpoint_number,
                mode="online",
                trajectory_id=0,
                save_dir=plots_save_dir,
            )
            
            # if wandb_logger is not None:
            #     log_dict = {}

            #     # -------------------------------
            #     # OFFLINE logging
            #     # -------------------------------
            #     log_dict.update({
            #         "eval/offline/initial_state_Q": wandb.Histogram(offline_qs[:1]),
            #         "eval/offline/initial_state_MC": wandb.Histogram(offline_mc_returns[:1]),
            #         "eval/offline/Q_minus_MC": wandb.Histogram(
            #             np.asarray(offline_qs) - np.asarray(offline_mc_returns)
            #         ),
            #         "eval/offline/q_mc_scatter_traj": wandb.Image(offline_fig),
            #         "eval/offline/q_time_traj": wandb.Image(offline_q_time_fig),
            #     })

            #     # -------------------------------
            #     # ONLINE logging
            #     # -------------------------------
            #     log_dict.update({
            #         "eval/online/initial_state_Q": wandb.Histogram(online_qs[:1]),
            #         "eval/online/initial_state_MC": wandb.Histogram(online_mc_returns[:1]),
            #         "eval/online/Q_minus_MC": wandb.Histogram(
            #             np.asarray(online_qs) - np.asarray(online_mc_returns)
            #         ),
            #         "eval/online/q_mc_scatter_traj": wandb.Image(online_fig),
            #         "eval/online/q_time_traj": wandb.Image(online_q_time_fig),
            #     })

            #     wandb.log(
            #         log_dict,
            #         step=int(checkpoint_number),
            #     )


                
    frames = []
    ind_traj = []
    
    trajectory_return = 0
    for transition, reward in zip(traj["observation"], traj["reward"]):
        assert transition["image"].shape[-1] == 3
        if len(transition["image"].shape) == 4:
            transition["image"] = transition["image"][0]
        image = transition["image"]  # .transpose(2, 0, 1)
        trajectory_return += reward
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
    
        save_rollout_gif(ind_traj, save_dir, step_i=0, rollout_j=0)
        ind_traj = []
                
        del ind_traj, frames
        import gc; gc.collect()




if __name__ == "__main__":
    app.run(train_agent)
