from __future__ import annotations

import os
import pickle
from typing import Dict, List

from critic_ws_pg import critic_params
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from absl import app, flags, logging
from ml_collections import config_flags
from tqdm import tqdm

from jaxrl_m.agents.continuous.expo_pi import ExpoPiLearner
from jaxrl_m.data.image_replay_buffer_pi import ImageReplayBufferPi
from jaxrl_m.data.bridge_dataset import glob_to_path_list
from jaxrl_m.utils.timer_utils import Timer
from openpi.training.config import get_config
import openpi.models.model as _model
from functools import partial
import flax.linen as nn
from jaxrl_m.utils.expo_utils import (
    StateActionValue,
    Ensemble,
    MLP,
)
from flax.training.train_state import TrainState
import optax
from jaxrl_m.agents.continuous.expo_pi import compute_q_all

import umap

from pathlib import Path
from typing import Union, Optional
import matplotlib.pyplot as plt
import matplotlib as mpl


print("\n\n\n IMPORTS DONE \n\n\n")
print("="*80)

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)
flags.DEFINE_string(
    "task_name",
    "put both moka pots on the stove", 
    "Name of fixed task"
)
flags.DEFINE_string("environment_name", "libero", "Environment name.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string(
    "output_path", "./critic_update_cache.pkl", "Path to save the cached data for critic updates."
)
flags.DEFINE_string(
    "replay_buffer_path", "", "Path to replay buffer tfrecords (can be a glob pattern)."
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
flags.DEFINE_float(
    "scale_success_alpha",
    0.05,
    "Alpha for the reward scaling factor.",
)

def plot_actions_colored_by_q(
    q_values,
    actions,
    out_path: Union[str, os.PathLike],
    *,
    cmap: str = "viridis",
    s: float = 25.0,
    alpha: float = 0.9,
    dpi: int = 200,
    title: Optional[str] = "Actions colored by Q value",
) -> None:
    """
    Scatter-plot 2D actions with point color proportional to q_values, and save to out_path.

    Args:
        q_values: array-like, shape (N,)
        actions: array-like, shape (N, 2)
        out_path: file path to save the plot (directories are created if needed)
        cmap: matplotlib colormap name
        s: marker size
        alpha: marker alpha
        dpi: save resolution
        title: plot title (None for no title)
    """
    q = np.asarray(q_values).reshape(-1)
    a = np.asarray(actions)

    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError(f"`actions` must have shape (N, 2). Got {a.shape}.")
    if q.shape[0] != a.shape[0]:
        raise ValueError(f"`q_values` and `actions` must have same N. Got {q.shape[0]} vs {a.shape[0]}.")

    # Keep only finite points
    mask = np.isfinite(q) & np.isfinite(a).all(axis=1)
    q = q[mask]
    a = a[mask]

    if q.size == 0:
        raise ValueError("No finite (q, action) pairs to plot after filtering.")

    qmin = float(np.min(q))
    qmax = float(np.max(q))
    if np.isclose(qmin, qmax):
        # Avoid zero range normalization
        eps = 1e-12 if abs(qmin) < 1e12 else 1.0
        qmin, qmax = qmin - eps, qmax + eps

    # Linear color mapping: q -> [0, 1] via Normalize :contentReference[oaicite:0]{index=0}
    norm = mpl.colors.Normalize(vmin=qmin, vmax=qmax, clip=True)  # :contentReference[oaicite:1]{index=1}

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(
        a[:, 0],
        a[:, 1],
        c=q,
        cmap=cmap,
        norm=norm,  # scatter supports norm/cmap for float color arrays :contentReference[oaicite:2]{index=2}
        s=s,
        alpha=alpha,
    )  # :contentReference[oaicite:3]{index=3}

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Q value")

    ax.set_xlabel("action[0]")
    ax.set_ylabel("action[1]")
    if title is not None:
        ax.set_title(title)
    ax.grid(True, linewidth=0.5, alpha=0.4)

    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")  # :contentReference[oaicite:4]{index=4}
    plt.close(fig)

def plot_q_trajectory(
    q_values: Union[np.ndarray, jnp.ndarray],
    out_path: Union[str, Path],
    *,
    step: int = 5,
    label: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: str = "Timestep",
    ylabel: str = "Q value",
) -> None:
    """
    Plot q-values across a trajectory.

    q_values: shape (T,)
    Uses x = arange(T) * step to match your reference code.
    """
    q = np.asarray(q_values).reshape(-1)
    if q.ndim != 1:
        raise ValueError(f"q_values must be 1D (T,). Got {q.shape}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(q.shape[0]) * step

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, q, label=label)

    ax.grid(True)  # :contentReference[oaicite:2]{index=2}
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)
    if label is not None:
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def process_batch(batch):
    diffusion_actions = batch['diffusion_actions']
    vlm_outputs = batch['vlm_output']
    timesteps = batch['episode_timestep']
    
    num_act_per_state = diffusion_actions.shape[1]
    diffusion_actions = diffusion_actions.reshape(-1, num_act_per_state, 70)
    vlm_outputs = vlm_outputs[:, None, :]                  # (B, 1, D)  (aka jnp.expand_dims(x, axis=1))
    vlm_outputs = jnp.repeat(vlm_outputs, num_act_per_state, axis=1)   # (B, N, D)

    # breakpoint()
    reducer = umap.UMAP()
    diffusion_actions_flat = diffusion_actions.reshape(-1, 70)
    diffusion_actions_flat_reduced = reducer.fit_transform(diffusion_actions_flat)
    diffusion_actions_reduced = diffusion_actions_flat_reduced.reshape(diffusion_actions.shape[0], diffusion_actions.shape[1], -1)
    
    chkpt = pickle.load(open('/home/skowshik/vla/codebase/PolicyAgnosticRL/results_expo_debug-td-clean_v10-scale5_actnorm/seed_0/checkpoint_4500.pkl', 'rb'))
    critic_params = chkpt['critic_params']

    # Create critic #
    critic_base_cls = partial(
        MLP,
        hidden_dims=(256, 256),
        activate_final=True,
        dropout_rate=None,
        use_layer_norm=True,
        use_pnorm=False,
        activations=nn.relu,
    )
    critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
    critic_def = Ensemble(critic_cls, num=10)
    # critic_params = critic_def.init(critic_key, dummy_observations, dummy_actions)["params"]
    critic = TrainState.create(
        apply_fn=critic_def.apply,
        params=critic_params,
        tx=optax.GradientTransformation(lambda _: None, lambda _: None),
    )

    q_vals = compute_q_all(critic.apply_fn, critic.params, vlm_outputs, diffusion_actions)

    # breakpoint()
    sampled_action = batch['actions'].reshape(-1, 70) 
    vlm_output = batch['vlm_output']
    q_vals_sampled_action = compute_q_all(critic.apply_fn, critic.params, vlm_output, sampled_action)

    return q_vals, diffusion_actions_reduced, timesteps, q_vals_sampled_action[0]

def main(_):
    # Prevent TensorFlow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    # Setup devices
    devices = jax.local_devices()
    num_devices = len(devices)
    logging.info(f"Found {num_devices} JAX devices")
    
    assert FLAGS.config.batch_size % num_devices == 0, \
        f"Batch size {FLAGS.config.batch_size} must be divisible by num_devices {num_devices}"

    # Get PI config
    pi_config = get_config("pi05_libero_custom_low_mem")
    pi_config.fsdp_devices = 1
    pi_config.exp_name = "dump_vlm_actions"
    pi_config.overwrite = True

    # Load dataset
    logging.info("Loading dataset...")
    if FLAGS.environment_name == "libero":
        from jaxrl_m.envs.libero import get_libero_tfrecord_dataset
        assert FLAGS.replay_buffer_path is not None, "Replay buffer path must be provided"

        data_paths = glob_to_path_list(FLAGS.replay_buffer_path)
        logging.info(f"Found {len(data_paths)} tfrecord files from replay buffer")
        
        # Create replay buffer iterator
        replay_buffer = ImageReplayBufferPi(
            data_paths=data_paths,
            seed=FLAGS.seed,
            train=False,
            task_name=FLAGS.task_name,
            use_wrist_view=FLAGS.use_wrist_view,
            use_language=FLAGS.use_lang,
            config=pi_config,
            final_step_sparse_reward=False,
            filter_successful_trajectories=False,
            alpha=0.05,
            scale_success_reward=False,
            **FLAGS.config.image_replay_buffer_kwargs,
        )
        dataset_iterator = replay_buffer.iterator(
            batch_size=FLAGS.config.agent_kwargs.batch_size
        )

    # Get example batch to initialize agent
    logging.info("Getting example batch...")

    dataset_iterator = replay_buffer.iterator(
        batch_size=FLAGS.config.agent_kwargs.batch_size
    )

    # Iterate through all batches
    logging.info("Processing batches...")

    with tqdm(desc="Processing batches") as pbar:
        batch = next(dataset_iterator)
        # breakpoint()
        q_vals, diffusion_actions_reduced, timesteps, q_vals_sampled_action = process_batch(batch)
        q_vals0 = q_vals[0]
        # breakpoint()
        os.makedirs(FLAGS.output_path, exist_ok=True)
        for idx, timestep in enumerate(timesteps):
            plot_actions_colored_by_q(q_vals0[idx], diffusion_actions_reduced[idx], os.path.join(FLAGS.output_path, f"q_vals_{timestep}.png"))

        plot_q_trajectory(q_vals_sampled_action, os.path.join(FLAGS.output_path, "q_vals_sampled_action.png"))
    


if __name__ == "__main__":
    app.run(main)

