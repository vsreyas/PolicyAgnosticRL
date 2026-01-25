"""Visualize state-value predictions along trajectories from tfrecord files."""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from absl import app, flags, logging
from PIL import Image

from jaxrl_m.agents.continuous.pi_vlm_cached.residual_ppo import PiResidualPPOCache
from jaxrl_m.data.bridge_dataset import glob_to_path_list
from jaxrl_m.data.image_replay_buffer_pi import ImageReplayBufferPi
from openpi.training.config import get_config

FLAGS = flags.FLAGS

flags.DEFINE_string("tfrecord_path", None, "Path to tfrecord file (can be glob pattern).")
flags.DEFINE_string("checkpoint_path", None, "Path to model checkpoint (.pkl file).")
flags.DEFINE_string("output_path", "./trajectory_visualization.mp4", "Output path for visualization.")
flags.DEFINE_string("task_name", "put both moka pots on the stove", "Task name.")
flags.DEFINE_string("pi_config_name", "pi05_libero_custom_low_mem_ep5", "PI config name.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("trajectory_index", 0, "Index of trajectory to load from tfrecord.")
flags.DEFINE_bool("use_wrist_view", True, "Use wrist view camera.")
flags.DEFINE_bool("use_lang", True, "Use language conditioning.")
flags.DEFINE_integer("fps", 10, "FPS for output video.")
flags.DEFINE_integer("output_dim", 1024, "Output video dimension (width and height).")


def load_trajectory_from_tfrecord(
    tfrecord_path: str,
    task_name: str,
    pi_config: any,
    trajectory_index: int = 0,
    seed: int = 0,
    use_wrist_view: bool = True,
    use_lang: bool = True,
) -> Dict[str, np.ndarray]:
    """Load a full trajectory from tfrecord file.

    Args:
        tfrecord_path: Path to tfrecord file (single file, not glob pattern)
        task_name: Name of the task
        pi_config: PI configuration object
        trajectory_index: Index of trajectory to load (default: 0 for first trajectory)
        seed: Random seed
        use_wrist_view: Whether to use wrist view
        use_lang: Whether to use language conditioning

    Returns:
        Dictionary containing trajectory data with keys like 'observations', 'actions', etc.
        Each value is an array of shape (T, ...) where T is the trajectory length.
    """
    # Prevent TensorFlow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    if '*' in tfrecord_path:
        data_paths = glob_to_path_list(tfrecord_path)
        logging.info(f"Found {len(data_paths)} tfrecord files, using file {trajectory_index}")
        tfrecord_path = data_paths[trajectory_index]

    logging.info(f"Loading trajectory from {tfrecord_path}")

    # Parse tfrecord file directly to get full trajectory
    from jaxrl_m.envs.libero import get_libero_tfrecord_dataset

    dataset = get_libero_tfrecord_dataset(
        tfrecord_regexp=tfrecord_path,
        use_wrist_view=use_wrist_view,
        use_language=use_lang,
        config=pi_config,
        is_pi=True,
        train=False,
        task_name=task_name,
        final_step_sparse_reward=False,
        filter_successful_trajectories=False,
    )

    # Get the full trajectory by collecting all timesteps
    trajectory_data = {
        'observations': {'image': [], 'proprio': []},
        'actions': [],
        'rewards': [],
        'terminals': [],
        'vlm_output': [],
    }

    if use_wrist_view:
        trajectory_data['observations']['image_wrist'] = []

    # Iterate through the dataset to collect all steps
    dataset_iter = dataset.iterator(batch_size=1)

    for i, batch in enumerate(dataset_iter):
        # Collect data
        trajectory_data['observations']['image'].append(batch['observations']['image'][0])
        trajectory_data['observations']['proprio'].append(batch['observations']['proprio'][0])

        if use_wrist_view and 'image_wrist' in batch['observations']:
            trajectory_data['observations']['image_wrist'].append(batch['observations']['image_wrist'][0])

        trajectory_data['actions'].append(batch['actions'][0])
        trajectory_data['rewards'].append(batch['rewards'][0])
        trajectory_data['terminals'].append(batch['terminals'][0])

        if 'vlm_output' in batch:
            trajectory_data['vlm_output'].append(batch['vlm_output'][0])

        # Stop if we've reached the end of the trajectory
        if batch['terminals'][0] or i >= 1000:  # safety limit
            break

    # Convert lists to arrays
    trajectory = {
        'observations': {
            'image': np.array(trajectory_data['observations']['image']),
            'proprio': np.array(trajectory_data['observations']['proprio']),
        },
        'actions': np.array(trajectory_data['actions']),
        'rewards': np.array(trajectory_data['rewards']),
        'terminals': np.array(trajectory_data['terminals']),
    }

    if use_wrist_view and len(trajectory_data['observations']['image_wrist']) > 0:
        trajectory['observations']['image_wrist'] = np.array(trajectory_data['observations']['image_wrist'])

    if len(trajectory_data['vlm_output']) > 0:
        trajectory['vlm_output'] = np.array(trajectory_data['vlm_output'])

    logging.info(f"Loaded trajectory with {len(trajectory['actions'])} timesteps")

    return trajectory


def load_model(checkpoint_path: str, pi_config: any, seed: int = 0) -> PiResidualPPOCache:
    """Load trained PPO model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint .pkl file
        pi_config: PI configuration object
        seed: Random seed

    Returns:
        Loaded PPO agent
    """
    logging.info(f"Loading checkpoint from {checkpoint_path}")

    rng = jax.random.PRNGKey(seed)

    agent = PiResidualPPOCache.create(
        config=pi_config,
        seed=seed,
        batch_size=1,  # batch_size for initialization
        rng=rng,
        N=4,
        n_edit_samples=4,
        params_path=checkpoint_path,
    )

    logging.info("Model loaded successfully")
    return agent


def compute_state_values(agent: PiResidualPPOCache, vlm_outputs: np.ndarray) -> np.ndarray:
    """Compute state values for trajectory using trained critic.

    Args:
        agent: Trained PPO agent
        vlm_outputs: VLM outputs for each timestep (T, hidden_dim)

    Returns:
        State values for each timestep (T,)
    """
    state_values = []
    for vlm_output in vlm_outputs:
        value = agent.get_state_values(vlm_output)
        state_values.append(float(value[0]))

    return np.array(state_values)


def create_value_plot(state_values: np.ndarray, current_step: int) -> np.ndarray:
    """Create a plot of state values up to current step.

    Args:
        state_values: Array of state values
        current_step: Current timestep to plot up to

    Returns:
        Image array (H, W, 3) in uint8 format
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    steps = np.arange(current_step + 1)
    values = state_values[:current_step + 1]

    ax.plot(steps, values, 'b-', linewidth=2)
    ax.scatter([current_step], [values[-1]], color='red', s=100, zorder=5)

    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('State Value', fontsize=12)
    ax.set_title('Predicted State Values', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Set consistent y-axis limits
    if len(state_values) > 0:
        y_min, y_max = state_values.min(), state_values.max()
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    fig.tight_layout()

    # Convert plot to image
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img = img[:, :, :3]  # Convert RGBA to RGB

    plt.close(fig)

    return img


def resize_image(img: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize image to target dimensions.

    Args:
        img: Input image (H, W, 3)
        height: Target height
        width: Target width

    Returns:
        Resized image
    """
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255.0).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((width, height), resample=Image.BILINEAR)

    return np.array(pil_img)


def create_visualization_video(
    trajectory: Dict[str, np.ndarray],
    state_values: np.ndarray,
    output_path: str,
    fps: int = 10,
    dim: int = 1024,
) -> None:
    """Create side-by-side visualization video.

    Args:
        trajectory: Trajectory data dictionary
        state_values: Predicted state values
        output_path: Path to save output video
        fps: Frames per second
        dim: Output video dimension (width and height)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract images from trajectory
    observations = trajectory['observations']
    images = observations['image']  # Shape: (T, H, W, 3)

    if images.ndim != 4:
        raise ValueError(f"Expected 4D image array (T, H, W, 3), got shape {images.shape}")

    num_frames = len(images)
    half_dim = dim // 2

    logging.info(f"Creating visualization with {num_frames} frames")

    with imageio.get_writer(
        str(output_path),
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
    ) as writer:
        for t in range(num_frames):
            # Get trajectory image
            traj_img = images[t]
            traj_img = resize_image(traj_img, dim, half_dim)

            # Create value plot
            value_plot = create_value_plot(state_values, t)
            value_plot = resize_image(value_plot, dim, half_dim)

            # Combine side by side: left=value plot, right=trajectory image
            combined = np.concatenate([value_plot, traj_img], axis=1)

            writer.append_data(combined)

    logging.info(f"Visualization saved to {output_path}")


def main(_):
    # Validate required flags
    if FLAGS.tfrecord_path is None:
        raise ValueError("--tfrecord_path must be provided")
    if FLAGS.checkpoint_path is None:
        raise ValueError("--checkpoint_path must be provided")

    # Setup JAX
    devices = jax.local_devices()
    logging.info(f"Found {len(devices)} JAX devices")

    # Load PI config
    pi_config = get_config(FLAGS.pi_config_name)
    pi_config.fsdp_devices = 1
    pi_config.exp_name = "trajectory_visualization"
    pi_config.overwrite = True

    # Load trajectory
    logging.info("Loading trajectory from tfrecord...")
    trajectory = load_trajectory_from_tfrecord(
        tfrecord_path=FLAGS.tfrecord_path,
        task_name=FLAGS.task_name,
        pi_config=pi_config,
        trajectory_index=FLAGS.trajectory_index,
        seed=FLAGS.seed,
        use_wrist_view=FLAGS.use_wrist_view,
        use_lang=FLAGS.use_lang,
    )

    # Load model
    logging.info("Loading trained model...")
    agent = load_model(
        checkpoint_path=FLAGS.checkpoint_path,
        pi_config=pi_config,
        seed=FLAGS.seed,
    )

    # Compute state values
    logging.info("Computing state values...")
    vlm_outputs = trajectory['vlm_output']  # Shape: (T, hidden_dim)
    state_values = compute_state_values(agent, vlm_outputs)

    logging.info(f"State value statistics:")
    logging.info(f"  Mean: {state_values.mean():.4f}")
    logging.info(f"  Std: {state_values.std():.4f}")
    logging.info(f"  Min: {state_values.min():.4f}")
    logging.info(f"  Max: {state_values.max():.4f}")

    # Create visualization
    logging.info("Creating visualization video...")
    create_visualization_video(
        trajectory=trajectory,
        state_values=state_values,
        output_path=FLAGS.output_path,
        fps=FLAGS.fps,
        dim=FLAGS.output_dim,
    )

    logging.info("Done!")


if __name__ == "__main__":
    app.run(main)
