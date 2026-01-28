"""Visualize state-value predictions along trajectories from tfrecord files."""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import tensorflow as tf
from absl import app, flags, logging
from ml_collections import config_flags
from PIL import Image
import umap

from jaxrl_m.agents.continuous.pi_vlm_cached import create_agent
from jaxrl_m.data.bridge_dataset import glob_to_path_list
from jaxrl_m.data.image_replay_buffer_pi import ImageReplayBufferPi
from openpi.training.config import get_config

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)
flags.DEFINE_string("tfrecord_path", None, "Path to tfrecord file (can be glob pattern).")
flags.DEFINE_string("checkpoint_path", None, "Path to model checkpoint (.pkl file).")
flags.DEFINE_string("output_path", "./trajectory_visualization.mp4", "Output path for visualization.")
flags.DEFINE_string("task_name", "put both moka pots on the stove", "Task name.")
flags.DEFINE_string("pi_config_name", "pi05_libero_custom_low_mem_ep5", "PI config name.")
flags.DEFINE_string("agent_name", "pi_residual_ppo", "Agent name (pi_residual_td3 or pi_residual_ppo).")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("trajectory_index", 0, "Index of trajectory to load from tfrecord.")
flags.DEFINE_bool("use_wrist_view", True, "Use wrist view camera.")
flags.DEFINE_bool("use_lang", True, "Use language conditioning.")
flags.DEFINE_integer("fps", 10, "FPS for output video.")
flags.DEFINE_integer("output_dim", 1024, "Output video dimension (width and height).")
flags.DEFINE_integer("vis_type", 1, "Visualization type: 1=state values, 2=UMAP Q-values.")


def load_trajectory_from_tfrecord(
    tfrecord_path: str,
    task_name: str,
    pi_config: any,
    trajectory_index: int = 0,
    seed: int = 0,
    use_wrist_view: bool = True,
    use_lang: bool = True,
    image_replay_buffer_kwargs: dict = None,
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
        image_replay_buffer_kwargs: Additional kwargs for replay buffer

    Returns:
        Dictionary containing trajectory data with keys like 'observations', 'actions', etc.
        Each value is an array of shape (T, ...) where T is the trajectory length.
    """
    # Prevent TensorFlow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    if image_replay_buffer_kwargs is None:
        image_replay_buffer_kwargs = {}

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
        **image_replay_buffer_kwargs,
    )

    # Get the full trajectory by collecting all timesteps
    trajectory_data = {
        'observations': {'image': [], 'proprio': []},
        'actions': [],
        'rewards': [],
        'terminals': [],
        'vlm_output': [],
        'action_samples': [],
        'diffusion_actions': [],
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
        
        if 'action_samples' in batch:
            trajectory_data['action_samples'].append(batch['action_samples'][0])
        
        if 'diffusion_actions' in batch:
            trajectory_data['diffusion_actions'].append(batch['diffusion_actions'][0])

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
    
    if len(trajectory_data['action_samples']) > 0:
        trajectory['action_samples'] = np.array(trajectory_data['action_samples'])
    
    if len(trajectory_data['diffusion_actions']) > 0:
        trajectory['diffusion_actions'] = np.array(trajectory_data['diffusion_actions'])

    logging.info(f"Loaded trajectory with {len(trajectory['actions'])} timesteps")

    return trajectory


def load_model(checkpoint_path: str, pi_config: any, agent_name: str = "pi_residual_ppo", seed: int = 0):
    """Load trained residual agent from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint .pkl file
        pi_config: PI configuration object
        agent_name: Name of agent to load ('pi_residual_td3' or 'pi_residual_ppo')
        seed: Random seed

    Returns:
        Loaded agent (PiResidualTD3Cache or PiResidualPPOCache)
    """
    logging.info(f"Loading {agent_name} checkpoint from {checkpoint_path}")

    rng = jax.random.PRNGKey(seed)

    agent = create_agent(
        agent_name=agent_name,
        config=pi_config,
        seed=seed,
        batch_size=1,  # batch_size for initialization
        rng=rng,
        params_path=checkpoint_path,
        N=4,
        n_edit_samples=4,
    )

    logging.info("Model loaded successfully")
    return agent


def compute_state_values(agent, vlm_outputs: np.ndarray) -> np.ndarray:
    """Compute state values for trajectory using trained critic.

    Args:
        agent: Trained residual agent (must have get_state_values method)
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
    agent,
    output_path: str,
    fps: int = 10,
    dim: int = 1024,
) -> None:
    """Create side-by-side visualization video.

    Args:
        trajectory: Trajectory data dictionary
        agent: Trained agent with get_state_values method
        output_path: Path to save output video
        fps: Frames per second
        dim: Output video dimension (width and height)
    """
    # Compute state values
    vlm_outputs = trajectory['vlm_output']
    state_values = compute_state_values(agent, vlm_outputs)
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


def plot_actions_colored_by_q(
    q_values,
    actions,
    title: Optional[str] = "Actions colored by Q value",
) -> np.ndarray:
    """Create scatter-plot of 2D actions colored by Q-values.
    
    Returns:
        Image array (H, W, 3) in uint8 format
    """
    q = np.asarray(q_values).reshape(-1)
    a = np.asarray(actions)
    
    # Keep only finite points
    mask = np.isfinite(q) & np.isfinite(a).all(axis=1)
    q = q[mask]
    a = a[mask]
    
    if q.size == 0:
        # Return blank image
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5, "No valid data", ha='center', va='center')
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = img[:, :, :3]
        plt.close(fig)
        return img
    
    qmin, qmax = float(np.min(q)), float(np.max(q))
    norm = mpl.colors.Normalize(vmin=qmin, vmax=qmax, clip=True)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(a[:, 0], a[:, 1], c=q, cmap='viridis', norm=norm, s=25, alpha=0.9)
    
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Q value")
    ax.set_xlabel("action[0]")
    ax.set_ylabel("action[1]")
    if title is not None:
        ax.set_title(title)
    ax.grid(True, linewidth=0.5, alpha=0.4)
    fig.tight_layout()
    
    # Convert to image
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img = img[:, :, :3]
    plt.close(fig)
    return img


def create_visualization_video_umap(
    trajectory: Dict[str, np.ndarray],
    agent,
    output_path: str,
    fps: int = 10,
    dim: int = 1024,
) -> None:
    """Create visualization with UMAP action embeddings and Q-values.
    
    Args:
        trajectory: Trajectory data with action_samples, diffusion_actions
        agent: Trained agent (must support Q-value computation)
        output_path: Path to save output video
        fps: Frames per second
        dim: Output video dimension (width and height)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logging.info("Computing Q-values for action samples...")
    
    # Get data
    action_dim = agent.action_dim // agent.action_horizon
    action_samples = trajectory['action_samples']  # (T, num_samples, action_horizon, action_dim)
    actions = trajectory['actions'][:, :, :action_dim]  # (T, action_horizon, action_dim)
    diffusion_actions = trajectory.get('diffusion_actions', actions)  # (T, action_horizon, action_dim)
    vlm_outputs = trajectory['vlm_output']  # (T, hidden_dim)
    images_base = trajectory['observations']['image']  # (T, H, W, 3)
    images_wrist = trajectory['observations'].get('image_wrist', images_base)
    
    T = len(actions)
    num_samples = action_samples.shape[1]
    action_horizon = actions.shape[1]
    action_dim = actions.shape[2]
    
    # Normalize actions using agent's normalization function (similar to residual_td3.py line 677)
    logging.info("Normalizing actions...")
    # No need to normalize actions as that should be done in the replay buffer
    diffusion_actions = agent.actor.norm_actions(diffusion_actions)  # (T, action_horizon, action_dim)
    # Normalize action_samples - need to reshape for norm_actions
    # action_samples_reshaped = action_samples.reshape(T * num_samples, action_horizon, action_dim)
    # action_samples_normalized = agent.actor.norm_actions(action_samples_reshaped)
    # action_samples = action_samples_normalized.reshape(T, num_samples, action_horizon, action_dim)
    
    # Reshape actions for Q-value computation: flatten action_horizon * action_dim
    action_samples_flat = action_samples.reshape(T, num_samples, -1)  # (T, num_samples, action_horizon*action_dim)
    actions_flat = actions.reshape(T, -1)  # (T, action_horizon*action_dim)
    diffusion_actions_flat = diffusion_actions.reshape(T, -1)
    
    # Compute Q-values using agent's compute_q method
    q_action_samples = []
    q_diffusion_actions = []
    
    for t in range(T):
        vlm_t = vlm_outputs[t]  # (hidden_dim,)
        
        # Expand vlm_output to match num_samples for action_samples
        vlm_t_repeated = jnp.repeat(jnp.expand_dims(vlm_t, 0), num_samples, axis=0)  # (num_samples, hidden_dim)
        
        # Q-values for action samples using agent's compute_q method
        q_samples_t = agent.compute_q(vlm_t_repeated, action_samples_flat[t])  # (num_samples,)
        q_action_samples.append(q_samples_t)
        
        # Q-values for diffusion action
        q_diffusion_t = agent.compute_q(vlm_t, diffusion_actions_flat[t])[0]  # scalar
        q_diffusion_actions.append(q_diffusion_t)
    
    q_action_samples = np.array(q_action_samples)  # (T, num_samples)
    q_diffusion_actions = np.array(q_diffusion_actions)  # (T,)
    
    logging.info("Training UMAP on action samples...")
    # Flatten all action samples for UMAP
    all_actions_flat = action_samples_flat.reshape(-1, action_horizon * action_dim)  # (T*num_samples, action_horizon*action_dim)
    reducer = umap.UMAP(n_components=2)
    actions_embedded = reducer.fit_transform(all_actions_flat)  # (T*num_samples, 2)
    actions_embedded = actions_embedded.reshape(T, num_samples, 2)  # (T, num_samples, 2)
    
    # Embed diffusion actions
    diffusion_actions_embedded = reducer.transform(diffusion_actions_flat)  # (T, 2)
    
    logging.info("Creating video frames...")
    # For 1x3 layout: each panel is third of total width, full height
    panel_width = dim // 3
    panel_height = dim // 2  # Keep reasonable aspect ratio
    
    with imageio.get_writer(
        str(output_path),
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
    ) as writer:
        for t in range(T):
            # Resize images
            img_base = resize_image(images_base[t], panel_height, panel_width)
            img_wrist = resize_image(images_wrist[t], panel_height, panel_width)
            
            # Create UMAP plot with Q-values
            # Add markers for selected and diffusion actions with Q-values
            fig, ax = plt.subplots(figsize=(6, 5))
            q_samples_t = q_action_samples[t]
            actions_2d_t = actions_embedded[t]
            
            # Plot action samples
            mask = np.isfinite(q_samples_t) & np.isfinite(actions_2d_t).all(axis=1)
            if mask.sum() > 0:
                qmin, qmax = q_samples_t[mask].min(), q_samples_t[mask].max()
                norm = mpl.colors.Normalize(vmin=qmin, vmax=qmax)
                sc = ax.scatter(actions_2d_t[mask, 0], actions_2d_t[mask, 1],
                              c=q_samples_t[mask], cmap='viridis', norm=norm, s=25, alpha=0.6)
                fig.colorbar(sc, ax=ax, label="Q value")
            
            # Plot diffusion action
            ax.scatter(diffusion_actions_embedded[t, 0], diffusion_actions_embedded[t, 1],
                      marker='X', s=200, c='yellow', edgecolors='black', linewidths=2,
                      label=f"Diffusion (Q={q_diffusion_actions[t]:.2f})")
            
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.set_title(f"UMAP Actions (t={t})")
            ax.legend()
            ax.grid(True, linewidth=0.5, alpha=0.4)
            fig.tight_layout()
            
            fig.canvas.draw()
            umap_img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            umap_img = umap_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            umap_img = umap_img[:, :, :3]
            plt.close(fig)
            
            umap_img = resize_image(umap_img, panel_height, panel_width)
            
            # Combine into 1x3 grid: [base | wrist | umap]
            frame = np.concatenate([img_base, img_wrist, umap_img], axis=1)
            
            writer.append_data(frame)
    
    logging.info(f"UMAP visualization saved to {output_path}")


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

    # Load model first
    logging.info("Loading trained model...")
    agent = load_model(
        checkpoint_path=FLAGS.checkpoint_path,
        pi_config=pi_config,
        agent_name=FLAGS.agent_name,
        seed=FLAGS.seed,
    )
    
    # Get image_replay_buffer_kwargs from config if available
    image_replay_buffer_kwargs = {}
    if FLAGS.config is not None and hasattr(FLAGS.config, 'image_replay_buffer_kwargs'):
        image_replay_buffer_kwargs = dict(FLAGS.config.image_replay_buffer_kwargs)
        logging.info(f"Using image_replay_buffer_kwargs from config: {image_replay_buffer_kwargs}")
    
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
        image_replay_buffer_kwargs=image_replay_buffer_kwargs,
    )

    # Create visualization based on vis_type
    if FLAGS.vis_type == 1:
        logging.info("Creating state value visualization...")
        create_visualization_video(
            trajectory=trajectory,
            agent=agent,
            output_path=FLAGS.output_path,
            fps=FLAGS.fps,
            dim=FLAGS.output_dim,
        )
    elif FLAGS.vis_type == 2:
        logging.info("Creating UMAP Q-value visualization...")
        if 'action_samples' not in trajectory:
            raise ValueError("action_samples not found in trajectory. Make sure replay buffer has load_action_samples=True")
        create_visualization_video_umap(
            trajectory=trajectory,
            agent=agent,
            output_path=FLAGS.output_path,
            fps=FLAGS.fps,
            dim=FLAGS.output_dim,
        )
    else:
        raise ValueError(f"Invalid vis_type: {FLAGS.vis_type}. Must be 1 or 2.")

    logging.info("Done!")


if __name__ == "__main__":
    app.run(main)
