"""Visualize state-value predictions along trajectories from tfrecord files."""

import os
import pickle
import tempfile
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
flags.DEFINE_string("critic_params_path", None, "Path to critic params (.pkl file) to override checkpoint.")
flags.DEFINE_string("edit_actor_params_path", None, "Path to edit actor params (.pkl file) to override checkpoint.")
flags.DEFINE_string("output_dir", "./trajectory_visualization", "Output directory for visualization files.")
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
flags.DEFINE_bool("plot_grad_q_line", False, "Plot Q-gradient line in UMAP visualization.")
flags.DEFINE_integer("grad_line_points", 20, "Number of points along gradient line.")
flags.DEFINE_float("grad_line_scale_low", -0.5, "Low scale for gradient line.")
flags.DEFINE_float("grad_line_scale_high", 0.5, "High scale for gradient line.")
flags.DEFINE_bool("generate_action_samples", False, "Generate new action samples from agent.")
flags.DEFINE_integer("num_action_samples", 100, "Number of action samples to generate per timestep.")
flags.DEFINE_bool("generate_base_actions", True, "Use sample_base_actions (True) or sample_actions (False) when generating action samples.")
flags.DEFINE_bool("visualize_residuals", False, "Visualize edit actions (residuals) from the edit actor.")
flags.DEFINE_integer("num_qs", 10, "Number of Q networks in the ensemble.")
flags.DEFINE_integer("num_min_qs", 2, "Number of Q networks to use for min computation.")


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
        'prompt': [],
    }

    if use_wrist_view:
        trajectory_data['observations']['wrist_image'] = []

    # Iterate through the dataset to collect all steps
    dataset_iter = dataset.iterator(batch_size=1)

    for i, batch in enumerate(dataset_iter):
        # Collect data
        trajectory_data['observations']['image'].append(batch['observations']['image'][0])
        trajectory_data['observations']['proprio'].append(batch['observations']['proprio'][0])

        if use_wrist_view and 'wrist_image' in batch['observations']:
            trajectory_data['observations']['wrist_image'].append(batch['observations']['wrist_image'][0])

        trajectory_data['actions'].append(batch['actions'][0])
        trajectory_data['rewards'].append(batch['rewards'][0])
        trajectory_data['terminals'].append(batch['terminals'][0])

        if 'vlm_output' in batch:
            trajectory_data['vlm_output'].append(batch['vlm_output'][0])
        
        if 'action_samples' in batch:
            trajectory_data['action_samples'].append(batch['action_samples'][0])
        
        if 'diffusion_actions' in batch:
            trajectory_data['diffusion_actions'].append(batch['diffusion_actions'][0])
        
        # if 'prompt_bytes' in batch:
        #     # Decode the prompt from bytes to string
        prompt_bytes = batch['prompt_bytes'][0]
        prompt_str = prompt_bytes.tobytes().split(b"\x00", 1)[0].decode("utf-8")
        trajectory_data['prompt'].append(prompt_str)

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

    if use_wrist_view and len(trajectory_data['observations']['wrist_image']) > 0:
        trajectory['observations']['wrist_image'] = np.array(trajectory_data['observations']['wrist_image'])

    if len(trajectory_data['vlm_output']) > 0:
        trajectory['vlm_output'] = np.array(trajectory_data['vlm_output'])
    
    if len(trajectory_data['action_samples']) > 0:
        trajectory['action_samples'] = np.array(trajectory_data['action_samples'])
    
    if len(trajectory_data['diffusion_actions']) > 0:
        trajectory['diffusion_actions'] = np.array(trajectory_data['diffusion_actions'])
    
    if len(trajectory_data['prompt']) > 0:
        # Store prompts as a list of strings (they should all be the same across timesteps)
        trajectory['prompt'] = trajectory_data['prompt'][0] if trajectory_data['prompt'] else ""

    logging.info(f"Loaded trajectory with {len(trajectory['actions'])} timesteps")

    return trajectory


def load_model(
    checkpoint_path: str, 
    pi_config: any, 
    agent_name: str = "pi_residual_ppo", 
    seed: int = 0,
    num_qs: int = 10,
    num_min_qs: int = 2,
    critic_params_path: Optional[str] = None,
    edit_actor_params_path: Optional[str] = None,
):
    """Load trained residual agent from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint .pkl file
        pi_config: PI configuration object
        agent_name: Name of agent to load ('pi_residual_td3' or 'pi_residual_ppo')
        seed: Random seed
        num_qs: Number of Q networks in the ensemble
        num_min_qs: Number of Q networks to use for min computation
        critic_params_path: Optional path to critic params file to override checkpoint
        edit_actor_params_path: Optional path to edit actor params file to override checkpoint

    Returns:
        Loaded agent (PiResidualTD3Cache or PiResidualPPOCache)
    """
    # Load base checkpoint and potentially override params
    final_checkpoint_path = checkpoint_path
    
    if critic_params_path is not None or edit_actor_params_path is not None:
        logging.info("Creating merged checkpoint with overrides...")
        
        # Load base checkpoint
        with open(checkpoint_path, 'rb') as f:
            checkpoint_dict = pickle.load(f)
        
        # Override critic params if provided
        if critic_params_path is not None:
            logging.info(f"Overriding critic params from {critic_params_path}")
            with open(critic_params_path, 'rb') as f:
                critic_override = pickle.load(f)
                if 'critic_params' in critic_override:
                    checkpoint_dict['critic_params'] = critic_override['critic_params']
                    # Also update target_critic_params
                    if 'target_critic_params' in critic_override:
                        checkpoint_dict['target_critic_params'] = critic_override['target_critic_params']
                    else:
                        # Use same params for target if not provided separately
                        checkpoint_dict['target_critic_params'] = critic_override['critic_params']
                else:
                    raise ValueError(f"critic_params not found in {critic_params_path}")
        
        # Override edit actor params if provided
        if edit_actor_params_path is not None:
            logging.info(f"Overriding edit actor params from {edit_actor_params_path}")
            with open(edit_actor_params_path, 'rb') as f:
                edit_actor_override = pickle.load(f)
                if 'edit_actor_params' in edit_actor_override:
                    checkpoint_dict['edit_actor_params'] = edit_actor_override['edit_actor_params']
                else:
                    raise ValueError(f"edit_actor_params not found in {edit_actor_params_path}")
        
        # Save merged checkpoint to temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False)
        pickle.dump(checkpoint_dict, temp_file)
        temp_file.close()
        final_checkpoint_path = temp_file.name
        logging.info(f"Created temporary merged checkpoint at {final_checkpoint_path}")
    
    logging.info(f"Loading {agent_name} checkpoint from {final_checkpoint_path}")

    rng = jax.random.PRNGKey(seed)

    agent = create_agent(
        agent_name=agent_name,
        config=pi_config,
        seed=seed,
        batch_size=1,  # batch_size for initialization
        rng=rng,
        params_path=final_checkpoint_path,
        N=4,
        n_edit_samples=4,
        num_qs=num_qs,
        num_min_qs=num_min_qs,
    )
    
    # Clean up temporary file if created
    if final_checkpoint_path != checkpoint_path:
        os.remove(final_checkpoint_path)
        logging.info("Cleaned up temporary checkpoint file")

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


def create_value_plot(state_values: np.ndarray, current_step: int, title: str = 'Predicted State Values', ylabel: str = 'State Value') -> np.ndarray:
    """Create a plot of state values up to current step.

    Args:
        state_values: Array of state values
        current_step: Current timestep to plot up to
        title: Plot title
        ylabel: Y-axis label

    Returns:
        Image array (H, W, 3) in uint8 format
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    steps = np.arange(current_step + 1)
    values = state_values[:current_step + 1]

    ax.plot(steps, values, 'b-', linewidth=2)
    ax.scatter([current_step], [values[-1]], color='red', s=100, zorder=5)

    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
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
    output_dir: str,
    fps: int = 10,
    dim: int = 1024,
) -> None:
    """Create side-by-side visualization video.

    Args:
        trajectory: Trajectory data dictionary
        agent: Trained agent with get_state_values method
        output_dir: Directory to save output files
        fps: Frames per second
        dim: Output video dimension (width and height)
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute state values
    vlm_outputs = trajectory['vlm_output']
    state_values = compute_state_values(agent, vlm_outputs)
    
    output_path = output_dir / "visualize.mp4"

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
    output_dir: str,
    fps: int = 10,
    dim: int = 1024,
    plot_grad_q_line: bool = False,
    grad_line_points: int = 20,
    grad_line_scale_low: float = -0.5,
    grad_line_scale_high: float = 0.5,
    generate_action_samples: bool = False,
    num_action_samples: int = 100,
    generate_base_actions: bool = True,
    visualize_residuals: bool = False,
) -> None:
    """Create visualization with UMAP action embeddings and Q-values.
    
    Args:
        trajectory: Trajectory data with action_samples, diffusion_actions
        agent: Trained agent (must support Q-value computation)
        output_dir: Directory to save output files
        fps: Frames per second
        dim: Output video dimension (width and height)
        plot_grad_q_line: Whether to plot Q-gradient line
        grad_line_points: Number of points along gradient line
        grad_line_scale_low: Low scale for gradient line
        grad_line_scale_high: High scale for gradient line
        generate_action_samples: Whether to generate new action samples from agent
        num_action_samples: Number of action samples to generate per timestep
        generate_base_actions: Use sample_base_actions (True) or sample_actions (False)
        visualize_residuals: Whether to visualize edit actions
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "visualize.mp4"
    
    logging.info("Computing Q-values for action samples...")
    
    # Get data
    action_dim = agent.action_dim // agent.action_horizon
    action_samples = trajectory['action_samples']  # (T, num_samples, action_horizon, action_dim) - always the original
    actions = trajectory['actions'][:, :, :action_dim]  # (T, action_horizon, action_dim)
    diffusion_actions = trajectory.get('diffusion_actions', actions)  # (T, action_horizon, action_dim)
    vlm_outputs = trajectory['vlm_output']  # (T, hidden_dim)
    images_base = trajectory['observations']['image']  # (T, H, W, 3)
    images_wrist = trajectory['observations'].get('wrist_image', images_base)
    
    T = len(actions)
    action_horizon = actions.shape[1]
    action_dim = actions.shape[2]
    num_samples = action_samples.shape[1]
    
    # Generate new action samples if requested
    action_samples_generated = None
    num_samples_generated = 0
    if generate_action_samples:
        logging.info(f"Generating {num_action_samples} action samples per timestep from agent...")
        generated_action_samples_list = []
        
        # Initialize RNG for sampling
        sample_rng = jax.random.PRNGKey(42)
        
        for t in range(T):
            # Create input dict for sample_base_actions
            # Need to unnormalize proprio as data from replay buffer is normalized
            proprio_normalized = trajectory['observations']['proprio'][t]  # (state_dim,)
            proprio_normalized = proprio_normalized[:agent.state_dim]
            # Add dummy actions for unnormalization (required by the transform)
            dummy_actions = np.zeros((action_horizon, action_dim), dtype=np.float32)
            proprio_dict = {
                "state": proprio_normalized[np.newaxis, :],  # (1, state_dim)
                "actions": dummy_actions  # (1, action_horizon, action_dim)
            }
            proprio_unnorm = agent.actor.unnormalize(proprio_dict)["state"][0]  # (state_dim,)
            
            obs_dict = {
                'proprio': proprio_unnorm,
                'image': trajectory['observations']['image'][t],
                'wrist_image': trajectory['observations']['wrist_image'][t],
                'prompt': trajectory['prompt'],
            }
            
            # Sample actions from agent with num_action_samples
            sample_rng, use_rng = jax.random.split(sample_rng)
            if generate_base_actions:
                result_dict = agent.sample_base_actions(obs_dict, seed=use_rng, num_diffusion_samples=num_action_samples)
                # Extract action_samples from the returned dictionary
                base_actions_samples = result_dict['action_samples']  # (num_action_samples, action_horizon, action_dim)
            else:
                result_dict = agent.sample_actions(obs_dict, seed=use_rng, use_deterministic_actions=True, num_action_samples=1)
                # Extract actions from the returned dictionary and add batch dimension
                base_actions_samples = result_dict['actions'][np.newaxis, ...]  # (1, action_horizon, action_dim)
            
            generated_action_samples_list.append(base_actions_samples)
        
        action_samples_generated = np.array(generated_action_samples_list)  # (T, num_action_samples, action_horizon, action_dim)
        num_samples_generated = num_action_samples
    
    # Normalize actions using agent's normalization function (similar to residual_td3.py line 677)
    logging.info("Normalizing actions...")
    # Keep original diffusion_actions for edit_action computation (get_edit_action normalizes internally)
    diffusion_actions_original = diffusion_actions.copy()
    # No need to normalize actions as that should be done in the replay buffer
    diffusion_actions = agent.actor.norm_actions(diffusion_actions)  # (T, action_horizon, action_dim)
    
    # Compute edit actions if visualize_residuals is enabled
    edit_actions = None
    if visualize_residuals:
        logging.info("Computing edit actions from edit actor...")
        edit_actions_list = []
        for t in range(T):
            # get_edit_action expects (action_dim,) or (batch, action_dim) and vlm_output
            # diffusion_actions_original is (T, action_horizon, action_dim) - need to flatten
            diffusion_action_t = diffusion_actions_original[t].reshape(-1)  # (action_horizon * action_dim,)
            vlm_t = vlm_outputs[t]  # (hidden_dim,)
            
            # get_edit_action normalizes internally, returns normalized actions (unnorm commented out)
            edit_action_t = agent.get_edit_action(diffusion_action_t, vlm_t, clip=True)
            edit_actions_list.append(np.array(edit_action_t))
        edit_actions = np.array(edit_actions_list)  # (T, action_horizon * action_dim)
    
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
    
    # Compute Q-values for edit actions if visualize_residuals is enabled
    q_edit_actions = None
    if visualize_residuals and edit_actions is not None:
        logging.info("Computing Q-values for edit actions...")
        q_edit_actions = []
        for t in range(T):
            vlm_t = vlm_outputs[t]
            q_edit_t = agent.compute_q(vlm_t, edit_actions[t])[0]  # scalar
            q_edit_actions.append(q_edit_t)
        q_edit_actions = np.array(q_edit_actions)  # (T,)
    
    # Compute Q-values for generated action samples if we generated new ones
    q_action_samples_generated = None
    action_samples_generated_flat = None
    if generate_action_samples:
        logging.info("Computing Q-values for generated action samples...")
        action_samples_generated_flat = action_samples_generated.reshape(T, num_samples_generated, -1)
        q_action_samples_generated = []
        for t in range(T):
            vlm_t = vlm_outputs[t]
            vlm_t_repeated = jnp.repeat(jnp.expand_dims(vlm_t, 0), num_samples_generated, axis=0)
            q_samples_t = agent.compute_q(vlm_t_repeated, action_samples_generated_flat[t])
            q_action_samples_generated.append(q_samples_t)
        q_action_samples_generated = np.array(q_action_samples_generated)  # (T, num_samples_generated)
    
    # Compute gradient line if requested
    grad_line_actions = None
    grad_line_q_values = None
    grad_line_embedded = None
    
    if plot_grad_q_line:
        logging.info("Computing Q-value gradients for diffusion actions...")
        
        # Define a function to compute Q-value for a single action
        def compute_q_single(action, vlm_output):
            """Compute Q-value for a single action given VLM output."""
            q_vals = agent.compute_q(vlm_output, action)
            return q_vals[0]  # Return scalar Q-value
        
        # Create gradient function
        grad_q_fn = jax.grad(compute_q_single, argnums=0)
        
        # Compute gradients and create line actions
        all_grad_line_actions = []  # (T, grad_line_points, action_horizon*action_dim)
        all_grad_line_q_values = []  # (T, grad_line_points)
        
        scales = np.linspace(grad_line_scale_low, grad_line_scale_high, grad_line_points)
        
        for t in range(T):
            vlm_t = vlm_outputs[t]
            diffusion_action_t = diffusion_actions_flat[t]
            
            # Compute gradient
            grad_t = grad_q_fn(diffusion_action_t, vlm_t)  # (action_horizon*action_dim,)
            grad_t = np.array(grad_t)
            
            # Normalize gradient to unit length
            grad_norm = np.linalg.norm(grad_t)
            if grad_norm > 1e-8:
                grad_t = grad_t / grad_norm
            
            # Create line of actions along gradient
            line_actions_t = []
            line_q_values_t = []
            
            for scale in scales:
                action_on_line = diffusion_action_t + scale * grad_t
                line_actions_t.append(action_on_line)
                
                # Compute Q-value for this action
                q_val = agent.compute_q(vlm_t, action_on_line)[0]
                line_q_values_t.append(q_val)
            
            all_grad_line_actions.append(np.array(line_actions_t))
            all_grad_line_q_values.append(np.array(line_q_values_t))
        
        grad_line_actions = np.array(all_grad_line_actions)  # (T, grad_line_points, action_horizon*action_dim)
        grad_line_q_values = np.array(all_grad_line_q_values)  # (T, grad_line_points)
    
    logging.info("Training UMAP on action samples (original)...")
    # Flatten all action samples for UMAP - train on original action_samples
    all_actions_flat = action_samples_flat.reshape(-1, action_horizon * action_dim)  # (T*num_samples, action_horizon*action_dim)
    reducer = umap.UMAP(n_components=2, random_state=42, n_jobs=1)
    actions_embedded = reducer.fit_transform(all_actions_flat)  # (T*num_samples, 2)
    actions_embedded = actions_embedded.reshape(T, num_samples, 2)  # (T, num_samples, 2)
    
    # Embed generated action samples if we generated new ones
    action_samples_generated_embedded = None
    if generate_action_samples:
        logging.info("Embedding generated action samples in UMAP space...")
        action_samples_generated_flat_all = action_samples_generated_flat.reshape(-1, action_horizon * action_dim)
        action_samples_generated_embedded = reducer.transform(action_samples_generated_flat_all)
        action_samples_generated_embedded = action_samples_generated_embedded.reshape(T, num_samples_generated, 2)
    
    # Embed diffusion actions
    diffusion_actions_embedded = reducer.transform(diffusion_actions_flat)  # (T, 2)
    
    # Embed edit actions if visualize_residuals is enabled
    edit_actions_embedded = None
    if visualize_residuals and edit_actions is not None:
        logging.info("Embedding edit actions in UMAP space...")
        edit_actions_embedded = reducer.transform(edit_actions)  # (T, 2)
    
    # Embed gradient line actions if computed
    if plot_grad_q_line:
        logging.info("Embedding gradient line actions in UMAP space...")
        grad_line_actions_flat = grad_line_actions.reshape(-1, action_horizon * action_dim)
        grad_line_embedded_flat = reducer.transform(grad_line_actions_flat)  # (T*grad_line_points, 2)
        grad_line_embedded = grad_line_embedded_flat.reshape(T, grad_line_points, 2)  # (T, grad_line_points, 2)
    
    logging.info("Creating video frames...")
    # For 2x2 layout: each panel is half of total width and height
    panel_width = dim // 2
    panel_height = dim // 2
    
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
            
            # Create Q-value trajectory plot
            q_plot = create_value_plot(
                q_diffusion_actions, 
                t, 
                title='Diffusion Action Q-Values',
                ylabel='Q Value'
            )
            q_plot = resize_image(q_plot, panel_height, panel_width)
            
            # Create UMAP plot with Q-values
            # Add markers for action samples and diffusion actions with Q-values
            fig, ax = plt.subplots(figsize=(6, 5))
            q_samples_t = q_action_samples[t]
            actions_2d_t = actions_embedded[t]
            
            # Determine Q-value range for consistent coloring (only for original action samples)
            q_finite = q_samples_t[np.isfinite(q_samples_t)]
            if len(q_finite) > 0:
                qmin, qmax = q_finite.min(), q_finite.max()
            else:
                qmin, qmax = 0, 1
            norm = mpl.colors.Normalize(vmin=qmin, vmax=qmax)
            
            # Plot original action samples (always present)
            mask = np.isfinite(q_samples_t) & np.isfinite(actions_2d_t).all(axis=1)
            if mask.sum() > 0:
                sc = ax.scatter(actions_2d_t[mask, 0], actions_2d_t[mask, 1],
                              c=q_samples_t[mask], cmap='viridis', norm=norm, s=25, alpha=0.6,
                              label="Action samples")
                fig.colorbar(sc, ax=ax, label="Q value")
            
            # Plot generated action samples if we generated new ones
            if generate_action_samples and action_samples_generated_embedded is not None:
                gen_2d_t = action_samples_generated_embedded[t]
                gen_mask = np.isfinite(gen_2d_t).all(axis=1)
                if gen_mask.sum() > 0:
                    ax.scatter(gen_2d_t[gen_mask, 0], gen_2d_t[gen_mask, 1],
                              marker='x', s=100, c='red', linewidths=2,
                              label="Generated samples")
            
            # Plot diffusion action
            ax.scatter(diffusion_actions_embedded[t, 0], diffusion_actions_embedded[t, 1],
                      marker='X', s=200, c='yellow', edgecolors='black', linewidths=2,
                      label=f"Diffusion (Q={q_diffusion_actions[t]:.2f})")
            
            # Plot edit action if visualize_residuals is enabled
            if visualize_residuals and edit_actions_embedded is not None and q_edit_actions is not None:
                ax.scatter(edit_actions_embedded[t, 0], edit_actions_embedded[t, 1],
                          marker='D', s=200, c='cyan', edgecolors='black', linewidths=2,
                          label=f"Edit (Q={q_edit_actions[t]:.2f})")
            
            # Plot gradient line if requested
            if plot_grad_q_line and grad_line_embedded is not None:
                grad_line_2d_t = grad_line_embedded[t]  # (grad_line_points, 2)
                grad_line_q_t = grad_line_q_values[t]  # (grad_line_points,)
                
                # Plot as red X markers
                grad_mask = np.isfinite(grad_line_q_t) & np.isfinite(grad_line_2d_t).all(axis=1)
                if grad_mask.sum() > 0:
                    ax.scatter(grad_line_2d_t[grad_mask, 0], grad_line_2d_t[grad_mask, 1],
                             marker='x', s=100, c='red', linewidths=2,
                             label='Grad Q line')
            
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
            
            # Combine into 2x2 grid: 
            # [base | wrist]
            # [q_plot | umap]
            top_row = np.concatenate([img_base, img_wrist], axis=1)
            bottom_row = np.concatenate([q_plot, umap_img], axis=1)
            frame = np.concatenate([top_row, bottom_row], axis=0)
            
            writer.append_data(frame)
    
    # Save raw actions and Q-values to numpy file
    data_to_save = {
        'action_samples': action_samples_flat,  # (T, num_samples, action_horizon*action_dim) - original samples
        'q_action_samples': q_action_samples,  # (T, num_samples)
        'actions': actions_flat,  # (T, action_horizon*action_dim)
        'diffusion_actions': diffusion_actions_flat,  # (T, action_horizon*action_dim)
        'q_diffusion_actions': q_diffusion_actions,  # (T,)
    }
    
    # Add generated samples if we generated new ones
    if generate_action_samples and action_samples_generated_flat is not None:
        data_to_save['action_samples_generated'] = action_samples_generated_flat  # (T, num_samples_generated, action_horizon*action_dim)
        data_to_save['q_action_samples_generated'] = q_action_samples_generated  # (T, num_samples_generated)
    
    # Add edit actions if visualize_residuals is enabled
    if visualize_residuals and edit_actions is not None:
        data_to_save['edit_actions'] = edit_actions  # (T, action_horizon*action_dim)
        data_to_save['q_edit_actions'] = q_edit_actions  # (T,)
    
    # Add gradient line data if computed
    if plot_grad_q_line and grad_line_actions is not None:
        data_to_save['grad_line_actions'] = grad_line_actions  # (T, grad_line_points, action_horizon*action_dim)
        data_to_save['grad_line_q_values'] = grad_line_q_values  # (T, grad_line_points)
    
    numpy_path = output_dir / "actions_and_q_values.npz"
    np.savez(numpy_path, **data_to_save)
    logging.info(f"Saved actions and Q-values to {numpy_path}")
    
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

    # Load model with optional parameter overrides
    logging.info("Loading trained model...")
    agent = load_model(
        checkpoint_path=FLAGS.checkpoint_path,
        pi_config=pi_config,
        agent_name=FLAGS.agent_name,
        seed=FLAGS.seed,
        num_qs=FLAGS.num_qs,
        num_min_qs=FLAGS.num_min_qs,
        critic_params_path=FLAGS.critic_params_path,
        edit_actor_params_path=FLAGS.edit_actor_params_path,
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
            output_dir=FLAGS.output_dir,
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
            output_dir=FLAGS.output_dir,
            fps=FLAGS.fps,
            dim=FLAGS.output_dim,
            plot_grad_q_line=FLAGS.plot_grad_q_line,
            grad_line_points=FLAGS.grad_line_points,
            grad_line_scale_low=FLAGS.grad_line_scale_low,
            grad_line_scale_high=FLAGS.grad_line_scale_high,
            generate_action_samples=FLAGS.generate_action_samples,
            num_action_samples=FLAGS.num_action_samples,
            generate_base_actions=FLAGS.generate_base_actions,
            visualize_residuals=FLAGS.visualize_residuals,
        )
    else:
        raise ValueError(f"Invalid vis_type: {FLAGS.vis_type}. Must be 1 or 2.")

    logging.info("Done!")


if __name__ == "__main__":
    app.run(main)
