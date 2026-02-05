"""Visualize Q-value landscape along gradient directions in UMAP space.

Two visualization types:
1. Gradient line: Scan along normalized grad_Q direction through diffusion action
2. Gradient ascent: Follow iterative gradient ascent trajectory from diffusion action
"""

import os
import pickle
import tempfile
from pathlib import Path
from functools import partial
from typing import Dict, Optional

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import tensorflow as tf
from absl import app, flags, logging
from PIL import Image
import umap

# Reuse shared helpers from the existing visualization script.
# Importing this module also registers its flags (tfrecord_path, checkpoint_path,
# output_dir, agent_name, pi_config_name, seed, etc.) so we only define new ones.
from visualize_traj_predictions import (
    load_trajectory_from_tfrecord,
    load_model,
    resize_image,
)
from openpi.training.config import get_config

FLAGS = flags.FLAGS

# ── Flags specific to this script ──────────────────────────────────────────────
flags.DEFINE_float("lim", 0.5, "Limit for gradient line scan (-lim to +lim).")
flags.DEFINE_integer("num_line_points", 50, "Number of points along the gradient line.")
flags.DEFINE_float("eta", 0.01, "Step size (learning rate) for gradient ascent.")
flags.DEFINE_integer("num_grad_steps", 50, "Number of gradient ascent steps.")
flags.DEFINE_integer("timestep", -1, "Specific timestep to visualize (-1 for all, creates video).")
flags.DEFINE_integer("plot_type", 3, "1=gradient line only, 2=gradient ascent only, 3=both.")
flags.DEFINE_bool("clip_actions", True, "Clip actions to [-1, 1] during gradient ascent.")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_q_scalar_fn(agent):
    """Return a pure function  (action, vlm_output) -> scalar Q  suitable for jax.grad."""
    def q_scalar(action, vlm_output):
        q = agent.compute_q(vlm_output.reshape(1, -1), action.reshape(1, -1))
        return q[0]
    return q_scalar


def compute_grad_line(agent, vlm_output, diffusion_action, lim, num_points):
    """Compute actions and Q-values along the *normalized* gradient direction.

    Args:
        agent: Trained agent with ``compute_q``.
        vlm_output: VLM hidden state ``(hidden_dim,)``.
        diffusion_action: Base action ``(action_dim,)`` (normalized / flattened).
        lim: Scan from ``-lim`` to ``+lim`` along the unit gradient.
        num_points: Number of equally-spaced points on the line.

    Returns:
        line_actions  ``(num_points, action_dim)``
        line_q_values ``(num_points,)``
        grad_dir      unit gradient vector ``(action_dim,)``
    """
    q_fn = _make_q_scalar_fn(agent)
    grad_fn = jax.grad(q_fn, argnums=0)

    grad = np.array(grad_fn(jnp.array(diffusion_action), jnp.array(vlm_output)))

    grad_norm = np.linalg.norm(grad)
    grad_dir = grad / grad_norm if grad_norm > 1e-8 else grad

    scales = np.linspace(-lim, lim, num_points)
    line_actions = np.array([diffusion_action + s * grad_dir for s in scales])

    # Batched Q computation
    vlm_repeated = np.repeat(vlm_output.reshape(1, -1), num_points, axis=0)
    line_q_values = np.array(agent.compute_q(vlm_repeated, line_actions))

    return line_actions, line_q_values, grad_dir


def compute_grad_ascent_trajectory(agent, vlm_output, start_action, eta, num_steps, clip=True):
    """Iteratively follow gradient ascent starting from *start_action*.

    Update rule::

        a_{k+1} = a_k + eta * grad_a Q(s, a_k)

    Args:
        agent: Trained agent with ``compute_q``.
        vlm_output: VLM hidden state ``(hidden_dim,)``.
        start_action: Starting action ``(action_dim,)`` (normalized / flattened).
        eta: Step size.
        num_steps: Number of gradient ascent iterations.
        clip: Whether to clip actions to ``[-1, 1]`` after each step.

    Returns:
        trajectory_actions  ``(num_steps + 1, action_dim)`` (includes start)
        trajectory_q_values ``(num_steps + 1,)``
    """
    q_fn = _make_q_scalar_fn(agent)
    grad_fn = jax.grad(q_fn, argnums=0)

    current = jnp.array(start_action)
    vlm_jnp = jnp.array(vlm_output)

    traj_actions = [np.array(current)]
    traj_q = [float(agent.compute_q(vlm_output, current)[0])]

    for _ in range(num_steps):
        g = grad_fn(current, vlm_jnp)
        current = current + eta * g
        if clip:
            current = jnp.clip(current, -1.0, 1.0)
        traj_actions.append(np.array(current))
        traj_q.append(float(agent.compute_q(vlm_output, current)[0]))

    return np.array(traj_actions), np.array(traj_q)


# ── Plotting ───────────────────────────────────────────────────────────────────

def create_grad_line_plot(line_2d, line_q, diff_2d, q_diff, timestep):
    """Scatter-plot of gradient-line points in UMAP space colored by Q.

    Returns:
        Image array ``(H, W, 3)`` uint8.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    q = np.asarray(line_q)
    mask = np.isfinite(q)
    qmin, qmax = float(q[mask].min()), float(q[mask].max())
    norm = mpl.colors.Normalize(vmin=qmin, vmax=qmax)

    sc = ax.scatter(
        line_2d[mask, 0], line_2d[mask, 1],
        c=q[mask], cmap="viridis", norm=norm, s=40, alpha=0.8,
        label="Grad line",
    )
    fig.colorbar(sc, ax=ax, label="Q value")

    # Light connecting line to show direction
    ax.plot(line_2d[:, 0], line_2d[:, 1], "k-", alpha=0.3, linewidth=1)

    # Diffusion action marker
    ax.scatter(
        diff_2d[0], diff_2d[1],
        marker="X", s=250, c="red", edgecolors="black", linewidths=2,
        label=f"Diffusion (Q={q_diff:.3f})", zorder=10,
    )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"Q along Gradient Direction (t={timestep})")
    ax.legend(loc="best")
    ax.grid(True, linewidth=0.5, alpha=0.4)
    fig.tight_layout()

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    plt.close(fig)
    return img


def create_grad_ascent_plot(traj_2d, traj_q, diff_2d, q_diff, timestep):
    """Scatter-plot of gradient-ascent trajectory with arrows in UMAP space.

    Returns:
        Image array ``(H, W, 3)`` uint8.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    q = np.asarray(traj_q)
    mask = np.isfinite(q)
    qmin, qmax = float(q[mask].min()), float(q[mask].max())
    norm = mpl.colors.Normalize(vmin=qmin, vmax=qmax)
    cmap = plt.cm.viridis

    # Trajectory points
    sc = ax.scatter(
        traj_2d[mask, 0], traj_2d[mask, 1],
        c=q[mask], cmap="viridis", norm=norm, s=50, alpha=0.8, zorder=5,
    )
    fig.colorbar(sc, ax=ax, label="Q value")

    # Arrows between consecutive points
    for i in range(len(traj_2d) - 1):
        if not (np.isfinite(traj_2d[i]).all() and np.isfinite(traj_2d[i + 1]).all()):
            continue
        color = cmap(norm(q[i]))
        ax.annotate(
            "",
            xy=(traj_2d[i + 1, 0], traj_2d[i + 1, 1]),
            xytext=(traj_2d[i, 0], traj_2d[i, 1]),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5, alpha=0.7),
        )

    # Start marker (diffusion action)
    ax.scatter(
        diff_2d[0], diff_2d[1],
        marker="X", s=250, c="red", edgecolors="black", linewidths=2,
        label=f"Start / Diffusion (Q={q_diff:.3f})", zorder=10,
    )

    # End marker
    ax.scatter(
        traj_2d[-1, 0], traj_2d[-1, 1],
        marker="s", s=200, c="lime", edgecolors="black", linewidths=2,
        label=f"End (Q={q[-1]:.3f})", zorder=10,
    )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"Gradient Ascent Trajectory (t={timestep})")
    ax.legend(loc="best")
    ax.grid(True, linewidth=0.5, alpha=0.4)
    fig.tight_layout()

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    plt.close(fig)
    return img


# ── Main ───────────────────────────────────────────────────────────────────────

def main(_):
    if FLAGS.tfrecord_path is None:
        raise ValueError("--tfrecord_path must be provided")
    if FLAGS.checkpoint_path is None:
        raise ValueError("--checkpoint_path must be provided")

    devices = jax.local_devices()
    logging.info(f"Found {len(devices)} JAX devices")

    # PI config
    pi_config = get_config(FLAGS.pi_config_name)
    pi_config.fsdp_devices = 1
    pi_config.exp_name = "grad_q_visualization"
    pi_config.overwrite = True

    # Load model
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

    # Image replay buffer kwargs from config
    image_replay_buffer_kwargs = {}
    if FLAGS.config is not None and hasattr(FLAGS.config, "image_replay_buffer_kwargs"):
        image_replay_buffer_kwargs = dict(FLAGS.config.image_replay_buffer_kwargs)

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

    # ── Prepare actions ────────────────────────────────────────────────────
    action_dim = agent.action_dim // agent.action_horizon
    actions = trajectory["actions"][:, :, :action_dim]
    diffusion_actions = trajectory.get("diffusion_actions", actions)
    vlm_outputs = trajectory["vlm_output"]
    images = trajectory["observations"]["image"]
    images_wrist = trajectory["observations"].get("wrist_image", images)

    T = len(actions)

    # Normalize and flatten diffusion actions
    diffusion_actions = agent.actor.norm_actions(diffusion_actions)
    diffusion_actions_flat = diffusion_actions.reshape(T, -1)

    # Timesteps to process
    timesteps = [FLAGS.timestep] if FLAGS.timestep >= 0 else list(range(T))

    output_dir = Path(FLAGS.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    do_line = FLAGS.plot_type in (1, 3)
    do_ascent = FLAGS.plot_type in (2, 3)

    # ── Compute raw data per timestep ──────────────────────────────────────
    all_line_actions: Dict[int, np.ndarray] = {}
    all_line_q: Dict[int, np.ndarray] = {}
    all_ascent_actions: Dict[int, np.ndarray] = {}
    all_ascent_q: Dict[int, np.ndarray] = {}

    for t in timesteps:
        logging.info(f"Processing timestep {t}/{T - 1}")
        vlm_t = vlm_outputs[t]
        diff_t = diffusion_actions_flat[t]

        if do_line:
            la, lq, _ = compute_grad_line(
                agent, vlm_t, diff_t,
                lim=FLAGS.lim, num_points=FLAGS.num_line_points,
            )
            all_line_actions[t] = la
            all_line_q[t] = lq

        if do_ascent:
            ta, tq = compute_grad_ascent_trajectory(
                agent, vlm_t, diff_t,
                eta=FLAGS.eta, num_steps=FLAGS.num_grad_steps,
                clip=FLAGS.clip_actions,
            )
            all_ascent_actions[t] = ta
            all_ascent_q[t] = tq

    # ── Fit a single UMAP on all computed points ───────────────────────────
    all_points = []
    for t in timesteps:
        all_points.append(diffusion_actions_flat[t].reshape(1, -1))
    if do_line:
        for t in timesteps:
            all_points.append(all_line_actions[t])
    if do_ascent:
        for t in timesteps:
            all_points.append(all_ascent_actions[t])

    all_points = np.concatenate(all_points, axis=0)
    logging.info(f"Fitting UMAP on {len(all_points)} points...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_jobs=1)
    all_embedded = reducer.fit_transform(all_points)

    # Split embeddings back
    idx = 0
    diffusion_emb: Dict[int, np.ndarray] = {}
    for t in timesteps:
        diffusion_emb[t] = all_embedded[idx]
        idx += 1

    line_emb: Dict[int, np.ndarray] = {}
    if do_line:
        for t in timesteps:
            n = len(all_line_actions[t])
            line_emb[t] = all_embedded[idx : idx + n]
            idx += n

    ascent_emb: Dict[int, np.ndarray] = {}
    if do_ascent:
        for t in timesteps:
            n = len(all_ascent_actions[t])
            ascent_emb[t] = all_embedded[idx : idx + n]
            idx += n

    # ── Create visualizations ──────────────────────────────────────────────
    dim = FLAGS.output_dim

    def _q_diff(t):
        return float(agent.compute_q(vlm_outputs[t], diffusion_actions_flat[t])[0])

    if len(timesteps) == 1:
        # Single timestep → save PNG images
        t = timesteps[0]
        q_d = _q_diff(t)

        if do_line:
            img = create_grad_line_plot(line_emb[t], all_line_q[t], diffusion_emb[t], q_d, t)
            p = output_dir / f"grad_line_t{t}.png"
            Image.fromarray(img).save(str(p))
            logging.info(f"Saved gradient line plot to {p}")

        if do_ascent:
            img = create_grad_ascent_plot(ascent_emb[t], all_ascent_q[t], diffusion_emb[t], q_d, t)
            p = output_dir / f"grad_ascent_t{t}.png"
            Image.fromarray(img).save(str(p))
            logging.info(f"Saved gradient ascent plot to {p}")

    else:
        # Multiple timesteps → MP4 video
        if FLAGS.plot_type == 3:
            # 2×2 grid: [base | wrist] / [grad_line | grad_ascent]
            pw, ph = dim // 2, dim // 2
            out_path = output_dir / "grad_q_plots.mp4"

            with imageio.get_writer(
                str(out_path), fps=FLAGS.fps, codec="libx264",
                quality=8, pixelformat="yuv420p",
            ) as writer:
                for t in timesteps:
                    q_d = _q_diff(t)
                    img_b = resize_image(images[t], ph, pw)
                    img_w = resize_image(images_wrist[t], ph, pw)
                    li = resize_image(
                        create_grad_line_plot(line_emb[t], all_line_q[t], diffusion_emb[t], q_d, t),
                        ph, pw,
                    )
                    ai = resize_image(
                        create_grad_ascent_plot(ascent_emb[t], all_ascent_q[t], diffusion_emb[t], q_d, t),
                        ph, pw,
                    )
                    top = np.concatenate([img_b, img_w], axis=1)
                    bot = np.concatenate([li, ai], axis=1)
                    writer.append_data(np.concatenate([top, bot], axis=0))

            logging.info(f"Saved video to {out_path}")

        else:
            # Single plot type: [observation | plot]
            pw, ph = dim // 2, dim
            suffix = "grad_line" if do_line else "grad_ascent"
            out_path = output_dir / f"{suffix}.mp4"

            with imageio.get_writer(
                str(out_path), fps=FLAGS.fps, codec="libx264",
                quality=8, pixelformat="yuv420p",
            ) as writer:
                for t in timesteps:
                    q_d = _q_diff(t)
                    img_obs = resize_image(images[t], ph, pw)

                    if do_line:
                        plot = create_grad_line_plot(line_emb[t], all_line_q[t], diffusion_emb[t], q_d, t)
                    else:
                        plot = create_grad_ascent_plot(ascent_emb[t], all_ascent_q[t], diffusion_emb[t], q_d, t)
                    plot = resize_image(plot, ph, pw)

                    writer.append_data(np.concatenate([img_obs, plot], axis=1))

            logging.info(f"Saved video to {out_path}")

    # ── Save raw data ──────────────────────────────────────────────────────
    data = {
        "timesteps": np.array(timesteps),
        "diffusion_actions": np.array([diffusion_actions_flat[t] for t in timesteps]),
    }
    if do_line:
        data["line_actions"] = np.array([all_line_actions[t] for t in timesteps])
        data["line_q_values"] = np.array([all_line_q[t] for t in timesteps])
    if do_ascent:
        data["ascent_actions"] = np.array([all_ascent_actions[t] for t in timesteps])
        data["ascent_q_values"] = np.array([all_ascent_q[t] for t in timesteps])

    npz_path = output_dir / "grad_q_data.npz"
    np.savez(npz_path, **data)
    logging.info(f"Saved raw data to {npz_path}")

    logging.info("Done!")


if __name__ == "__main__":
    app.run(main)
