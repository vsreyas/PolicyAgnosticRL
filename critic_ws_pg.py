import jax
import jax.numpy as jnp
import pickle
from functools import partial
import numpy as np
import tensorflow as tf
import pandas as pd
import optax
from flax.training.train_state import TrainState
import matplotlib.pyplot as plt
import os
import flax.linen as nn

from jaxrl_m.utils.expo_utils import (
    StateActionValue,
    Ensemble,
    MLP,
)
from jaxrl_m.agents.continuous.expo_pi import compute_q_all

chkpt = pickle.load(open('/home/skowshik/vla/codebase/PolicyAgnosticRL/results_expo_debug-td-clean_v13-scale10_200_actnorm/seed_0/checkpoint_4500.pkl', 'rb'))
critic_params = chkpt['critic_params']
# breakpoint()
vlm_cache = pickle.load(open('/home/skowshik/vla/codebase/PolicyAgnosticRL/clean_skip_v1_server_scale10_200_actnorm.pkl', 'rb'))
# vlm_actions_replay_ep10_1_ep0, vlm_actions_replay_ep10_3_v4
# breakpoint()

rng = jax.random.PRNGKey(0)
critic_key, rng = jax.random.split(rng)
dummy_observations = jnp.ones((1, 2048 + 8))
dummy_actions = jnp.ones((1, 10, 7))

# critic_base_cls = partial(
#             MLP,
#             hidden_dims=hidden_dims,
#             activate_final=True,
#             dropout_rate=critic_dropout_rate,
#             use_layer_norm=critic_layer_norm,
#             use_pnorm=use_pnorm,
#             # activations=nn.swish,
#             activations=nn.relu,
#         )
#         critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
#         critic_def = Ensemble(critic_cls, num=num_qs)

critic_base_cls = partial(
    MLP,
    hidden_dims=(512, 512, 512, 512),
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

# mean_vlm_outputs = np.mean(vlm_cache['current_vlm_outputs'], axis=0)
# std_vlm_outputs = np.std(vlm_cache['current_vlm_outputs'], axis=0)
# vlm_cache['current_vlm_outputs'] = (vlm_cache['current_vlm_outputs'] - mean_vlm_outputs) / (std_vlm_outputs + 1e-8)
# vlm_cache['next_vlm_outputs'] = (vlm_cache['next_vlm_outputs'] - mean_vlm_outputs) / (std_vlm_outputs + 1e-8)

# Construct a df from vlm_cache for reference with breakpoints
data = vlm_cache
df = pd.DataFrame(data['episode_ids'])
df.columns = ['episode_id']
df['episode_timesteps'] = data['episode_timesteps']
df['terminals'] = data['terminals']
df['truncates'] = data['truncates']
df['rewards'] = data['rewards']
df['masks'] = data['masks']
df['mc_returns'] = data['mc_returns']
# df['dones'] = df['terminals'] | df['truncates']
# breakpoint()

def get_q(vlm_cache_idx):
    vlm_output = vlm_cache['current_vlm_outputs'][vlm_cache_idx].reshape(-1, 2056)
    # current_state = vlm_cache['current_states'][vlm_cache_idx].reshape(-1, 8)
    # vlm_output_with_state = jnp.concatenate([vlm_output, current_state], axis=1)
    vlm_output_with_state = vlm_output
    action_sequence = vlm_cache['actions'][vlm_cache_idx]
    if action_sequence.ndim == 2:
        action_sequence = action_sequence.reshape(-1, 70)
    elif action_sequence.ndim == 3:
        action_sequence = action_sequence[:, :, :7].reshape(-1, 70)
    else:
        raise ValueError(f"Action sequence has invalid shape: {action_sequence.shape}")
    
    q_vals = compute_q_all(critic.apply_fn, critic.params, vlm_output_with_state, action_sequence)
    return q_vals


### Successful and failed trajectory q progression across time ###
def get_q_next(vlm_cache_idx):
    vlm_output = vlm_cache['next_vlm_outputs'][vlm_cache_idx].reshape(-1, 2056)
    # next_state = vlm_cache['next_states'][vlm_cache_idx].reshape(-1, 8)
    # vlm_output_with_state = jnp.concatenate([vlm_output, next_state], axis=1)
    vlm_output_with_state = vlm_output
    action_idx = 0
    action_sequence = vlm_cache['next_actions'][vlm_cache_idx]
    if action_sequence.ndim == 2:
        action_sequence = action_sequence.reshape(-1, 70)
    elif action_sequence.ndim == 3:
        action_sequence = action_sequence[:, :, :7].reshape(-1, 70)
    else:
        raise ValueError(f"Action sequence has invalid shape: {action_sequence.shape}")
    
    q_vals = compute_q_all(critic.apply_fn, critic.params, vlm_output_with_state, action_sequence)
    return q_vals

def get_ep_rows(episode_id):
    return df[df['episode_id']==episode_id].index.to_list()

# breakpoint()

# q_vals = compute_q_all(critic_def.apply_fn, critic_params, vlm_cache['current_vlm_outputs'], vlm_cache['actions'])
# breakpoint()
# Pick one successful and one failed trajectory from data
successful_traj_idx = df[df['terminals'] == True].index.to_list()
failed_traj_idx = df[df['terminals'] == False].index.to_list()
successful_episode_ids = df.iloc[successful_traj_idx]['episode_id'].unique()
all_episode_ids = df['episode_id'].unique()
failed_episode_ids = list(set(all_episode_ids) - set(successful_episode_ids))

# breakpoint()

ep_idx_to_plot = 0
ep_success_id = get_ep_rows(successful_episode_ids[ep_idx_to_plot])
ep_failed_id = get_ep_rows(failed_episode_ids[ep_idx_to_plot])

print("\n\n\n\n\n")
print("Successful episode ID: ", successful_episode_ids[ep_idx_to_plot])
print("Failed episode ID: ", failed_episode_ids[ep_idx_to_plot])
print("\n\n\n\n\n")


q_vals_success = get_q(ep_success_id)
q_vals_failed = get_q(ep_failed_id)

# breakpoint()

mc_returns_success = vlm_cache['mc_returns'][ep_success_id]
mc_returns_failed = vlm_cache['mc_returns'][ep_failed_id]

# Get q_next_vals
q_next_vals_success = get_q_next(ep_success_id)
q_next_vals_failed = get_q_next(ep_failed_id)

# q_vals0[0, :]
# breakpoint()

# Plot all these to different plots in a folder
os.makedirs('plots', exist_ok=True)
# for i in range(len(q_vals_success)):
#     plt.figure(figsize=(10, 6))
#     plt.plot(q_vals_success[i, :], label='ep_success_id_terminals_{}'.format(df.iloc[ep_success_id]['terminals'].sum()))
#     plt.plot(q_vals_failed[i, :], label='ep_failed_id_terminals_{}'.format(df.iloc[ep_failed_id]['terminals'].sum()))
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f'plots/q_vals_{i}.png')

for i in range(len(q_vals_success)):
    n = q_vals_success[i, :].shape[0]              # length of each curve
    x = np.arange(n) * 5                     # 0, 5, 10, ...

    plt.figure(figsize=(10, 6))
    plt.plot(x, q_vals_success[i, :],
             label=f"ep_success_id_terminals_{df.iloc[ep_success_id]['terminals'].sum()}")
    
    n = q_vals_failed[i, :].shape[0]
    x = np.arange(n) * 5
    plt.plot(x, q_vals_failed[i, :],
             label=f"ep_failed_id_terminals_{df.iloc[ep_failed_id]['terminals'].sum()}")

    plt.grid(True)  # adds grid :contentReference[oaicite:0]{index=0}
    plt.xlabel("Timestep")                   # now in multiples of 5
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/q_vals_{i}.png")
    plt.close()

# for i in range(len(q_vals_success)):
#     plt.figure(figsize=(10, 6))
#     plt.scatter(q_vals_success[i, :], mc_returns_success)
#     plt.scatter(q_vals_failed[i, :], mc_returns_failed)
#     plt.xlabel('Q-Values')
#     plt.ylabel('MC Returns')
#     plt.title('Q-Values vs MC Returns')
#     plt.savefig(f'plots/q_vals_vs_mc_returns_{i}.png')

# for i in range(len(q_next_vals_success)):
#     plt.figure(figsize=(10, 6))
#     plt.plot(q_next_vals_success[i, :], label='ep_success_id')
#     plt.plot(q_next_vals_failed[i, :], label='ep_failed_id')
#     # plt.plot(q_next_vals1[i, :], label='ep1')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f'plots/q_next_vals_success_{i}.png')
#     plt.savefig(f'plots/q_next_vals_failed_{i}.png')

### Plot all successful trajectories in one plot and all failed trajectories in another plot ###
def _extract_q1(q_vals: np.ndarray) -> np.ndarray:
    """
    Return Q1 with shape (N, T).
    Accepts:
      - (N, T)              already Q1
      - (N, 2, T)           -> take [:, 0, :]
      - (N, T, 2)           -> take [:, :, 0]
      - (N, T, K) (K>=1)    -> take [:, :, 0]
    """
    q = np.asarray(q_vals)
    if q.ndim == 2:
        return q
    if q.ndim == 3:
        if q.shape[1] == 2:        # (N, 2, T)
            return q[:, 0, :]
        if q.shape[-1] == 2:       # (N, T, 2)
            return q[:, :, 0]
        return q[:, :, 0]          # (N, T, K)
    raise ValueError(f"Unexpected q_vals shape: {q.shape}")

def plot_all_trajectories_q1(q_vals: np.ndarray, out_path: str, title: str, x_scale: int = 5):
    q1 = _extract_q1(q_vals)             # (N, T)
    N, T = q1.shape
    x = np.arange(T) * x_scale           # numeric scaling by 5

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each trajectory (light)
    for i in range(N):
        ax.plot(x, q1[i], alpha=0.2, linewidth=1.0)

    # Overlay mean ± std for readability
    mean = q1.mean(axis=0)
    std = q1.std(axis=0)
    ax.plot(x, mean, linewidth=2.0, label=f"mean (n={N})")
    ax.fill_between(x, mean - std, mean + std, alpha=0.2)

    ax.set_title(title)
    ax.set_xlabel(f"Time step (×{x_scale})")
    ax.set_ylabel("Q1")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

# --- Save two figures ---
plot_all_trajectories_q1(
    q_vals_success,
    out_path="plots/q1_all_success.png",
    title="Q1 over time — all successful trajectories",
    x_scale=5,
)

plot_all_trajectories_q1(
    q_vals_failed,
    out_path="plots/q1_all_failed.png",
    title="Q1 over time — all failed trajectories",
    x_scale=5,
)


### Epsilon purturbation experiments ###
EPSILON: float = 0.05
def _reshape_action_sequence(action_sequence: np.ndarray) -> np.ndarray:
    """Return action_sequence as float32 array of shape (T, 70)."""
    a = np.asarray(action_sequence)

    if a.ndim == 2:
        # Either (T, 70) already, or (T, 10*7) equivalent
        return a.astype(np.float32).reshape(-1, 70)

    if a.ndim == 3:
        # (T, 10, 7/...) -> keep first 7 dims, flatten to 70
        a = a[:, :, :7]
        return a.astype(np.float32).reshape(-1, 70)

    raise ValueError(f"Action sequence has invalid shape: {a.shape}")


def get_q_noisy(vlm_cache_idx, epsilon: float = EPSILON, seed: int = 0) -> np.ndarray:
    """
    Compute q values after adding N(0, epsilon) noise to each action element.
    Returns same shape as get_q(): typically (num_qs, T).
    """
    rng = np.random.default_rng(seed)

    vlm_output = vlm_cache["current_vlm_outputs"][vlm_cache_idx].reshape(-1, 2056)
    vlm_output_with_state = vlm_output

    a = vlm_cache["actions"][vlm_cache_idx]
    a_reshaped = _reshape_action_sequence(a)  # (T, 70)

    noise = rng.normal(loc=0.0, scale=epsilon, size=a_reshaped.shape).astype(np.float32)
    a_noisy = a_reshaped + noise

    q_vals = compute_q_all(
        critic.apply_fn,
        critic.params,
        jnp.asarray(vlm_output_with_state),
        jnp.asarray(a_noisy),
    )
    return np.asarray(q_vals)


def _extract_q1_from_qvals(q_vals: np.ndarray) -> np.ndarray:
    """
    Given q_vals shaped (num_qs, T) or (T, num_qs), return Q1 as (T,).
    """
    q = np.asarray(q_vals)
    if q.ndim != 2:
        raise ValueError(f"Expected 2D q_vals, got {q.shape}")

    # Your script uses q_vals[i, :] => (num_qs, T)
    if q.shape[0] <= 32 and q.shape[1] > q.shape[0]:
        return q[0, :]

    # Fallback: assume (T, num_qs)
    return q[:, 0]


def plot_q1_trajectories_variable_length(
    q1_trajs: list[np.ndarray],
    out_path: str,
    title: str,
    x_scale: int = 5,
):
    """
    Plot many variable-length Q1 trajectories on one figure + mean±std (nan-padded).
    """
    if len(q1_trajs) == 0:
        print(f"[WARN] No trajectories to plot for {out_path}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # individual curves
    for q1 in q1_trajs:
        q1 = np.asarray(q1).reshape(-1)
        x = np.arange(len(q1)) * x_scale
        ax.plot(x, q1, alpha=0.15, linewidth=1.0)

    # nan-padded mean±std
    max_T = max(len(np.asarray(q1).reshape(-1)) for q1 in q1_trajs)
    pad = np.full((len(q1_trajs), max_T), np.nan, dtype=np.float32)
    for i, q1 in enumerate(q1_trajs):
        q1 = np.asarray(q1).reshape(-1).astype(np.float32)
        pad[i, : len(q1)] = q1

    x = np.arange(max_T) * x_scale
    mean = np.nanmean(pad, axis=0)
    std = np.nanstd(pad, axis=0)

    ax.plot(x, mean, linewidth=2.0, label=f"mean (n={len(q1_trajs)})")
    ax.fill_between(x, mean - std, mean + std, alpha=0.2)

    ax.set_title(title)
    ax.set_xlabel(f"Time step (×{x_scale})")
    ax.set_ylabel("Q1 (noisy actions)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

# --- New: noisy-action Q plots (Q1) for ALL successful vs failed episodes ---
q1_noisy_success_trajs = []
for ep_id in successful_episode_ids:
    rows = get_ep_rows(ep_id)
    q_noisy = get_q_noisy(rows, epsilon=EPSILON, seed=int(ep_id))
    q1_noisy_success_trajs.append(_extract_q1_from_qvals(q_noisy))

q1_noisy_failed_trajs = []
for ep_id in failed_episode_ids:
    rows = get_ep_rows(ep_id)
    q_noisy = get_q_noisy(rows, epsilon=EPSILON, seed=int(ep_id))
    q1_noisy_failed_trajs.append(_extract_q1_from_qvals(q_noisy))

plot_q1_trajectories_variable_length(
    q1_noisy_success_trajs,
    out_path="plots/q1_noisy_all_success.png",
    title=f"Q1 over time with action noise N(0, {EPSILON}) — successful episodes",
    x_scale=5,
)

plot_q1_trajectories_variable_length(
    q1_noisy_failed_trajs,
    out_path="plots/q1_noisy_all_failed.png",
    title=f"Q1 over time with action noise N(0, {EPSILON}) — failed episodes",
    x_scale=5,
)