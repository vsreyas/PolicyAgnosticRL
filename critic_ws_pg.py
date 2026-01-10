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

chkpt = pickle.load(open('/home/sreyas/vla/PolicyAgnosticRL/results_expo_debug-expo_clean_v7/checkpoint_11999.pkl', 'rb'))
critic_params = chkpt['critic_params']
# breakpoint()
vlm_cache = pickle.load(open('/home/sreyas/vla/PolicyAgnosticRL/pkl_files/CLEAN_traj15_V3_eval_imgfix.pkl', 'rb'))
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

ep_success_id = get_ep_rows(successful_episode_ids[-1])
ep_failed_id = get_ep_rows(failed_episode_ids[-1])

print("\n\n\n\n\n")
print("Successful episode ID: ", successful_episode_ids[-1])
print("Failed episode ID: ", failed_episode_ids[-1])
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
for i in range(len(q_vals_success)):
    plt.figure(figsize=(10, 6))
    plt.plot(q_vals_success[i, :], label='ep_success_id_terminals_{}'.format(df.iloc[ep_success_id]['terminals'].sum()))
    plt.plot(q_vals_failed[i, :], label='ep_failed_id_terminals_{}'.format(df.iloc[ep_failed_id]['terminals'].sum()))
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/q_vals_{i}.png')

for i in range(len(q_vals_success)):
    plt.figure(figsize=(10, 6))
    plt.scatter(q_vals_success[i, :], mc_returns_success)
    plt.scatter(q_vals_failed[i, :], mc_returns_failed)
    plt.xlabel('Q-Values')
    plt.ylabel('MC Returns')
    plt.title('Q-Values vs MC Returns')
    plt.savefig(f'plots/q_vals_vs_mc_returns_{i}.png')

for i in range(len(q_next_vals_success)):
    plt.figure(figsize=(10, 6))
    plt.plot(q_next_vals_success[i, :], label='ep_success_id')
    plt.plot(q_next_vals_failed[i, :], label='ep_failed_id')
    # plt.plot(q_next_vals1[i, :], label='ep1')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/q_next_vals_success_{i}.png')
    plt.savefig(f'plots/q_next_vals_failed_{i}.png')