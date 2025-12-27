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

from jaxrl_m.utils.expo_utils import (
    StateActionValue,
    Ensemble,
    MLP,
)
from jaxrl_m.agents.continuous.expo_pi import compute_q_all

chkpt = pickle.load(open('results_expo_debug-v3/seed_0/checkpoint_200000.pkl', 'rb'))
critic_params = chkpt['critic_params']
# breakpoint()
vlm_cache = pickle.load(open('/home/skowshik/vla/codebase/PolicyAgnosticRL/outputs/vlm_actions_replay_ep10_1_ep0.pkl', 'rb'))
# breakpoint()

rng = jax.random.PRNGKey(0)
critic_key, rng = jax.random.split(rng)
dummy_observations = jnp.ones((1, 2048 + 8))
dummy_actions = jnp.ones((1, 10, 7))

critic_base_cls = partial(
    MLP,
    hidden_dims=(256, 256, 256, 256),
    activate_final=False,
    dropout_rate=None,
    use_layer_norm=False,
    use_pnorm=False,
)
critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
critic_def = Ensemble(critic_cls, num=2)
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
df['dones'] = df['terminals'] | df['truncates']

def get_q(vlm_cache_idx):
    vlm_output = vlm_cache['current_vlm_outputs'][vlm_cache_idx].reshape(-1, 2048)
    current_state = vlm_cache['current_states'][vlm_cache_idx].reshape(-1, 8)
    vlm_output_with_state = jnp.concatenate([vlm_output, current_state], axis=1)
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
    vlm_output = vlm_cache['next_vlm_outputs'][vlm_cache_idx].reshape(-1, 2048)
    next_state = vlm_cache['next_states'][vlm_cache_idx].reshape(-1, 8)
    vlm_output_with_state = jnp.concatenate([vlm_output, next_state], axis=1)
    action_idx = 0
    action_sequence = vlm_cache['next_actions'][vlm_cache_idx, action_idx]
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

# q_vals = compute_q_all(critic_def.apply_fn, critic_params, vlm_cache['current_vlm_outputs'], vlm_cache['actions'])
# breakpoint()
ep0 = get_ep_rows(0)
# ep1 = get_ep_rows(1)
# ep2 = get_ep_rows(2)
# ep3 = get_ep_rows(3)
# ep4 = get_ep_rows(4)
# ep5 = get_ep_rows(5)
# ep6 = get_ep_rows(6)
# ep7 = get_ep_rows(7)
# ep8 = get_ep_rows(8)
# ep9 = get_ep_rows(9)
q_vals0 = get_q(ep0)
# q_vals1 = get_q(ep1)
# q_vals2 = get_q(ep2)
# q_vals3 = get_q(ep3)
# q_vals4 = get_q(ep4)
# q_vals5 = get_q(ep5)
# q_vals6 = get_q(ep6)
# q_vals7 = get_q(ep7)
# q_vals8 = get_q(ep8)

# Get q_next_vals
q_next_vals0 = get_q_next(ep0)
# q_next_vals1 = get_q_next(ep1)
# q_next_vals2 = get_q_next(ep2)
# q_next_vals3 = get_q_next(ep3)
# q_next_vals4 = get_q_next(ep4)
# q_next_vals5 = get_q_next(ep5)
# q_next_vals6 = get_q_next(ep6)
# q_next_vals7 = get_q_next(ep7)
# q_next_vals8 = get_q_next(ep8)

# q_vals0[0, :]
# breakpoint()

# Plot all these to different plots in a folder
os.makedirs('plots', exist_ok=True)
for i in range(len(q_vals0)):
    plt.figure(figsize=(10, 6))
    plt.plot(q_vals0[i, :], label='ep0')
    # plt.plot(q_vals1[i, :], label='ep1')
    # plt.plot(q_vals2[i, :], label='ep2')
    # plt.plot(q_vals3[i, :], label='ep3')
    # plt.plot(q_vals4[i, :], label='ep4')
    # plt.plot(q_vals5[i, :], label='ep5')
    # plt.plot(q_vals6[i, :], label='ep6')
    # plt.plot(q_vals7[i, :], label='ep7')
    # plt.plot(q_vals8[i, :], label='ep8')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/q_vals_{i}.png')

for i in range(len(q_next_vals0)):
    plt.figure(figsize=(10, 6))
    plt.plot(q_next_vals0[i, :], label='ep0')
    # plt.plot(q_next_vals1[i, :], label='ep1')
    # plt.plot(q_next_vals2[i, :], label='ep2')
    # plt.plot(q_next_vals3[i, :], label='ep3')
    # plt.plot(q_next_vals4[i, :], label='ep4')
    # plt.plot(q_next_vals5[i, :], label='ep5')
    # plt.plot(q_next_vals6[i, :], label='ep6')
    # plt.plot(q_next_vals7[i, :], label='ep7')
    # plt.plot(q_next_vals8[i, :], label='ep8')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/q_next_vals_{i}.png')