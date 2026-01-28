"""Starter code from the RLPD repository https://github.com/ikostrikov/rlpd"""

from functools import partial
from typing import Dict, Optional, Sequence, Tuple

import flax
import gym
import jax
# jax.config.update("jax_log_compiles", True)
# jax.config.update("jax_explain_cache_misses", True)
# jax.config.update("jax_traceback_filtering", "off")  # more context in logs

import jax.numpy as jnp
import optax
from flax import struct
import flax.linen as nn
from flax.training.train_state import TrainState

# Openpi imports #
import openpi.shared.array_typing as at
import openpi.models.model as _model
import jax.tree_util as jtu

import numpy as np
import copy
import time
import pickle
import logging
import os

from jaxrl_m.utils.expo_utils import (
    TanhNormal,
    StateValue,
    StateActionValue,
    StateAndStateActionValue,
    ResidualActor,
    Ensemble,
    subsample_ensemble,
    Agent,
    Temperature,
    MLP,
    repeat_observations,
    repeat_observations_batched,
    repeat_observations_openpi,
    append_substr_to_dict_keys,
    add_batch_dim,
)
from jaxrl_m.common.typing import Batch, Data, PRNGKey


from jaxrl_m.agents.continuous.pi_0 import PiPolicy
from openpi.training.config import TrainConfig
import openpi.models.gemma as _gemma

class PiResidualBase(object):
    
    def sample_base_actions(self, _observations: Data, is_target=False, *args, **kwargs):
        '''
        For given state/observation, samles self.N actions from base policy; For first self.n_edit_samples actions, samples from edit_actor;
        Combine all (N + n_edit_samples) actions and compute Q-values; Return the action with the highest Q-value;
        '''
        out_dict = {}

        seed = kwargs.pop("seed", None)
        seed, rng = jax.random.split(seed)

        num_action_samples = kwargs.pop("num_action_samples", 1)

        # Repeat observations to sample `N` actions#
        # observations = repeat_observations(_observations, self.N, axis=0)
        observations = _observations

        rng_actions, rng = jax.random.split(seed)
        actions, vlm_output, processed_obs = self.actor.sample_actions_with_vlm_output(
            rng_actions, observations)

        if num_action_samples == 1:
            diffusion_actions = actions.copy()  # (1, action_horizon, action_dim)
        else:
            diffusion_actions = actions.copy()

            rng, rng_diffusion_samples = jax.random.split(rng)
            batched_obs = add_batch_dim(observations)
            observations_repeated = repeat_observations_openpi(
                batched_obs, num_action_samples, axis=0)
            diffusion_samples, _, _ = self.actor.sample_actions_with_vlm_output(
                rng_diffusion_samples, observations_repeated)
            diffusion_samples = self.actor.norm_actions(diffusion_samples)

        # Take mean across tokens as representation from VLM #
        # (1, pi0_hidden_dims)
        vlm_output = jnp.mean(vlm_output[0][:, :512, :], axis=1)
        state = processed_obs['state'][0, :8][None, :8]  # State dimension
        # (1, pi0_hidden_dims + state_dim)
        vlm_output = jnp.concatenate([vlm_output, state], axis=1)

        # action = diffusion_actions
        if num_action_samples == 1:
            action = diffusion_actions
        
        # Compute state values for caching and advantages calculations in PPO #
        # v_values = compute_v_all(self.critic.apply_fn, self.critic.params, vlm_output)

        rng, _ = jax.random.split(rng, 2)
        out_dict = {
            "actions": action.squeeze(),  # (action_horizon, action_dim)
            "vlm_output": vlm_output[0],  # (pi0_hidden_dims,)
            "diffusion_actions": diffusion_actions,
            # "state_values": v_values[0],  # (1,)
            "log_probs": jnp.array(-0.5),
        }

        if num_action_samples > 1:
            out_dict["action_samples"] = diffusion_samples

        return out_dict

    def get_vlm_output(self, _observations: Data, is_target=False, *args, **kwargs):
        '''
        For given state/observation, samles self.N actions from base policy; For first self.n_edit_samples actions, samples from edit_actor;
        Combine all (N + n_edit_samples) actions and compute Q-values; Return the action with the highest Q-value;
        '''
        seed = kwargs.pop("seed", None)
        seed, rng = jax.random.split(seed)
        # Do a forward pass to get VLM output #
        seed, rng = jax.random.split(rng)
        if not is_target:
            vlm_output, _, processed_obs = self.actor.get_vlm_output(
                rng, _observations, processed_obs=False, infer=True, return_processed_obs=True)
        else:
            vlm_output, _, processed_obs = self.target_actor.get_vlm_output(
                rng, _observations, processed_obs=False, infer=True, return_processed_obs=True)

        # Take mean across tokens as representation from VLM #
        # (N, pi0_hidden_dims)
        vlm_output = jnp.mean(vlm_output[0][:, :512, :], axis=1)
        state = processed_obs['state'][0, :8][None, :8]  # State dimension
        # (1, pi0_hidden_dims + state_dim)
        vlm_output = jnp.concatenate([vlm_output, state], axis=1)

        return vlm_output[0]
