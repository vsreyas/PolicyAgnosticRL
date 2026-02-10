"""GRPO variant of TD3 for residual policy learning.
Based on residual_td3.py with GRPO-style actor updates using advantage-weighted regression.
"""

from functools import partial
from typing import Dict, Optional, Sequence, Tuple

import flax
import gym
import jax
import jax.numpy as jnp
import os
import pickle
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

from jaxrl_m.utils.expo_utils import (
    TanhNormal,
    StateActionValue,
    StateAndStateActionValue,
    ResidualActor,
    ResidualTanhEditActor,
    MLPTanhEditActor,
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
from jaxrl_m.agents.continuous.pi_vlm_cached.residual_base import PiResidualBase


def decay_mask_fn(params):
    flat_params = flax.traverse_util.flatten_dict(params)
    flat_mask = {path: path[-1] != "bias" for path in flat_params}
    return flax.core.FrozenDict(flax.traverse_util.unflatten_dict(flat_mask))


@partial(jax.jit, static_argnames=('critic_fn'))
def compute_q(critic_fn, critic_params, observations, actions):

    q_values = critic_fn({'params': critic_params},
                         observations, actions, False)
    q_values = q_values.mean(axis=0)
    return q_values


@partial(jax.jit, static_argnames=('critic_fn'))
def compute_q_all(critic_fn, critic_params, observations, actions):
    q_values = critic_fn({'params': critic_params},
                         observations, actions, False)
    return q_values


@partial(jax.jit, static_argnames=('critic_fn', 'num_steps'))
def _gradient_ascent_actions(critic_fn, critic_params, vlm_output, start_action, eta_ascent, num_steps: int, zero_grad_gripper):
    """Perform gradient ascent on actions to maximize Q-value.

    Args:
        critic_fn: Critic network apply function
        critic_params: Critic network parameters
        vlm_output: VLM output features (1, vlm_dim)
        start_action: Initial action (action_dim,)
        eta_ascent: Step size for gradient ascent
        num_steps: Number of gradient ascent steps
        zero_grad_gripper: Scalar (0/1). If 1, zero out gradient for gripper dimension.

    Returns:
        Final action after gradient ascent (action_dim,)
    """
    def q_scalar(action):
        # Compute Q-value for a single action
        q = compute_q(critic_fn, critic_params, vlm_output, action.reshape(1, -1))
        # return q[0]
        return q.mean(axis=0)

    grad_fn = jax.grad(q_scalar)

    def ascent_step(i, action):
        g = grad_fn(action)
        g = g.reshape(10, 7)
        g = g.at[:, -1].set(g[:, -1] * (1.0 - zero_grad_gripper))
        g = g.reshape(-1)
        action = action + eta_ascent * g
        # action = jnp.clip(action, -1.0, 1.0)
        return action

    final_action = jax.lax.fori_loop(0, num_steps, ascent_step, start_action)
    return final_action


@partial(jax.jit, static_argnames=('critic_fn', 'num_steps', 'beta', 'epsilon'))
def _rmsprop_ascent_actions(critic_fn, critic_params, vlm_output, start_action, eta_ascent, num_steps: int, beta: float = 0.99, epsilon: float = 1e-8):
    """Perform RMSProp-based gradient ascent on actions to maximize Q-value.

    Args:
        critic_fn: Critic network apply function
        critic_params: Critic network parameters
        vlm_output: VLM output features (1, vlm_dim)
        start_action: Initial action (action_dim,)
        eta_ascent: Step size for gradient ascent
        num_steps: Number of gradient ascent steps
        beta: Decay rate for moving average of squared gradients
        epsilon: Small constant for numerical stability

    Returns:
        Final action after RMSProp ascent (action_dim,)
    """
    def q_scalar(action):
        # Compute Q-value for a single action
        q = compute_q(critic_fn, critic_params, vlm_output, action.reshape(1, -1))
        return q.mean(axis=0)
        # return q[0]

    grad_fn = jax.grad(q_scalar)

    def ascent_step(carry, i):
        action, v = carry
        g = grad_fn(action)
        v = beta * v + (1.0 - beta) * jnp.square(g)
        action = action + eta_ascent * g / (jnp.sqrt(v) + epsilon)
        action = jnp.clip(action, -1.0, 1.0)
        return (action, v), None

    # Initialize v (running average of squared gradients) to zeros
    v_init = jnp.zeros_like(start_action)
    (final_action, _), _ = jax.lax.scan(ascent_step, (start_action, v_init), jnp.arange(num_steps))
    return final_action


@partial(jax.jit, static_argnames=('critic_fn', 'num_steps', 'beta1', 'beta2', 'epsilon'))
def _adam_ascent_actions(critic_fn, critic_params, vlm_output, start_action, eta_ascent, num_steps: int, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
    """Perform Adam-based gradient ascent on actions to maximize Q-value.

    Args:
        critic_fn: Critic network apply function
        critic_params: Critic network parameters
        vlm_output: VLM output features (1, vlm_dim)
        start_action: Initial action (action_dim,)
        eta_ascent: Step size for gradient ascent
        num_steps: Number of gradient ascent steps
        beta1: Decay rate for first moment estimate
        beta2: Decay rate for second moment estimate
        epsilon: Small constant for numerical stability

    Returns:
        Final action after Adam ascent (action_dim,)
    """
    def q_scalar(action):
        # Compute Q-value for a single action
        q = compute_q(critic_fn, critic_params, vlm_output, action.reshape(1, -1))
        return q.mean(axis=0)

    grad_fn = jax.grad(q_scalar)

    def ascent_step(carry, i):
        action, m, v = carry
        t = i + 1  # Step count starts from 1
        g = grad_fn(action)

        # Update biased first and second moment estimates
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * jnp.square(g)

        # Compute bias-corrected moment estimates
        m_hat = m / (1.0 - jnp.power(beta1, t))
        v_hat = v / (1.0 - jnp.power(beta2, t))

        # Update action
        action = action + eta_ascent * m_hat / (jnp.sqrt(v_hat) + epsilon)
        # action = jnp.clip(action, -1.0, 1.0)

        return (action, m, v), None

    # Initialize m and v (first and second moment estimates) to zeros
    m_init = jnp.zeros_like(start_action)
    v_init = jnp.zeros_like(start_action)
    (final_action, _, _), _ = jax.lax.scan(ascent_step, (start_action, m_init, v_init), jnp.arange(num_steps))
    return final_action

# ----------------------------------------------------------------------
# Jitted inner steps for edit-actor and critic updates
# ----------------------------------------------------------------------


@partial(
    jax.jit,
    static_argnames=(
        "entropy_scale",
        "edit_action_scale",
        "edit_actor_apply_fn",
        "critic_apply_fn",
        "temp_apply_fn",
    ),
)
def _edit_actor_loss_and_grad_residual_q(
    edit_actor,
    actor_params,
    critic_params,
    temp_params,
    vlm_output,
    batch_actions,
    dropout_key,
    key,
    key2,
    entropy_scale: float,
    edit_action_scale: float,
    edit_actor_apply_fn,
    critic_apply_fn,
    temp_apply_fn,
    mc_target,
    success,
    bc_warmup,
):
    """TD3-style actor loss with residual actions: maximizes Q(base + scale * edit)."""

    def loss_fn(actor_params):
        edit_actions = edit_actor_apply_fn(
            {"params": actor_params}, vlm_output, batch_actions)
        edit_actions = edit_actions.reshape(-1, 10, 7)
        edit_actions = edit_actions.at[:, :, -1].set(0.0)
        edit_actions = edit_actions.reshape(batch_actions.shape)
        actions = batch_actions + edit_action_scale * edit_actions
        actions = jnp.clip(actions, -1.0, 1.0)

        qs = critic_apply_fn(
            {"params": critic_params},
            vlm_output,
            actions,
            False,
            rngs={"dropout": key2},
        )
        q = qs.mean(axis=0)

        qs_base = critic_apply_fn(
            {"params": critic_params},
            vlm_output,
            batch_actions,
            False,
            rngs={"dropout": key2},
        )
        q_base = qs_base.mean(axis=0)

        q_loss = -q.mean()
        zero_reg = 10.0 * jnp.square(edit_actions).mean()
        edit_actor_loss = q_loss # + zero_reg
        # edit_actor_loss = zero_reg

        metrics = {
            "edit_q_mean": q.mean(),
            "edit_q_min": q.min(),
            "edit_q_max": q.max(),
            "base_q_mean": q_base.mean(),
            "base_q_min": q_base.min(),
            "base_q_max": q_base.max(),
            "edit_actor_loss": edit_actor_loss,
            "q_loss": q_loss,
            "entropy": 0.0,
            "bc_warmup": jnp.float32(bc_warmup),
            "edit_action_scale": edit_action_scale,
            "zero_reg_loss": zero_reg,
        }

        edit_actions_reshaped = edit_actions.reshape(-1, 10, 7)
        for i in range(7):
            metrics[f"edit_action_{i}_mean"] = edit_actions_reshaped[:, :, i].mean()
            metrics[f"edit_action_{i}_std"] = edit_actions_reshaped[:, :, i].std()
            metrics[f"edit_action_{i}_max"] = edit_actions_reshaped[:, :, i].max()
            metrics[f"edit_action_{i}_min"] = edit_actions_reshaped[:, :, i].min()

        batch_actions_reshaped = batch_actions.reshape(-1, 10, 7)
        for i in range(7):
            metrics[f"batch_action_{i}_mean"] = batch_actions_reshaped[:, :, i].mean()
            metrics[f"batch_action_{i}_std"] = batch_actions_reshaped[:, :, i].std()
            metrics[f"batch_action_{i}_max"] = batch_actions_reshaped[:, :, i].max()
            metrics[f"batch_action_{i}_min"] = batch_actions_reshaped[:, :, i].min()

        return edit_actor_loss, metrics

    (loss, metrics), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(actor_params)
    edit_actor = edit_actor.apply_gradients(grads=grads)

    return edit_actor, grads, metrics


# @partial(
#     jax.jit,
#     static_argnames=(
#         "entropy_scale",
#         "edit_action_scale",
#         "edit_actor_apply_fn",
#         "critic_apply_fn",
#         "temp_apply_fn",
#         "grpo_beta",
#         "grpo_weight_threshold",
#     ),
# )
# def _edit_actor_loss_and_grad_grpo(
#     edit_actor,
#     actor_params,
#     critic_params,
#     temp_params,
#     vlm_output,
#     batch_actions,
#     dropout_key,
#     key,
#     key2,
#     entropy_scale: float,
#     edit_action_scale: float,
#     edit_actor_apply_fn,
#     critic_apply_fn,
#     temp_apply_fn,
#     mc_target,
#     success,
#     bc_warmup,
#     action_samples,  # (batch_size, num_samples, action_dim)
#     grpo_beta: float,  # Temperature for exp(advantage)
#     grpo_weight_threshold: float,  # Threshold for zeroing out small weights
# ):
#     """GRPO-style actor loss: advantage-weighted regression over action samples."""
#     batch_size = vlm_output.shape[0]
#     num_samples = action_samples.shape[1]
#     action_dim = action_samples.shape[2]
#
#     def loss_fn(actor_params):
#         key_q, key_qbase, key_samples = jax.random.split(key2, 3)
#         actions = edit_actor_apply_fn(
#             {"params": actor_params}, vlm_output, batch_actions)
#         edit_actions = actions.copy()
#         actions = jnp.clip(actions, -1.0, 1.0)
#         qs = critic_apply_fn(
#             {"params": critic_params}, vlm_output, actions, False, rngs={"dropout": key_q})
#         q = qs.mean(axis=0)
#         qs_base = critic_apply_fn(
#             {"params": critic_params}, vlm_output, batch_actions, False, rngs={"dropout": key_qbase})
#         q_base = qs_base.mean(axis=0)
#         sample_keys = jax.random.split(key_samples, num_samples)
#         def compute_q_for_sample(sample_actions, sample_key):
#             return critic_apply_fn(
#                 {"params": critic_params}, vlm_output, sample_actions, False,
#                 rngs={"dropout": sample_key}).mean(axis=0)
#         action_samples_transposed = jnp.transpose(action_samples, (1, 0, 2))
#         qs_samples = jax.vmap(compute_q_for_sample)(action_samples_transposed, sample_keys)
#         qs_samples = jnp.transpose(qs_samples, (1, 0))
#         q_mean = qs_samples.mean(axis=1, keepdims=True)
#         advantages = qs_samples - q_mean
#         weights = advantages
#         weights = jnp.where(weights >= grpo_weight_threshold, weights, 0.0)
#         weights = jnp.exp(weights / grpo_beta) - 1.0
#         actions_expanded = jnp.expand_dims(actions, axis=1)
#         sq_diff = jnp.square(actions_expanded - action_samples)
#         sq_diff_sum = sq_diff.sum(axis=-1)
#         grpo_loss = (weights * sq_diff_sum).sum(axis=1).mean()
#         success_float = jnp.float32(success)
#         bc_loss = (jnp.square(batch_actions - actions) * success_float).mean()
#         edit_actor_loss = bc_loss * 1000.0 + grpo_loss * (1.0 - bc_warmup)
#         metrics = { ... }
#         return edit_actor_loss, metrics
#     (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(actor_params)
#     edit_actor = edit_actor.apply_gradients(grads=grads)
#     return edit_actor, grads, metrics


@partial(jax.jit, static_argnames=("critic_apply_fn", "edit_actor_apply_fn", "target_critic_apply_fn", "tau"))
def _critic_loss_and_grad(
    critic_params,
    target_critic_params,
    vlm_output,
    actions,
    mc_target,
    key,
    critic_apply_fn,
    target_params,
    next_vlm_output,
    next_base_actions,
    edit_actor_apply_fn,
    edit_actor_params,
    target_critic_apply_fn,
    terminals,
    rewards,
    discount,
    critic,
    tau,
):
    """Single jitted step for critic: forward + loss + grad."""
    r_samples = _sample_deterministic_actions(
        edit_actor_apply_fn, edit_actor_params, next_vlm_output, next_base_actions)
    next_actions = r_samples
    next_actions = jnp.clip(next_actions, -1.0, 1.0)
    next_qs = compute_q(target_critic_apply_fn, target_params,
                        next_vlm_output, next_actions)
    masks = 1.0 - terminals
    target_q = rewards + discount * masks * next_qs
    target_q = jax.lax.stop_gradient(target_q)

    def loss_fn(critic_params):
        qs = critic_apply_fn(
            {"params": critic_params},
            vlm_output,
            actions,
            False,
            rngs={"dropout": key},
        )

        critic_loss_td = optax.losses.huber_loss(qs, target_q).mean()
        critic_loss_mc = ((qs - mc_target) ** 2).mean()
        critic_loss = critic_loss_td

        metrics = {
            "critic_loss": critic_loss,
            "critic_loss_td": critic_loss_td,
            "critic_loss_mc": critic_loss_mc,
            "q_mean": qs.mean(),
            "q_std": qs.std(),
            "q_max": qs.max(),
            "q_min": qs.min(),
            "target_q_mean": target_q.mean(),
            "target_q_std": target_q.std(),
            "target_q_max": target_q.max(),
            "target_q_min": target_q.min(),
            "q_diff": (qs - target_q).mean(),
            "q_diff_std": (qs - target_q).std(),
            "q_diff_max": (qs - target_q).max(),
            "q_diff_min": (qs - target_q).min(),
        }

        num_heads = qs.shape[0]
        for i in range(num_heads):
            qi = qs[i]
            metrics[f"q{i+1}_mean"] = qi.mean()
            metrics[f"q{i+1}_std"] = qi.std()
            metrics[f"q{i+1}_max"] = qi.max()
            metrics[f"q{i+1}_min"] = qi.min()

        return critic_loss, metrics

    (loss, metrics), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(critic_params)
    critic = critic.apply_gradients(grads=grads)
    target_critic_params = optax.incremental_update(
        critic.params, target_critic_params, tau
    )
    return critic, target_critic_params, grads, metrics


@partial(jax.jit, static_argnames=("critic_apply_fn", "target_critic_apply_fn", "tau"))
def _sarsa_loss_and_grad(
    critic_params,
    target_critic_params,
    vlm_output,
    actions,
    mc_target,
    key,
    critic_apply_fn,
    target_params,
    next_vlm_output,
    batch_next_actions,
    target_critic_apply_fn,
    terminals,
    rewards,
    discount,
    critic,
    tau,
    success,
    success_weight,
):
    """Single jitted step for critic: forward + loss + grad."""
    next_actions = jnp.clip(batch_next_actions, -1.0, 1.0)
    next_qs = compute_q(target_critic_apply_fn, target_params,
                        next_vlm_output, next_actions)
    masks = 1.0 - terminals
    target_q = rewards + discount * masks * next_qs
    target_q = jax.lax.stop_gradient(target_q)

    def loss_fn(critic_params):
        qs = critic_apply_fn(
            {"params": critic_params},
            vlm_output,
            actions,
            False,
            rngs={"dropout": key},
        )

        per_sample_loss = optax.losses.huber_loss(qs, target_q)  # (num_heads, batch_size)
        sample_weights = success * success_weight + (1.0 - success) * (1.0 - success_weight)
        critic_loss_td = (per_sample_loss * sample_weights[None, :]).mean()
        critic_loss_mc = ((qs - mc_target) ** 2).mean()
        critic_loss = critic_loss_td

        metrics = {
            "sarsa_critic_loss": critic_loss,
            "sarsa_critic_loss_td": critic_loss_td,
            "sarsa_critic_loss_mc": critic_loss_mc,
            "sarsa_success_weight": success_weight,
            "q_mean": qs.mean(),
            "q_std": qs.std(),
            "q_max": qs.max(),
            "q_min": qs.min(),
            "target_q_mean": target_q.mean(),
            "target_q_std": target_q.std(),
            "target_q_max": target_q.max(),
            "target_q_min": target_q.min(),
            "q_diff": (qs - target_q).mean(),
            "q_diff_std": (qs - target_q).std(),
            "q_diff_max": (qs - target_q).max(),
            "q_diff_min": (qs - target_q).min(),
        }

        num_heads = qs.shape[0]
        for i in range(num_heads):
            qi = qs[i]
            metrics[f"q{i+1}_mean"] = qi.mean()
            metrics[f"q{i+1}_std"] = qi.std()
            metrics[f"q{i+1}_max"] = qi.max()
            metrics[f"q{i+1}_min"] = qi.min()

        return critic_loss, metrics

    (loss, metrics), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(critic_params)
    critic = critic.apply_gradients(grads=grads)
    target_critic_params = optax.incremental_update(
        critic.params, target_critic_params, tau
    )
    return critic, target_critic_params, grads, metrics


@partial(jax.jit, static_argnames=("critic_apply_fn", "target_critic_apply_fn", "tau", "cql_alpha", "cql_temp", "calql_lower_bound", "calql_random_actions"))
def _calql_loss_and_grad(
    critic_params,
    target_critic_params,
    vlm_output,
    actions,
    mc_target,
    key,
    critic_apply_fn,
    target_params,
    next_vlm_output,
    batch_next_actions,
    target_critic_apply_fn,
    terminals,
    rewards,
    discount,
    critic,
    tau,
    action_samples,  # (batch_size, num_samples, action_dim)
    cql_alpha: float,
    cql_temp: float,
    calql_lower_bound: float,
    calql_random_actions: int = 0,
):
    """Cal-QL loss: TD loss + CQL regularization with constant lower bound.
    
    Cal-QL modifies CQL by applying max(Q, lower_bound) to OOD action Q-values,
    ensuring Q-values are calibrated (lower bounded by the constant).
    """
    # Compute TD target using SARSA-style next actions
    next_actions = jnp.clip(batch_next_actions, -1.0, 1.0)
    next_qs = compute_q(target_critic_apply_fn, target_params,
                        next_vlm_output, next_actions)
    masks = 1.0 - terminals
    target_q = rewards + discount * masks * next_qs
    target_q = jax.lax.stop_gradient(target_q)

    batch_size = actions.shape[0]
    action_dim = actions.shape[1]

    # Sample random actions in [-1, 1] if requested
    if calql_random_actions > 0:
        random_key, key = jax.random.split(key)
        random_actions = jax.random.uniform(
            random_key, (batch_size, calql_random_actions, action_dim), minval=-1.0, maxval=1.0
        )
        # Concatenate with existing action_samples
        action_samples = jnp.concatenate([action_samples, random_actions], axis=1)

    num_samples = action_samples.shape[1]

    def loss_fn(critic_params):
        # Split key for different uses to avoid RNG reuse
        key_data, key_ood_base = jax.random.split(key)

        # Q-values on dataset actions
        qs = critic_apply_fn(
            {"params": critic_params},
            vlm_output,
            actions,
            False,
            rngs={"dropout": key_data},
        )  # (num_heads, batch_size)

        # TD loss
        critic_loss_td = optax.losses.huber_loss(qs, target_q).mean()

        # CQL regularization: penalize high Q-values on OOD actions
        # Compute Q-values for action_samples (OOD actions from base policy)
        sample_keys = jax.random.split(key_ood_base, num_samples)
        def compute_q_for_sample(sample_actions, sample_key):
            return critic_apply_fn(
                {"params": critic_params},
                vlm_output,
                sample_actions,
                False,
                rngs={"dropout": sample_key},
            )  # (num_heads, batch_size)

        # Transpose to (num_samples, batch_size, action_dim) for vmapping
        action_samples_transposed = jnp.transpose(action_samples, (1, 0, 2))
        # Vmap over num_samples with unique RNG keys
        qs_ood = jax.vmap(compute_q_for_sample)(action_samples_transposed, sample_keys)
        # qs_ood shape: (num_samples, num_heads, batch_size)
        # Transpose to (num_heads, batch_size, num_samples)
        qs_ood = jnp.transpose(qs_ood, (1, 2, 0))

        # Cal-QL: Apply constant lower bound
        calql_bound_rate = (qs_ood < calql_lower_bound).mean()
        qs_ood_calibrated = jnp.maximum(qs_ood, calql_lower_bound)

        all_qs = qs_ood_calibrated

        # CQL loss: logsumexp(Q_ood) - Q_data
        cql_ood_values = (
            jax.scipy.special.logsumexp(all_qs / cql_temp, axis=-1) * cql_temp
        )  # (num_heads, batch_size)
        cql_q_diff = cql_ood_values - qs  # (num_heads, batch_size)
        cql_loss = cql_q_diff.mean()

        # Total loss
        critic_loss_mc = ((qs - mc_target) ** 2).mean()
        critic_loss = critic_loss_td + cql_alpha * cql_loss

        metrics = {
            "calql_critic_loss": critic_loss,
            "calql_critic_loss_td": critic_loss_td,
            "calql_critic_loss_mc": critic_loss_mc,
            "calql_cql_loss": cql_loss,
            "calql_cql_alpha": cql_alpha,
            "calql_lower_bound": calql_lower_bound,
            "calql_bound_rate": calql_bound_rate,
            "calql_ood_q_mean": qs_ood.mean(),
            "calql_ood_q_calibrated_mean": qs_ood_calibrated.mean(),
            "calql_ood_q_calibrated_max": qs_ood_calibrated.max(),
            "calql_ood_q_calibrated_min": qs_ood_calibrated.min(),
            "q_mean": qs.mean(),
            "q_std": qs.std(),
            "q_max": qs.max(),
            "q_min": qs.min(),
            "target_q_mean": target_q.mean(),
            "target_q_std": target_q.std(),
            "target_q_max": target_q.max(),
            "target_q_min": target_q.min(),
            "q_diff": (qs - target_q).mean(),
            "q_diff_std": (qs - target_q).std(),
            "q_diff_max": (qs - target_q).max(),
            "q_diff_min": (qs - target_q).min(),
        }

        num_heads = qs.shape[0]
        for i in range(num_heads):
            qi = qs[i]
            metrics[f"q{i+1}_mean"] = qi.mean()
            metrics[f"q{i+1}_std"] = qi.std()
            metrics[f"q{i+1}_max"] = qi.max()
            metrics[f"q{i+1}_min"] = qi.min()

        return critic_loss, metrics

    (loss, metrics), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(critic_params)
    critic = critic.apply_gradients(grads=grads)
    target_critic_params = optax.incremental_update(
        critic.params, target_critic_params, tau
    )
    return critic, target_critic_params, grads, metrics


@partial(jax.jit, static_argnames=("critic_apply_fn"))
def q_loss(
    critic_params,
    vlm_output,
    actions,
    mc_target,
    key,
    critic_apply_fn,
    critic,
):
    """Single jitted step for critic: forward + loss + grad."""

    def loss_fn(actions):
        qs = critic_apply_fn(
            {"params": critic_params},
            vlm_output,
            actions,
            False,
            rngs={"dropout": key},
        )

        q_loss = qs.mean()
        return q_loss, {"q_loss": q_loss}

    (loss, metrics), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(actions)

    return grads, metrics


@partial(jax.jit, static_argnames="apply_fn")
def _sample_deterministic_actions(apply_fn, params, observations: np.ndarray, base_actions: np.ndarray) -> np.ndarray:
    means = apply_fn({"params": params}, observations, base_actions)
    return means


class PiResidualTD3GRPO(Agent):
    """
    GRPO variant of TD3 for residual policy learning.
    Uses advantage-weighted regression over action samples for actor updates.
    """
    actor: PiPolicy
    critic: TrainState
    target_critic: TrainState
    target_actor: PiPolicy
    edit_actor: TrainState
    temp: TrainState
    action_dim: int = struct.field(pytree_node=False)
    state_dim: int = struct.field(pytree_node=False)
    action_horizon: int = struct.field(pytree_node=False)
    T: int = struct.field(pytree_node=False)
    N: int = struct.field(pytree_node=False)
    n_edit_samples: int = struct.field(pytree_node=False)
    edit_action_scale: float = struct.field(pytree_node=False)
    batch_split: int = struct.field(pytree_node=False)
    M: int = struct.field(pytree_node=False)
    ddpm_temperature: float
    actor_tau: float
    tau: float
    discount: float
    target_entropy: float
    entropy_scale: bool
    num_qs: int = struct.field(pytree_node=False)
    num_min_qs: Optional[int] = struct.field(pytree_node=False)
    backup_entropy: bool = struct.field(pytree_node=False)
    pi0_hidden_dims: int
    exploration_epsilon: float
    grpo_beta: float  # GRPO temperature for advantage weighting
    cql_alpha: float  # CQL regularization weight
    cql_temp: float  # CQL temperature for logsumexp
    residual_bc_actor: bool = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        config: TrainConfig,
        seed: int,
        batch_size: int,
        action_dim: int = 7,
        state_dim: int = 8,
        action_horizon: int = 10,
        pi0_hidden_dims: int = 4096,
        rng: PRNGKey | None = None,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (512, 512, 512, 512),
        discount: float = 0.99,
        tau: float = 0.005,
        num_qs: int = 10,
        num_min_qs: Optional[int] = 2,
        critic_dropout_rate: Optional[float] = None,
        critic_weight_decay: Optional[float] = None,
        critic_layer_norm: bool = True,
        params_path: Optional[str] = None,
        target_entropy: Optional[float] = None,
        entropy_scale: float = 1.0,
        init_temperature: float = 1.0,
        backup_entropy: bool = True,
        use_pnorm: bool = False,
        adjust_target_entropy: bool = True,
        use_critic_resnet: bool = False,
        time_dim: int = 128,
        actor_drop: Optional[float] = None,
        d_actor_drop: Optional[float] = None,
        T: int = 10,
        N: int = 4,
        batch_split: int = 1,
        M: int = 0,
        n_edit_samples: int = 4,
        edit_action_scale: float = 0.3,
        actor_layer_norm: bool = True,
        clip_sampler: bool = True,
        decay_steps: Optional[int] = int(3e6),
        actor_tau: float = 0.003,
        actor_dropout_rate: Optional[float] = None,
        actor_num_blocks: int = 3,
        ddpm_temperature: float = 1.0,
        beta_schedule: str = 'vp',
        batch_size_dict_key: str = 'actions',
        entropy_mul_scale_factor: float = 3.0,
        exploration_epsilon: float = 0.2,
        grpo_beta: float = 1.0,  # GRPO temperature
        cql_alpha: float = 5.0,  # CQL regularization weight
        cql_temp: float = 1.0,  # CQL temperature for logsumexp
        residual_bc_actor: bool = False,
        load_critic_params: bool = True,
        load_edit_actor_params: bool = False,
    ):
        assert N >= n_edit_samples, f"N must be greater than or equal to n_edit_samples, got N={N} and n_edit_samples={n_edit_samples}"

        paligemma_config = _gemma.get_config(config.model.paligemma_variant)
        pi0_hidden_dims = paligemma_config.width
        action_horizon = config.model.action_horizon
        action_dim = action_dim * action_horizon

        if target_entropy is None:
            target_entropy = -action_dim / 2.0

            if adjust_target_entropy:
                target_entropy = -action_dim / 2 + \
                    action_dim * jnp.log(edit_action_scale)

            target_entropy = target_entropy * entropy_mul_scale_factor

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        # Load params if `params_path` is provided
        critic_params = None
        target_critic_params = None
        edit_actor_params = None
        temp_params = None
        if params_path is not None:
            critic_params, target_critic_params, edit_actor_params, temp_params = load_checkpoint(params_path)

        # Init Pi0 model
        actor = PiPolicy(rng=rng, config=config, is_target=False)
        target_actor = actor

        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        # Init edit actor
        dummy_observations = jnp.ones(
            (batch_size, pi0_hidden_dims + state_dim))
        dummy_actions = jnp.ones((batch_size, action_dim))
        # edit_actor_def = ResidualTanhEditActor(
        #     action_dim, hidden_dims=hidden_dims, num_residual_blocks=3)
        edit_actor_def = MLPTanhEditActor(
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            activation=nn.swish,
            use_layer_norm=True,
            use_pnorm=use_pnorm,
        )
        edit_observations = jnp.concatenate(
            [dummy_observations, jnp.ones((batch_size, action_dim))], axis=1)

        if edit_actor_params is None or not load_edit_actor_params:
            print("\n\n\nInitializing edit actor parameters from scratch...\n\n\n")
            edit_actor_params = edit_actor_def.init(
                actor_key, dummy_observations, dummy_actions)["params"]
        else:
            print("\n\n\nInitializing edit actor parameters loaded from checkpoint...\n\n\n")

        edit_actor = TrainState.create(
            apply_fn=edit_actor_def.apply,
            params=edit_actor_params,
            tx=optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adam(learning_rate=actor_lr),
            )
        )

        # Init critic
        critic_base_cls = partial(
            MLP,
            hidden_dims=hidden_dims,
            activate_final=True,
            dropout_rate=critic_dropout_rate,
            use_layer_norm=critic_layer_norm,
            use_pnorm=use_pnorm,
            activations=nn.swish,
        )
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=num_qs)

        if critic_params is None or not load_critic_params:
            print("\n\n\nInitializing critic parameters from scratch...\n\n\n")
            critic_params = critic_def.init(
                critic_key, dummy_observations, dummy_actions)["params"]
        else:
            print("\n\n\nInitializing critic parameters loaded from checkpoint...\n\n\n")

        if critic_weight_decay is not None and critic_weight_decay > 0:
            tx = optax.adamw(
                learning_rate=critic_lr,
                weight_decay=critic_weight_decay,
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adam(learning_rate=critic_lr),
            )

        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=tx,
        )

        # Create target critic with same architecture
        target_critic_def = Ensemble(critic_cls, num=num_min_qs or num_qs)
        target_critic_init_params = target_critic_params if target_critic_params is not None else critic_params
        target_critic = TrainState.create(
            apply_fn=target_critic_def.apply,
            params=target_critic_init_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        temp_def = Temperature(init_temperature)

        if temp_params is None:
            print("\n\n\nInitializing temperature parameters from scratch...\n\n\n")
            temp_params = temp_def.init(temp_key)["params"]
        else:
            print("\n\n\nInitializing temperature parameters loaded from checkpoint...\n\n\n")

        temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adam(learning_rate=temp_lr),
            )
        )

        del dummy_observations, dummy_actions, edit_observations
        del edit_actor_params, critic_params, temp_params

        return cls(
            rng=rng,
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            target_actor=target_actor,
            edit_actor=edit_actor,
            action_dim=action_dim,
            action_horizon=action_horizon,
            state_dim=state_dim,
            T=T,
            N=N,
            n_edit_samples=n_edit_samples,
            edit_action_scale=edit_action_scale,
            batch_split=batch_split,
            M=M,
            actor_tau=actor_tau,
            ddpm_temperature=ddpm_temperature,
            temp=temp,
            target_entropy=target_entropy,
            entropy_scale=entropy_scale,
            tau=tau,
            discount=discount,
            num_qs=num_qs,
            num_min_qs=num_min_qs,
            backup_entropy=backup_entropy,
            pi0_hidden_dims=pi0_hidden_dims,
            exploration_epsilon=exploration_epsilon,
            grpo_beta=grpo_beta,
            cql_alpha=cql_alpha,
            cql_temp=cql_temp,
            residual_bc_actor=residual_bc_actor,
        )

    def sample_actions(self, _observations: Data, is_target=False, *args, **kwargs):
        '''
        Sample actions from edit actor. Also samples action_samples from base policy
        for GRPO training.
        '''
        out_dict = {}

        seed = kwargs.pop("seed", None)
        seed, rng = jax.random.split(seed)
        use_deterministic_actions = kwargs.pop(
            "use_deterministic_actions", False)
        num_action_samples = kwargs.pop("num_action_samples", 1)
        num_diffusion_samples = kwargs.pop("num_diffusion_samples", 1)

        observations = _observations

        # Do a forward pass to get VLM output
        seed, rng = jax.random.split(seed)

        actions, vlm_output, processed_obs = self.actor.sample_actions_with_vlm_output(
            rng, observations)
        diffusion_actions = actions.copy()
        actions = self.actor.norm_actions(actions)

        # Sample additional action samples for GRPO if requested
        if num_diffusion_samples > 1:
            rng, rng_diffusion_samples = jax.random.split(rng)
            batched_obs = add_batch_dim(observations)
            observations_repeated = repeat_observations_openpi(
                batched_obs, num_diffusion_samples, axis=0)
            diffusion_samples, _, _ = self.actor.sample_actions_with_vlm_output(
                rng_diffusion_samples, observations_repeated)
            diffusion_samples = self.actor.norm_actions(diffusion_samples)
        else:
            diffusion_samples = diffusion_actions

        # Take mean across tokens as representation from VLM
        vlm_output = jnp.mean(vlm_output[0][:, :512, :], axis=1)
        state = processed_obs['state'][0, :8][None, :8]
        vlm_output = jnp.concatenate([vlm_output, state], axis=1)

        seed, rng = jax.random.split(seed)
        actions = actions.reshape(1, self.action_dim)
        r_observations = vlm_output
        
        edit_actions = _sample_deterministic_actions(
            self.edit_actor.apply_fn, self.edit_actor.params, r_observations, actions)
        edit_actions = edit_actions.reshape(-1, 10, 7)
        edit_actions = edit_actions.at[:, :, -1].set(0.0)
        edit_actions = edit_actions.reshape(actions.shape)

        if not use_deterministic_actions:
            rng, rng_exploration = jax.random.split(rng)
            exploration_noise = jax.random.normal(
                rng_exploration, actions.shape) * self.exploration_epsilon
            exploration_noise = jnp.clip(exploration_noise, -1.0, 1.0)
            edit_actions = edit_actions + exploration_noise

        # edit_actions = jnp.clip(edit_actions, -1.0, 1.0)
        
        final_action = actions + self.edit_action_scale * edit_actions
        final_action = jnp.clip(final_action, -1.0, 1.0)
        final_action = final_action.reshape(
            self.action_horizon, self.action_dim // self.action_horizon)

        final_action = self.actor.unnorm_actions(final_action)

        rng, _ = jax.random.split(rng, 2)
        out_dict = {
            "actions": final_action,
            "vlm_output": vlm_output[0],
            "diffusion_actions": diffusion_actions,
        }

        # Add action_samples for GRPO
        if num_diffusion_samples > 1:
            out_dict["action_samples"] = diffusion_samples
        else:
            out_dict["action_samples"] = diffusion_actions

        return out_dict

    def sample_base_actions(self, _observations: Data, is_target=False, *args, **kwargs):
        '''
        Sample actions from base policy without edit actor.
        '''
        out_dict = {}

        seed = kwargs.pop("seed", None)
        seed, rng = jax.random.split(seed)

        num_action_samples = kwargs.pop("num_diffusion_samples", 1)

        observations = _observations

        rng_actions, rng = jax.random.split(seed)
        actions, vlm_output, processed_obs = self.actor.sample_actions_with_vlm_output(
            rng_actions, observations)

        if num_action_samples == 1:
            diffusion_actions = actions.copy()
        else:
            diffusion_actions = actions.copy()

            rng, rng_diffusion_samples = jax.random.split(rng)
            batched_obs = add_batch_dim(observations)
            observations_repeated = repeat_observations_openpi(
                batched_obs, num_action_samples, axis=0)
            diffusion_samples, _, _ = self.actor.sample_actions_with_vlm_output(
                rng_diffusion_samples, observations_repeated)
            diffusion_samples = self.actor.norm_actions(diffusion_samples)

        vlm_output = jnp.mean(vlm_output[0][:, :512, :], axis=1)
        state = processed_obs['state'][0, :8][None, :8]
        vlm_output = jnp.concatenate([vlm_output, state], axis=1)
        action = diffusion_actions

        rng, _ = jax.random.split(rng, 2)
        out_dict = {
            "actions": action.squeeze(),
            "vlm_output": vlm_output[0],
            "diffusion_actions": diffusion_actions,
        }

        if num_action_samples > 1:
            out_dict["action_samples"] = diffusion_samples
        else:
            out_dict["action_samples"] = diffusion_actions

        return out_dict

    def get_vlm_output(self, _observations: Data, is_target=False, *args, **kwargs):
        seed = kwargs.pop("seed", None)
        rng, _ = jax.random.split(seed)
        if not is_target:
            vlm_output, _, processed_obs = self.actor.get_vlm_output(
                rng, _observations, processed_obs=False, infer=True, return_processed_obs=True)
        else:
            vlm_output, _, processed_obs = self.target_actor.get_vlm_output(
                rng, _observations, processed_obs=False, infer=True, return_processed_obs=True)

        vlm_output = jnp.mean(vlm_output[0][:, :512, :], axis=1)
        state = processed_obs['state'][0, :8][None, :8]
        vlm_output = jnp.concatenate([vlm_output, state], axis=1)

        return vlm_output[0]
    
    def compute_q(self, vlm_output: np.ndarray, actions: np.ndarray, *args, **kwargs):
        """Compute Q-values for given VLM output and actions."""
        if vlm_output.ndim == 1:
            vlm_output = vlm_output.reshape(1, -1)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)

        q_values = compute_q_all(
            self.critic.apply_fn,
            self.critic.params,
            vlm_output,
            actions
        )
        return q_values[0]

    def sample_base_actions_gradq(self, _observations: Data, eta_ascent: float = 0.01, num_ascent_steps: int = 50,
                                    optimizer_type: str = "gradient_ascent",
                                    rmsprop_beta: float = 0.9, rmsprop_epsilon: float = 1e-8,
                                    adam_beta1: float = 0.9, adam_beta2: float = 0.999, adam_epsilon: float = 1e-8,
                                    zero_grad_gripper: bool = False,
                                    *args, **kwargs):
        """Sample base actions and ascend them using gradient of Q-function.

        Args:
            _observations: Observations from environment
            eta_ascent: Step size for gradient ascent
            num_ascent_steps: Number of gradient ascent steps
            optimizer_type: Type of optimizer to use ("gradient_ascent", "rmsprop", "adam")
            rmsprop_beta: Decay rate for RMSProp moving average
            rmsprop_epsilon: Small constant for RMSProp numerical stability
            adam_beta1: First moment decay rate for Adam
            adam_beta2: Second moment decay rate for Adam
            adam_epsilon: Small constant for Adam numerical stability
            zero_grad_gripper: If True, zero out gradient for gripper (last of 7) dimension

        Returns:
            Dictionary with ascended actions and metadata
        """
        # First sample base actions
        out_dict = self.sample_base_actions(_observations, *args, **kwargs)

        # Get the base action and vlm_output
        base_action = out_dict["actions"]  # (action_horizon, action_dim)
        vlm_output = out_dict["vlm_output"]  # (vlm_dim,)

        # Normalize and flatten the action for gradient ascent
        base_action_norm = self.actor.norm_actions(base_action.reshape(1, self.action_horizon, -1))
        start_action = jnp.array(base_action_norm.reshape(-1))  # Flatten to (action_dim,)
        vlm_output_expanded = jnp.array(vlm_output.reshape(1, -1))

        # Perform JIT-compiled gradient ascent with selected optimizer
        zero_grad_gripper_scalar = jnp.float32(zero_grad_gripper)
        if optimizer_type == "gradient_ascent":
            final_action_flat = _gradient_ascent_actions(
                self.critic.apply_fn,
                self.critic.params,
                vlm_output_expanded,
                start_action,
                eta_ascent,
                num_ascent_steps,
                zero_grad_gripper_scalar
            )
        elif optimizer_type == "rmsprop":
            final_action_flat = _rmsprop_ascent_actions(
                self.critic.apply_fn,
                self.critic.params,
                vlm_output_expanded,
                start_action,
                eta_ascent,
                num_ascent_steps,
                rmsprop_beta,
                rmsprop_epsilon
            )
        elif optimizer_type == "adam":
            final_action_flat = _adam_ascent_actions(
                self.critic.apply_fn,
                self.critic.params,
                vlm_output_expanded,
                start_action,
                eta_ascent,
                num_ascent_steps,
                adam_beta1,
                adam_beta2,
                adam_epsilon
            )
        else:
            raise ValueError(f"Unknown optimizer_type: {optimizer_type}. Must be one of: gradient_ascent, rmsprop, adam")

        # Debug: Print norm between final and start actions
        # action_diff_norm = jnp.linalg.norm(final_action_flat - start_action)
        # print(f"[DEBUG] Action diff norm (final - start): {action_diff_norm}")

        # Reshape back to (action_horizon, action_dim)
        final_action = final_action_flat.reshape(self.action_horizon, self.action_dim // self.action_horizon)

        # Unnormalize actions before returning
        final_action = self.actor.unnorm_actions(final_action)

        out_dict["actions"] = final_action
        return out_dict

    def sample_bon_actions(self, _observations: Data, bon_actions: int = 4, *args, **kwargs):
        """Sample multiple base actions and select the best one based on Q-values (no gradient ascent).

        This implements a Best-of-N (BON) approach where we:
        1. Sample 'bon_actions' different actions from the base policy (batched)
        2. Subsample num_min_qs Q networks from num_qs
        3. Compute Q values using min(Q1(s,a), Q2(s,a)) for all sampled actions
        4. Select the action with the highest Q value

        Args:
            _observations: Observations from environment
            bon_actions: Number of actions to sample from base policy

        Returns:
            Dictionary with best action and metadata
        """
        seed = kwargs.get("seed", None)
        rng = seed

        # Repeat observations bon_actions times for batched sampling
        batched_obs = add_batch_dim(_observations)
        observations_repeated = repeat_observations_openpi(batched_obs, bon_actions, axis=0)

        # Sample bon_actions different actions from base policy in a single batched call
        sample_rng, rng = jax.random.split(rng)
        sampled_actions, vlm_output, processed_obs = self.actor.sample_actions_with_vlm_output(
            sample_rng, observations_repeated
        )
        # sampled_actions shape: (bon_actions, action_horizon, action_dim)

        # Normalize actions
        sampled_actions_norm = self.actor.norm_actions(sampled_actions)
        # Flatten actions: (bon_actions, action_dim_total)
        sampled_actions_flat = sampled_actions_norm.reshape(bon_actions, -1)

        # Get VLM output (use the first one since they're all the same observation)
        vlm_output_single = jnp.mean(vlm_output[0][0:1, :512, :], axis=1)  # Take only first, shape (1, vlm_dim)
        state = processed_obs['state'][0, :8][None, :8]
        vlm_output_final = jnp.concatenate([vlm_output_single, state], axis=1)  # (1, vlm_dim + state_dim)

        # Subsample num_min_qs Q networks from num_qs
        subsample_rng, rng = jax.random.split(rng)
        subsampled_critic_params = subsample_ensemble(
            subsample_rng, self.critic.params, self.num_min_qs, self.num_qs
        )

        # Compute Q-values for all sampled actions in a batched manner
        # Expand vlm_output to match batch size
        vlm_output_repeated = jnp.repeat(vlm_output_final, bon_actions, axis=0)  # (bon_actions, vlm_dim)

        # Compute Q values using subsampled critics
        # Use target_critic.apply_fn because it's designed to work with num_min_qs critics
        qs = compute_q_all(
            self.target_critic.apply_fn,
            subsampled_critic_params,
            vlm_output_repeated,
            sampled_actions_flat
        )  # (num_min_qs, bon_actions)

        # Take minimum across the Q networks for each action
        q_min_values = jnp.min(qs, axis=0)  # (bon_actions,)

        # Select the action with the highest Q value
        best_action_idx = jnp.argmax(q_min_values)
        best_action_flat = sampled_actions_flat[best_action_idx]
        best_q_value = q_min_values[best_action_idx]

        # Reshape back to (action_horizon, action_dim)
        final_action = best_action_flat.reshape(self.action_horizon, self.action_dim // self.action_horizon)

        # Unnormalize actions before returning
        final_action = self.actor.unnorm_actions(final_action)

        out_dict = {
            "actions": final_action,
            "vlm_output": vlm_output_final[0],
            "diffusion_actions": sampled_actions[0],  # Return first sampled action as diffusion_actions
            "action_samples": sampled_actions,  # Return all sampled actions
            "bon_q_values": q_min_values,
            "bon_best_q_value": best_q_value,
            "bon_best_idx": best_action_idx,
        }

        return out_dict

    def sample_bon_actions_gradq(self, _observations: Data, bon_actions: int = 4,
                                  eta_ascent: float = 0.01, num_ascent_steps: int = 50,
                                  optimizer_type: str = "gradient_ascent",
                                  rmsprop_beta: float = 0.9, rmsprop_epsilon: float = 1e-8,
                                  adam_beta1: float = 0.9, adam_beta2: float = 0.999, adam_epsilon: float = 1e-8,
                                  zero_grad_gripper: bool = False,
                                  *args, **kwargs):
        """Sample multiple base actions, ascend each, and select the best one based on Q-values.

        This implements a Best-of-N (BON) approach where we:
        1. Sample 'bon_actions' different actions from the base policy (batched)
        2. Perform gradient ascent on each action separately (vmapped)
        3. Subsample num_min_qs Q networks from num_qs
        4. Compute Q values using min(Q1(s,a), Q2(s,a)) for all ascended actions
        5. Select the action with the highest Q value

        Args:
            _observations: Observations from environment
            bon_actions: Number of actions to sample from base policy
            eta_ascent: Step size for gradient ascent
            num_ascent_steps: Number of gradient ascent steps
            optimizer_type: Type of optimizer to use ("gradient_ascent", "rmsprop", "adam")
            rmsprop_beta: Decay rate for RMSProp moving average
            rmsprop_epsilon: Small constant for RMSProp numerical stability
            adam_beta1: First moment decay rate for Adam
            adam_beta2: Second moment decay rate for Adam
            adam_epsilon: Small constant for Adam numerical stability
            zero_grad_gripper: If True, zero out gradient for gripper (last of 7) dimension

        Returns:
            Dictionary with best ascended action and metadata
        """
        seed = kwargs.get("seed", None)
        rng = seed

        # Repeat observations bon_actions times for batched sampling
        batched_obs = add_batch_dim(_observations)
        observations_repeated = repeat_observations_openpi(batched_obs, bon_actions, axis=0)

        # Sample bon_actions different actions from base policy in a single batched call
        sample_rng, rng = jax.random.split(rng)
        kwargs_sample = {**kwargs, "seed": sample_rng}
        sampled_actions, vlm_output, processed_obs = self.actor.sample_actions_with_vlm_output(
            sample_rng, observations_repeated
        )
        # sampled_actions shape: (bon_actions, action_horizon, action_dim)

        # Normalize actions for gradient ascent
        sampled_actions_norm = self.actor.norm_actions(sampled_actions)
        # Flatten actions: (bon_actions, action_dim_total)
        sampled_actions_flat = sampled_actions_norm.reshape(bon_actions, -1)

        # Get VLM output (use the first one since they're all the same observation)
        vlm_output_single = jnp.mean(vlm_output[0][0:1, :512, :], axis=1)  # Take only first, shape (1, vlm_dim)
        state = processed_obs['state'][0, :8][None, :8]
        vlm_output_final = jnp.concatenate([vlm_output_single, state], axis=1)  # (1, vlm_dim + state_dim)

        # Select the ascent function based on optimizer_type
        zero_grad_gripper_scalar = jnp.float32(zero_grad_gripper)
        if optimizer_type == "gradient_ascent":
            ascent_fn = _gradient_ascent_actions
            ascent_args = (eta_ascent, num_ascent_steps, zero_grad_gripper_scalar)
        elif optimizer_type == "rmsprop":
            ascent_fn = _rmsprop_ascent_actions
            ascent_args = (eta_ascent, num_ascent_steps, rmsprop_beta, rmsprop_epsilon)
        elif optimizer_type == "adam":
            ascent_fn = _adam_ascent_actions
            ascent_args = (eta_ascent, num_ascent_steps, adam_beta1, adam_beta2, adam_epsilon)
        else:
            raise ValueError(f"Unknown optimizer_type: {optimizer_type}. Must be one of: gradient_ascent, rmsprop, adam")

        # Vmap the ascent function over all sampled actions
        # Create a vectorized version that processes all actions in parallel
        def ascend_single_action(start_action):
            return ascent_fn(
                self.critic.apply_fn,
                self.critic.params,
                vlm_output_final,
                start_action,
                *ascent_args
            )

        ascended_actions_flat = jax.vmap(ascend_single_action)(sampled_actions_flat)
        # ascended_actions_flat shape: (bon_actions, action_dim_total)

        # Subsample num_min_qs Q networks from num_qs
        subsample_rng, rng = jax.random.split(rng)
        subsampled_critic_params = subsample_ensemble(
            subsample_rng, self.critic.params, self.num_min_qs, self.num_qs
        )

        # Compute Q-values for all ascended actions in a batched manner
        # Expand vlm_output to match batch size
        vlm_output_repeated = jnp.repeat(vlm_output_final, bon_actions, axis=0)  # (bon_actions, vlm_dim)

        # Compute Q values using subsampled critics with compute_q_all
        # Use target_critic.apply_fn because it's designed to work with num_min_qs critics
        qs = compute_q_all(
            self.target_critic.apply_fn,
            subsampled_critic_params,
            vlm_output_repeated,
            ascended_actions_flat
        )  # (num_min_qs, bon_actions)

        # Take minimum across the Q networks for each action
        q_min_values = jnp.min(qs, axis=0)  # (bon_actions,)

        # Select the action with the highest Q value
        best_action_idx = jnp.argmax(q_min_values)
        best_action_flat = ascended_actions_flat[best_action_idx]
        best_q_value = q_min_values[best_action_idx]

        # Reshape back to (action_horizon, action_dim)
        final_action = best_action_flat.reshape(self.action_horizon, self.action_dim // self.action_horizon)

        # Unnormalize actions before returning
        final_action = self.actor.unnorm_actions(final_action)

        out_dict = {
            "actions": final_action,
            "vlm_output": vlm_output_final[0],
            "diffusion_actions": sampled_actions[0],  # Return first sampled action as diffusion_actions
            "action_samples": sampled_actions,  # Return all sampled actions
            "bon_q_values": q_min_values,
            "bon_best_q_value": best_q_value,
            "bon_best_idx": best_action_idx,
        }

        return out_dict

    def sample_bon_actions_gradq_v2(self, _observations: Data, bon_actions: int = 4,
                                     eta_ascent: float = 0.01, num_ascent_steps: int = 50,
                                     optimizer_type: str = "gradient_ascent",
                                     rmsprop_beta: float = 0.9, rmsprop_epsilon: float = 1e-8,
                                     adam_beta1: float = 0.9, adam_beta2: float = 0.999, adam_epsilon: float = 1e-8,
                                     zero_grad_gripper: bool = False,
                                     *args, **kwargs):
        """Sample multiple base actions, select the best via Q-ranking, then gradient ascend the best.

        Unlike sample_bon_actions_gradq which ascends ALL sampled actions before ranking,
        this version:
        1. Samples 'bon_actions' different actions from the base policy (batched)
        2. Subsamples num_min_qs Q networks from num_qs
        3. Ranks the sampled actions using min(Q) across subsampled networks
        4. Selects the action with the highest Q value
        5. Performs gradient ascent only on the selected best action

        Args:
            _observations: Observations from environment
            bon_actions: Number of actions to sample from base policy
            eta_ascent: Step size for gradient ascent
            num_ascent_steps: Number of gradient ascent steps
            optimizer_type: Type of optimizer to use ("gradient_ascent", "rmsprop", "adam")
            rmsprop_beta: Decay rate for RMSProp moving average
            rmsprop_epsilon: Small constant for RMSProp numerical stability
            adam_beta1: First moment decay rate for Adam
            adam_beta2: Second moment decay rate for Adam
            adam_epsilon: Small constant for Adam numerical stability
            zero_grad_gripper: If True, zero out gradient for gripper (last of 7) dimension

        Returns:
            Dictionary with best ascended action and metadata
        """
        seed = kwargs.get("seed", None)
        rng = seed

        # Repeat observations bon_actions times for batched sampling
        batched_obs = add_batch_dim(_observations)
        observations_repeated = repeat_observations_openpi(batched_obs, bon_actions, axis=0)

        # Sample bon_actions different actions from base policy in a single batched call
        sample_rng, rng = jax.random.split(rng)
        sampled_actions, vlm_output, processed_obs = self.actor.sample_actions_with_vlm_output(
            sample_rng, observations_repeated
        )
        # sampled_actions shape: (bon_actions, action_horizon, action_dim)

        # Normalize actions
        sampled_actions_norm = self.actor.norm_actions(sampled_actions)
        # Flatten actions: (bon_actions, action_dim_total)
        sampled_actions_flat = sampled_actions_norm.reshape(bon_actions, -1)

        # Get VLM output (use the first one since they're all the same observation)
        vlm_output_single = jnp.mean(vlm_output[0][0:1, :512, :], axis=1)  # Take only first, shape (1, vlm_dim)
        state = processed_obs['state'][0, :8][None, :8]
        vlm_output_final = jnp.concatenate([vlm_output_single, state], axis=1)  # (1, vlm_dim + state_dim)

        # --- Step 1: Rank sampled actions using subsampled Q networks ---
        subsample_rng, rng = jax.random.split(rng)
        subsampled_critic_params = subsample_ensemble(
            subsample_rng, self.critic.params, self.num_min_qs, self.num_qs
        )

        # Expand vlm_output to match batch size
        vlm_output_repeated = jnp.repeat(vlm_output_final, bon_actions, axis=0)  # (bon_actions, vlm_dim)

        # Compute Q values using subsampled critics
        qs = compute_q_all(
            self.target_critic.apply_fn,
            subsampled_critic_params,
            vlm_output_repeated,
            sampled_actions_flat
        )  # (num_min_qs, bon_actions)

        # Take minimum across the Q networks for each action
        q_min_values = jnp.min(qs, axis=0)  # (bon_actions,)

        # Select the action with the highest Q value
        best_action_idx = jnp.argmax(q_min_values)
        best_action_flat = sampled_actions_flat[best_action_idx]
        best_q_value_before = q_min_values[best_action_idx]

        # --- Step 2: Gradient ascend only the best action ---
        zero_grad_gripper_scalar = jnp.float32(zero_grad_gripper)
        if optimizer_type == "gradient_ascent":
            final_action_flat = _gradient_ascent_actions(
                self.critic.apply_fn,
                self.critic.params,
                vlm_output_final,
                best_action_flat,
                eta_ascent,
                num_ascent_steps,
                zero_grad_gripper_scalar
            )
        elif optimizer_type == "rmsprop":
            final_action_flat = _rmsprop_ascent_actions(
                self.critic.apply_fn,
                self.critic.params,
                vlm_output_final,
                best_action_flat,
                eta_ascent,
                num_ascent_steps,
                rmsprop_beta,
                rmsprop_epsilon
            )
        elif optimizer_type == "adam":
            final_action_flat = _adam_ascent_actions(
                self.critic.apply_fn,
                self.critic.params,
                vlm_output_final,
                best_action_flat,
                eta_ascent,
                num_ascent_steps,
                adam_beta1,
                adam_beta2,
                adam_epsilon
            )
        else:
            raise ValueError(f"Unknown optimizer_type: {optimizer_type}. Must be one of: gradient_ascent, rmsprop, adam")

        # Reshape back to (action_horizon, action_dim)
        final_action = final_action_flat.reshape(self.action_horizon, self.action_dim // self.action_horizon)

        # Unnormalize actions before returning
        final_action = self.actor.unnorm_actions(final_action)

        out_dict = {
            "actions": final_action,
            "vlm_output": vlm_output_final[0],
            "diffusion_actions": sampled_actions[0],  # Return first sampled action as diffusion_actions
            "action_samples": sampled_actions,  # Return all sampled actions
            "bon_q_values": q_min_values,
            "bon_best_q_value_before_ascent": best_q_value_before,
            "bon_best_idx": best_action_idx,
        }

        return out_dict

    def get_edit_action(self, base_actions: np.ndarray, vlm_output: np.ndarray, clip: bool = True) -> np.ndarray:
        """Compute edit actions given base actions and VLM output.
        
        Args:
            base_actions: Base actions from the policy, shape (batch_size, action_dim) or (action_dim,)
            vlm_output: VLM output features, shape (batch_size, vlm_dim) or (vlm_dim,)
            clip: Whether to clip actions to [-1, 1]
            
        Returns:
            edit_actions: Edited actions, same shape as base_actions
        """
        # Handle 1D inputs
        squeeze_output = False
        if base_actions.ndim == 1:
            base_actions = base_actions.reshape(1, -1)
            squeeze_output = True
        if vlm_output.ndim == 1:
            vlm_output = vlm_output.reshape(1, -1)
        
        # Normalize base actions
        base_actions = base_actions.reshape(-1, self.action_horizon, self.action_dim // self.action_horizon)
        base_actions_norm = self.actor.norm_actions(base_actions)
        base_actions_norm = base_actions_norm.reshape(-1, self.action_dim)
        
        # Compute edit actions
        edit_actions = self.edit_actor.apply_fn(
            {"params": self.edit_actor.params}, vlm_output, base_actions_norm
        )
        
        if clip:
            edit_actions = jnp.clip(edit_actions, -1.0, 1.0)
        
        # Unnormalize actions
        # edit_actions = self.actor.unnorm_actions(edit_actions)
        
        if squeeze_output:
            edit_actions = edit_actions[0]
        
        return edit_actions

    def update_edit_actor(self, batch: Batch, *args, **kwargs) -> Tuple[Agent, Dict[str, float]]:
        seed = kwargs.pop("seed", None)
        bc_warmup = kwargs.pop("bc_warmup", 0.0)

        assert seed is not None, "seed must be provided"
        rng = seed

        base_actions = batch['diffusion_actions']
        base_actions = self.actor.norm_actions(base_actions)
        vlm_output = batch['vlm_output']
        base_actions = base_actions.reshape(-1, self.action_dim)

        dropout_rng, rng = jax.random.split(rng)
        rng1, rng = jax.random.split(rng)
        rng2, rng = jax.random.split(rng)

        # Residual Q-maximizing loss: new_action = base + scale * edit, maximize Q(new_action)
        edit_actor, grads, actor_info = _edit_actor_loss_and_grad_residual_q(
            self.edit_actor,
            self.edit_actor.params,
            self.critic.params,
            self.temp.params,
            vlm_output,
            base_actions,
            dropout_rng,
            rng1,
            rng2,
            self.entropy_scale,
            self.edit_action_scale,
            self.edit_actor.apply_fn,
            self.critic.apply_fn,
            self.temp.apply_fn,
            batch["mc_returns"],
            batch["success"][:, None],
            float(bc_warmup),
        )

        actor_info["edit_actor_grad_norm"] = optax.global_norm(grads)
        actor_info["edit_actor_param_norm"] = optax.global_norm(edit_actor.params)
        actor_info["target_entropy"] = self.target_entropy

        return self.replace(edit_actor=edit_actor, rng=rng), actor_info

    def update_critic(self, batch, *args, **kwargs) -> Tuple[TrainState, Dict[str, float]]:
        seed = kwargs.pop("seed", None)
        timer = kwargs.pop("timer", None)
        update_sarsa = kwargs.pop("update_sarsa", False)
        update_calql = kwargs.pop("update_calql", False)
        calql_lower_bound = kwargs.pop("calql_lower_bound", -1000.0)
        cql_alpha = kwargs.pop("cql_alpha", self.cql_alpha)
        cql_temp = kwargs.pop("cql_temp", self.cql_temp)
        calql_random_actions = kwargs.pop("calql_random_actions", 0)
        success_weight = kwargs.pop("success_weight", 0.5)
        assert seed is not None, "seed must be provided"
        rng = seed

        next_base_actions = batch['next_diffusion_actions']
        next_base_actions = self.actor.norm_actions(next_base_actions)
        next_vlm_output = batch['next_vlm_output']
        current_vlm_output = batch['vlm_output']
        actions = batch["actions"].reshape(-1, self.action_dim)
        next_base_actions = next_base_actions.reshape(-1, self.action_dim)
        subsample_rng, rng = jax.random.split(rng)
        target_params = subsample_ensemble(
            subsample_rng, self.target_critic.params, self.num_min_qs, self.num_qs
        )

        timer.tick("critic_loss_and_grad_time")
        rng1, rng = jax.random.split(rng)

        if update_calql:
            # Cal-QL warmup: use action_samples as OOD actions with constant lower bound
            action_samples = batch["action_samples"]  # (batch_size, num_samples, action_horizon, action_dim)
            # Normalize and reshape action samples
            action_samples = self.actor.norm_actions(action_samples)
            action_samples = action_samples.reshape(action_samples.shape[0], action_samples.shape[1], -1)  # (batch_size, num_samples, action_dim)
            
            critic, target_critic_params, grads, info = _calql_loss_and_grad(
                self.critic.params,
                self.target_critic.params,
                current_vlm_output,
                actions,
                batch["mc_returns"],
                rng1,
                self.critic.apply_fn,
                target_params,
                next_vlm_output,
                batch["next_actions"].reshape(-1, self.action_dim),
                self.target_critic.apply_fn,
                batch["terminals"],
                batch["rewards"],
                self.discount,
                self.critic,
                self.tau,
                action_samples,
                cql_alpha,
                cql_temp,
                calql_lower_bound,
                calql_random_actions,
            )

        elif update_sarsa:
            critic, target_critic_params, grads, info = _sarsa_loss_and_grad(
                self.critic.params,
                self.target_critic.params,
                current_vlm_output,
                actions,
                batch["mc_returns"],
                rng1,
                self.critic.apply_fn,
                target_params,
                next_vlm_output,
                batch["next_actions"].reshape(-1, self.action_dim),
                self.target_critic.apply_fn,
                batch["terminals"],
                batch["rewards"],
                self.discount,
                self.critic,
                self.tau,
                batch["success"],
                success_weight,
            )

        else:
            critic, target_critic_params, grads, info = _critic_loss_and_grad(
                self.critic.params,
                self.target_critic.params,
                current_vlm_output,
                actions,
                batch["mc_returns"],
                rng1,
                self.critic.apply_fn,
                target_params,
                next_vlm_output,
                next_base_actions,
                self.edit_actor.apply_fn,
                self.edit_actor.params,
                self.target_critic.apply_fn,
                batch["terminals"],
                batch["rewards"],
                self.discount,
                self.critic,
                self.tau,
            )
        timer.tock("critic_loss_and_grad_time")

        timer.tick("apply_gradients_time")
        info["critic_grad_norm"] = optax.global_norm(grads)
        timer.tock("apply_gradients_time")

        timer.tick("incremental_update_time")
        target_critic = self.target_critic.replace(params=target_critic_params)
        timer.tock("incremental_update_time")

        info["critic_param_norm"] = optax.global_norm(critic.params)
        info["target_critic_param_norm"] = optax.global_norm(target_critic.params)

        return self.replace(critic=critic, target_critic=target_critic, rng=rng), info

    def preproess_batch(self, batch: Batch, *args, **kwargs) -> Batch:
        action_dim = self.action_dim // self.action_horizon
        if batch['actions'].shape[-1] > action_dim:
            if batch['actions'].ndim == 3:
                batch['actions'] = batch['actions'][:, :, :action_dim]
            elif batch['actions'].ndim == 2:
                batch['actions'] = batch['actions'][:, :action_dim]
            else:
                raise ValueError(
                    f"Actions must be 3 or 2 dimensional, got {batch['actions'].shape}")

        return batch

    @staticmethod
    def _resolve_critic_update_type(critic_warmup, critic_warmup_type, online_critic_update_type):
        """Factory method to determine critic update flags from parameters.

        Returns:
            (update_sarsa, update_calql) boolean tuple.
        """
        if critic_warmup:
            if critic_warmup_type == "sarsa":
                return True, False
            elif critic_warmup_type == "calql":
                return False, True
            else:
                raise ValueError(f"Unknown critic_warmup_type: {critic_warmup_type}. Must be 'sarsa' or 'calql'.")
        elif online_critic_update_type is not None:
            if online_critic_update_type == "sarsa":
                return True, False
            elif online_critic_update_type == "calql":
                return False, True
            else:
                raise ValueError(f"Unknown online_critic_update_type: {online_critic_update_type}. Must be 'sarsa' or 'calql'.")
        else:
            return False, False

    def update(
                self,
                _observations: Data,
                utd_ratio: int,
                update_critic: bool = True,
                update_edit_actor: bool = True,
                critic_warmup: bool = False,
                critic_warmup_type: str = "sarsa",  # "sarsa" or "calql"
                calql_lower_bound: float = -1000.0,
                cql_alpha: float = None,
                cql_temp: float = None,
                calql_random_actions: int = 0,
                edit_actor_warmup: bool = False,
                grpo_beta: float = None,
                grpo_weight_threshold: float = 0.0,
                edit_actor_utd_ratio: int = 1,
                online_critic_update_type: Optional[str] = None,
                success_weight: float = 0.5,
                *args, **kwargs
            ):
        timer = kwargs.pop("timer", None)
        new_agent = self
        timer.tick("total_update_time")
        timer.tick("preprocess_time")

        batch = self.preproess_batch(_observations.copy())
        state = batch['observations']['proprio'][:, :8].copy()
        next_state = batch['next_observations']['proprio'][:, :8].copy()

        relevant_keys = [
            "actions", "rewards", "masks", "mc_returns",
            "terminals", "truncates", "vlm_output", "next_vlm_output", "diffusion_actions", "next_diffusion_actions",
            "next_actions", "success", "action_samples",
        ]
        batch = {k: v for k, v in batch.items() if k in relevant_keys}
        batch['state'] = state
        batch['next_state'] = next_state

        timer.tock("preprocess_time")

        batch_size = state.shape[0]
        print("---")
        print("Batch size for update: ", batch_size)

        seed = kwargs.pop("seed", None)
        assert seed is not None, "Seed must be provided"
        rng = seed

        critic_info = {}
        actor_update_info = {}
        actor_info = {}

        # Determine critic update type via factory method
        update_sarsa, update_calql = self._resolve_critic_update_type(
            critic_warmup, critic_warmup_type, online_critic_update_type)

        # Use instance defaults if not provided
        if grpo_beta is None:
            grpo_beta = self.grpo_beta
        if cql_alpha is None:
            cql_alpha = self.cql_alpha
        if cql_temp is None:
            cql_temp = self.cql_temp

        if update_critic:
            for _ in range(utd_ratio):
                data_rng, rng = jax.random.split(rng)
                new_agent, critic_info = new_agent.update_critic(
                    batch, timer=timer, seed=data_rng, update_sarsa=update_sarsa, update_calql=update_calql,
                    calql_lower_bound=calql_lower_bound, cql_alpha=cql_alpha, cql_temp=cql_temp,
                    calql_random_actions=calql_random_actions, success_weight=success_weight)

        if update_edit_actor:
            for _ in range(edit_actor_utd_ratio):
                edit_actor_rng, rng = jax.random.split(rng)
                new_agent, actor_info = new_agent.update_edit_actor(
                    batch, seed=edit_actor_rng, bc_warmup=edit_actor_warmup)
            entropy = actor_info["entropy"]
            actor_info = append_substr_to_dict_keys(actor_info, "edit_actor")

        timer.tock("total_update_time")
        print(timer.get_total_times(reset=False))

        return new_agent, {**actor_info, **critic_info, **actor_update_info}

    def save_checkpoint(self, path: str, step: int):
        checkpoint = {
            'critic_params': self.critic.params,
            'target_critic_params': self.target_critic.params,
            'edit_actor_params': self.edit_actor.params,
            'temp_params': self.temp.params,
            'step': step,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"Saved checkpoint to {path}")


def load_checkpoint(path: str):
    checkpoint = pickle.load(open(path, 'rb'))
    critic_params = checkpoint['critic_params']
    target_critic_params = checkpoint.get('target_critic_params', None)
    edit_actor_params = checkpoint['edit_actor_params']
    temp_params = checkpoint['temp_params']

    if target_critic_params is None:
        target_critic_params = critic_params

    return critic_params, target_critic_params, edit_actor_params, temp_params
