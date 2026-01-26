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


def decay_mask_fn(params):
    flat_params = flax.traverse_util.flatten_dict(params)
    flat_mask = {path: path[-1] != "bias" for path in flat_params}
    return flax.core.FrozenDict(flax.traverse_util.unflatten_dict(flat_mask))


@partial(jax.jit, static_argnames=('critic_fn'))
def compute_q(critic_fn, critic_params, observations, actions):

    q_values = critic_fn({'params': critic_params},
                         observations, actions, False)
    # q_values = q_values.min(axis=0)
    q_values = q_values.mean(axis=0)
    return q_values


@partial(jax.jit, static_argnames=('critic_fn'))
def compute_q_all(critic_fn, critic_params, observations, actions):
    q_values = critic_fn({'params': critic_params},
                         observations, actions, False)
    # q_values = q_values
    return q_values

@partial(jax.jit, static_argnames=('critic_fn'))
def compute_v_all(critic_fn, critic_params, observations):
    v_values = critic_fn({'params': critic_params},
                         observations, False)
    return v_values

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
def _edit_actor_loss_and_grad(
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
    """Single jitted step for edit_actor: forward + loss + grad."""

    def loss_fn(actor_params):
        actions = edit_actor_apply_fn(
            {"params": actor_params}, vlm_output, batch_actions)
        edit_actions = actions.copy()
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

        # Only apply BC loss to successful trajectories #
        bc_loss = (jnp.square(batch_actions - actions) * success).mean()
        q_loss = -q.mean()
        edit_actor_loss = bc_loss * 1000.0 + (q_loss * (1.0 - bc_warmup))

        metrics = {
            "edit_q_mean": q.mean(),
            "edit_q_min": q.min(),
            "edit_q_max": q.max(),
            "base_q_mean": q_base.mean(),
            "base_q_min": q_base.min(),
            "base_q_max": q_base.max(),
            "edit_actor_loss": edit_actor_loss,
            "bc_loss": bc_loss,
            "q_loss": q_loss,
            "entropy": 0.0,
            "bc_warmup": bc_warmup,
        }

        edit_actions = edit_actions.reshape(-1, 10, 7)
        for i in range(7):
            metrics[f"edit_action_{i}_mean"] = edit_actions[:, :, i].mean()
            metrics[f"edit_action_{i}_std"] = edit_actions[:, :, i].std()
            metrics[f"edit_action_{i}_max"] = edit_actions[:, :, i].max()
            metrics[f"edit_action_{i}_min"] = edit_actions[:, :, i].min()

        batch_actions_reshaped = batch_actions.reshape(-1, 10, 7)
        for i in range(7):
            metrics[f"batch_action_{i}_mean"] = batch_actions_reshaped[:, :, i].mean(
            )
            metrics[f"batch_action_{i}_std"] = batch_actions_reshaped[:, :, i].std(
            )
            metrics[f"batch_action_{i}_max"] = batch_actions_reshaped[:, :, i].max(
            )
            metrics[f"batch_action_{i}_min"] = batch_actions_reshaped[:, :, i].min(
            )

        return edit_actor_loss, metrics

    (loss, metrics), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(actor_params)
    edit_actor = edit_actor.apply_gradients(grads=grads)

    return edit_actor, grads, metrics


@partial(jax.jit, static_argnames=("critic_apply_fn", "tau"))
def _critic_loss_and_grad(
    critic_params,
    target_critic_params,
    vlm_output,
    returns,
    key,
    critic_apply_fn,
    critic,
    tau,
    success_wt,
    success_mask,
):
    """Single jitted step for critic: forward + loss + grad."""

    def loss_fn(critic_params):
        values_all = critic_apply_fn(
            {"params": critic_params},
            vlm_output,
            False,
            rngs={"dropout": key},
        )
        values = values_all.mean(axis=0)

        critic_loss_unweighted = (values - returns) ** 2.0 # optax.losses.huber_loss(values, returns)
        # Weight using success_wt
        critic_loss_success = critic_loss_unweighted * success_mask * success_wt
        critic_loss_failure = critic_loss_unweighted * (1.0 - success_mask)
        critic_loss = (critic_loss_success + critic_loss_failure).mean()

        metrics = {
            "critic_loss": critic_loss,
            "critic_loss_success": critic_loss_success.mean(),
            "critic_loss_failure": critic_loss_failure.mean(),
            "value_mean": values.mean(),
            "value_std": values.std(),
            "value_max": values.max(),
            "value_min": values.min(),
            "returns_mean": returns.mean(),
            "returns_std": returns.std(),
            "returns_max": returns.max(),
            "returns_min": returns.min(),
        }

        # Add per-ensemble metrics
        num_heads = values_all.shape[0]
        for i in range(num_heads):
            vi = values_all[i]
            metrics[f"v{i+1}_mean"] = vi.mean()
            metrics[f"v{i+1}_std"] = vi.std()
            metrics[f"v{i+1}_max"] = vi.max()
            metrics[f"v{i+1}_min"] = vi.min()

        return critic_loss, metrics

    (loss, metrics), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(critic_params)
    critic = critic.apply_gradients(grads=grads)

    return critic, target_critic_params, grads, metrics


@partial(jax.jit, static_argnames=("critic_apply_fn", "target_critic_apply_fn", "tau"))
def _critic_td_loss_and_grad(
    critic_params,
    target_critic_params,
    vlm_output,
    rewards,
    key,
    critic_apply_fn,
    next_vlm_output,
    terminals,
    discount,
    critic,
    mc_returns,
    target_critic_apply_fn,
    tau,
    target_params,
):
    """Single jitted step for critic: forward + loss + grad."""
    def loss_fn(critic_params):
        values = critic_apply_fn(
            {"params": critic_params},
            vlm_output,
            False,
            rngs={"dropout": key},
        )

        # Use target critic for next state values
        values_next_state = target_critic_apply_fn(
            {"params": target_params},
            next_vlm_output,
            False,
        )
        # Take mean across ensemble for target
        values_next_state = values_next_state.mean(axis=0)

        masks = 1.0 - terminals
        target_v = rewards + discount * masks * values_next_state
        target_v = jax.lax.stop_gradient(target_v)

        # Compute loss for each ensemble member
        critic_loss = optax.losses.huber_loss(values, target_v).mean()
        critic_loss_mc = ((values - mc_returns) ** 2).mean()

        metrics = {
            "critic_loss": critic_loss,
            "critic_loss_mc": critic_loss_mc,
            "value_mean": values.mean(),
            "value_std": values.std(),
            "value_max": values.max(),
            "value_min": values.min(),
            "target_v_mean": target_v.mean(),
            "target_v_std": target_v.std(),
            "target_v_max": target_v.max(),
            "target_v_min": target_v.min(),
        }

        # Add per-ensemble metrics
        num_heads = values.shape[0]
        for i in range(num_heads):
            vi = values[i]
            metrics[f"v{i+1}_mean"] = vi.mean()
            metrics[f"v{i+1}_std"] = vi.std()
            metrics[f"v{i+1}_max"] = vi.max()
            metrics[f"v{i+1}_min"] = vi.min()

        return critic_loss, metrics

    (loss, metrics), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(critic_params)
    critic = critic.apply_gradients(grads=grads)

    # Update target critic with soft update
    target_critic_params = optax.incremental_update(
        critic.params, target_critic_params, tau
    )

    return critic, target_critic_params, grads, metrics


@partial(jax.jit, static_argnames="apply_fn")
def _sample_actions(rng, apply_fn, params, observations: np.ndarray, base_actions: np.ndarray) -> np.ndarray:
    key, rng = jax.random.split(rng)
    dist, means, log_stds = apply_fn({"params": params}, observations, base_actions)
    return dist.sample(seed=key), means, log_stds, rng


class PiResidualPPOCache(Agent):
    """
    Update temperature based computation based on SB3: https://stable-baselines3.readthedocs.io/en/v1.0/_modules/stable_baselines3/sac/sac.html?utm_source=chatgpt.com
    Original implementation seems to be buggy
    """
    actor: PiPolicy
    critic: TrainState
    target_critic: TrainState
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
    num_min_qs: Optional[int] = struct.field(
        pytree_node=False
    )  # See M in RedQ https://arxiv.org/abs/2101.05982
    backup_entropy: bool = struct.field(pytree_node=False)
    pi0_hidden_dims: int
    exploration_epsilon: float
    critic_success_wt: float

    @classmethod
    def create(
        cls,
        config: TrainConfig,
        seed: int,
        # observations: Data,
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
        num_qs: int = 1,
        num_min_qs: Optional[int] = None,
        critic_dropout_rate: Optional[float] = None,
        critic_weight_decay: Optional[float] = None,
        critic_layer_norm: bool = True,
        params_path: Optional[str] = None,
        target_entropy: Optional[float] = None,
        entropy_scale: float = 1.0,
        init_temperature: float = 1.0,
        backup_entropy: bool = True,
        use_pnorm: bool = False,
        # NOTE: Make it `True` so that outputs are valid
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
        edit_action_scale: float = 3.0,
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
        exploration_epsilon: float = 0.05,
        critic_success_wt: float = 1.0,
    ):
        # Assertions
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

        # batch_size = observations[batch_size_dict_key].shape[0]

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        # Init Pi0 model #
        actor = PiPolicy(rng=rng, config=config, is_target=False)

        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)
        
        # Load params if `params_path` is provided #
        critic_params = None
        target_critic_params = None
        edit_actor_params = None
        temp_params = None
        if params_path is not None:
            critic_params, target_critic_params, edit_actor_params, temp_params = load_checkpoint(params_path)

        # Init edit actor #
        # Edit actor for now will take in pi0 VLM output hidden states, predicted base actions, concatenate them and compute residual action
        # dummy_observations = jnp.ones((batch_size, pi0_hidden_dims)) # For initializing the edit actor
        # Inlcude state concatenation as well
        dummy_observations = jnp.ones(
            (batch_size, pi0_hidden_dims + state_dim))
        # For initializing the critic
        dummy_actions = jnp.ones((batch_size, action_dim))
        edit_actor_base_cls = partial(
            ResidualActor,
            action_dim=action_dim, hidden_dims=hidden_dims, num_residual_blocks=3
        )
        edit_actor_def = TanhNormal(edit_actor_base_cls, action_dim)
        edit_observations = jnp.concatenate(
            [dummy_observations, jnp.ones((batch_size, action_dim))], axis=1)

        if edit_actor_params is None:
            print("\n\n\nInitializing edit actor parameters from scratch...\n\n\n")
            edit_actor_params = edit_actor_def.init(
                actor_key, dummy_observations, dummy_actions)["params"]
        else:
            print(
                "\n\n\nInitializing edit actor parameters loaded from checkpoint...\n\n\n")

        edit_actor = TrainState.create(
            apply_fn=edit_actor_def.apply,
            params=edit_actor_params,
            tx=optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adam(learning_rate=actor_lr),
            )
        )

        # Init critic #
        critic_base_cls = partial(
            MLP,
            hidden_dims=hidden_dims,
            activate_final=True,
            dropout_rate=critic_dropout_rate,
            use_layer_norm=critic_layer_norm,
            use_pnorm=use_pnorm,
            activations=nn.swish,
        )
        # critic_cls = partial(StateAndStateActionValue, base_cls=critic_base_cls)
        critic_cls = partial(StateValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=num_qs)

        if critic_params is None:
            print("\n\n\nInitializing critic parameters from scratch...\n\n\n")
            critic_params = critic_def.init(
                critic_key, dummy_observations)["params"]

        if critic_weight_decay is not None:
            tx = optax.adamw(
                learning_rate=critic_lr,
                weight_decay=critic_weight_decay,
                mask=decay_mask_fn,
            )
        else:
            # tx = optax.adam(learning_rate=critic_lr)
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
        # Use target_critic_params if loaded, otherwise use critic_params
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

        temp_params = temp_def.init(temp_key)["params"]
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
            critic_success_wt=critic_success_wt,
        )
    
    def get_state_values(self, vlm_output: np.ndarray, *args, **kwargs):
        vlm_output = vlm_output.reshape(1, -1)
        return compute_v_all(self.critic.apply_fn, self.critic.params, vlm_output)[0]

    def sample_actions(self, _observations: Data, is_target=False, *args, **kwargs):
        '''
        For given state/observation, samles self.N actions from base policy; For first self.n_edit_samples actions, samples from edit_actor;
        Combine all (N + n_edit_samples) actions and compute Q-values; Return the action with the highest Q-value;
        '''
        out_dict = {}

        seed = kwargs.pop("seed", None)
        seed, rng = jax.random.split(seed)
        # timer = kwargs.pop("timer", None)
        use_deterministic_actions = kwargs.pop(
            "use_deterministic_actions", False)
        num_action_samples = kwargs.pop("num_action_samples", 1)
        
        # Repeat observations to sample `N` actions#
        # observations = repeat_observations(_observations, self.N, axis=0)
        observations = _observations

        # Do a forward pass to get VLM output #
        seed, rng = jax.random.split(seed)

        actions, vlm_output, processed_obs = self.actor.sample_actions_with_vlm_output(
            rng, observations)
        # (1, action_horizon, action_dim) # Unnormalized actions #
        diffusion_actions = actions.copy()
        # Returned actions are not normalized, normalize before passing to edit actor #
        # timer.tick("norm_actions_time")
        actions = self.actor.norm_actions(actions)

        # Take mean across tokens as representation from VLM #
        # (1, pi0_hidden_dims)
        vlm_output = jnp.mean(vlm_output[0][:, :512, :], axis=1)
        state = processed_obs['state'][0, :8][None, :8]  # State dimension
        # (1, pi0_hidden_dims + state_dim)
        vlm_output = jnp.concatenate([vlm_output, state], axis=1)

        # NOTE: In all the computation in this class, self.action_dim = action_horizon * action_dim, so keep that semantic in mind #
        # This is done so that without any code modification, edit_actor outputs an action chunk #
        seed, rng = jax.random.split(seed)
        actions = actions.reshape(1, self.action_dim)
        r_observations = vlm_output
        
        actions, means, log_stds, rng = _sample_actions(
            rng, self.edit_actor.apply_fn, self.edit_actor.params, r_observations, actions
        )

        if use_deterministic_actions:
            actions = means

        actions = jnp.clip(actions, -1.0, 1.0)
        final_action = actions.reshape(
            self.action_horizon, self.action_dim // self.action_horizon
        )

        # Unnormalize actions before returning #
        final_action = self.actor.unnorm_actions(final_action)

        # Compute state values for caching and advantages calculations in PPO #
        v_values = compute_v_all(self.critic.apply_fn, self.critic.params, vlm_output)

        rng, _ = jax.random.split(rng, 2)
        out_dict = {
            "actions": final_action,  # (action_horizon, action_dim)
            "vlm_output": vlm_output[0],  # (pi0_hidden_dims,)
            "state_values": v_values[0],  # (1,)
            "diffusion_actions": diffusion_actions
        }

        return out_dict

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
        v_values = compute_v_all(self.critic.apply_fn, self.critic.params, vlm_output)

        rng, _ = jax.random.split(rng, 2)
        out_dict = {
            "actions": action.squeeze(),  # (action_horizon, action_dim)
            "vlm_output": vlm_output[0],  # (pi0_hidden_dims,)
            "diffusion_actions": diffusion_actions,
            "state_values": v_values[0],  # (1,)
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

    def update_edit_actor(self, batch: Batch, *args, **kwargs) -> Tuple[Agent, Dict[str, float]]:
        seed = kwargs.pop("seed", None)
        bc_warmup = kwargs.pop("bc_warmup", 0.0)

        assert seed is not None, "seed must be provided"
        rng = seed

        base_actions = batch['diffusion_actions']
        base_actions = self.actor.norm_actions(base_actions)
        vlm_output = batch['vlm_output']
        base_actions = base_actions.reshape(-1, self.action_dim)

        # Get vlm output for observations #
        dropout_rng, rng = jax.random.split(rng)
        rng1, rng = jax.random.split(rng)
        rng2, rng = jax.random.split(rng)
        # Use JITted version #
        edit_actor, grads, actor_info = _edit_actor_loss_and_grad(
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
            bc_warmup,
        )

        # q_loss_rng, rng = jax.random.split(rng)
        # q_loss_grads, q_loss_metrics = q_loss(
        #     self.critic.params,
        #     vlm_output,
        #     base_actions,
        #     batch["mc_returns"],
        #     q_loss_rng,
        #     self.critic.apply_fn,
        #     self.critic,
        # )

        actor_info["edit_actor_grad_norm"] = optax.global_norm(grads)
        # edit_actor = self.edit_actor.apply_gradients(grads=grads)
        actor_info["edit_actor_param_norm"] = optax.global_norm(
            edit_actor.params)
        actor_info["target_entropy"] = self.target_entropy

        # actor_info.update(q_loss_metrics)
        # actor_info["q_loss_grad_norm"] = optax.global_norm(q_loss_grads)

        return self.replace(edit_actor=edit_actor, rng=rng), actor_info

    def update_critic(self, batch, *args, **kwargs) -> Tuple[TrainState, Dict[str, float]]:
        seed = kwargs.pop("seed", None)
        timer = kwargs.pop("timer", None)
        critic_warmup = kwargs.pop("critic_warmup", False)
        assert seed is not None, "seed must be provided"
        rng = seed

        # Sample next_actions by sampling from current edit policy #
        next_vlm_output = batch['next_vlm_output']
        current_vlm_output = batch['vlm_output']

        # Use JITted version #
        timer.tick("critic_loss_and_grad_time")
        rng1, rng = jax.random.split(rng)

        if critic_warmup:
            # Subsample target ensemble
            subsample_rng, rng = jax.random.split(rng)
            target_params = subsample_ensemble(
                subsample_rng, self.target_critic.params, self.num_min_qs, self.num_qs
            )

            critic, target_critic_params, grads, info = _critic_td_loss_and_grad(
                self.critic.params,
                self.target_critic.params,
                current_vlm_output,
                batch["rewards"],
                rng1,
                self.critic.apply_fn,
                next_vlm_output,
                batch["terminals"],
                self.discount,
                self.critic,
                batch["mc_returns"],
                self.target_critic.apply_fn,
                self.tau,
                target_params,
            )

            target_critic = self.target_critic.replace(params=target_critic_params)

        else:
            current_returns = batch['returns_tf_from_adv']
            critic, target_critic_params, grads, info = _critic_loss_and_grad(
                self.critic.params,
                self.target_critic.params,
                current_vlm_output,
                current_returns,
                rng1,
                self.critic.apply_fn,
                self.critic,
                self.tau,
                self.critic_success_wt,
                batch["success"]
            )

            target_critic = self.target_critic.replace(params=target_critic_params)

        timer.tock("critic_loss_and_grad_time")

        timer.tick("apply_gradients_time")
        info["critic_grad_norm"] = optax.global_norm(grads)
        timer.tock("apply_gradients_time")

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

    def update(
                self, 
                _observations: Data, 
                utd_ratio: int, 
                update_critic: bool = True, 
                update_edit_actor: bool = True, 
                critic_warmup: bool = False,
                edit_actor_warmup: bool = False,
                *args, **kwargs
            ):
        timer = kwargs.pop("timer", None)
        new_agent = self
        timer.tick("total_update_time")
        timer.tick("preprocess_time")

        # Preprocess batch at once to save computation #
        batch = self.preproess_batch(_observations.copy())
        state = batch['observations']['proprio'][:, :8].copy()
        next_state = batch['next_observations']['proprio'][:, :8].copy()

        # Filter batch to only keep relevant information #
        relevant_keys = [
            "actions", "rewards", "masks", "mc_returns",
            "terminals", "truncates", "vlm_output", "next_vlm_output", "diffusion_actions", "next_diffusion_actions",
            "next_actions", "success", "returns_tf_from_adv",
        ]
        batch = {k: v for k, v in batch.items() if k in relevant_keys}
        # Normalization of state has already happened in image_replay_buffer_pi.py #
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

        if update_critic:        
            for _ in range(utd_ratio):
                data_rng, rng = jax.random.split(rng)
                new_agent, critic_info = new_agent.update_critic(
                    batch, timer=timer, seed=data_rng, critic_warmup=critic_warmup)

        if update_edit_actor:
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
        logging.info(f"Saved checkpoint to {path}")

def load_checkpoint(path: str):
    checkpoint = pickle.load(open(path, 'rb'))
    critic_params = checkpoint['critic_params']
    target_critic_params = checkpoint.get('target_critic_params', None)
    edit_actor_params = checkpoint['edit_actor_params']
    temp_params = checkpoint['temp_params']

    # If target_critic_params not in checkpoint (old checkpoint), use critic_params
    if target_critic_params is None:
        target_critic_params = critic_params

    return critic_params, target_critic_params, edit_actor_params, temp_params

