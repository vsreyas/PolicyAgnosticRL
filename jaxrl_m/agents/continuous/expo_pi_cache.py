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

from jaxrl_m.utils.expo_utils import (
    TanhNormal,
    StateActionValue,
    Ensemble,
    subsample_ensemble,
    Agent,
    Temperature,
    MLP,
    repeat_observations,
    repeat_observations_batched,
    repeat_observations_openpi,
    append_substr_to_dict_keys,
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

    q_values = critic_fn({'params': critic_params}, observations, actions, False)
    # q_values = q_values.min(axis=0)
    q_values = q_values.mean(axis=0)
    return q_values

@partial(jax.jit, static_argnames=('critic_fn'))
def compute_q_all(critic_fn, critic_params, observations, actions):
    q_values = critic_fn({'params': critic_params}, observations, actions, False)
    # q_values = q_values
    return q_values

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
):
    """Single jitted step for edit_actor: forward + loss + grad."""

    def loss_fn(actor_params):
        # [B, D_vlm + D_action]
        edit_observations = jnp.concatenate([vlm_output, batch_actions], axis=1)

        dist = edit_actor_apply_fn(
            {"params": actor_params},
            edit_observations,
            training=False,
            rngs={"dropout": dropout_key},
        )
        actions_sampled = dist.sample(seed=key)
        log_probs_orig = dist.log_prob(actions_sampled)
        edited_actions = actions_sampled * edit_action_scale
        edit_actions = edited_actions.copy()


        log_probs = log_probs_orig - actions_sampled.shape[-1] * jnp.log(edit_action_scale)

        actions = edited_actions + batch_actions

        qs = critic_apply_fn(
            {"params": critic_params},
            vlm_output,
            actions,
            False,
            rngs={"dropout": key2},
        )
        q = qs.mean(axis=0)

        temperature = temp_apply_fn({"params": temp_params})
        edit_actor_loss = (entropy_scale * log_probs * temperature - q).mean()
        # edit_actor_loss = -q.mean()

        metrics = {
            "edit_q": q.mean(),
            "edit_actor_loss": edit_actor_loss,
            "entropy": -log_probs.mean(),
            # "log_probs": log_probs.mean(),
        }

        edit_actions = edit_actions.reshape(-1, 10, 7)
        for i in range(7):
            metrics[f"edit_action_{i}_mean"] = edit_actions[:, :, i].mean()
            metrics[f"edit_action_{i}_std"] = edit_actions[:, :, i].std()
            metrics[f"edit_action_{i}_max"] = edit_actions[:, :, i].max()
            metrics[f"edit_action_{i}_min"] = edit_actions[:, :, i].min()
        
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


@partial(jax.jit, static_argnames=("critic_apply_fn", "edit_actor_apply_fn", "target_critic_apply_fn", "q_clip_low", "q_clip_high", "tau"))
def _critic_loss_and_grad(
    critic_params,
    target_critic_params,
    vlm_output,
    actions,
    # target_q,
    mc_target,
    key,
    critic_apply_fn,
    # #
    target_params, # Subsampled target critic parameters
    next_vlm_output,
    next_base_actions,
    sample_key,
    edit_actor_apply_fn,
    edit_actor_params,
    edit_action_scale,
    target_critic_apply_fn,
    terminals,
    rewards,
    discount,
    # #
    critic,
    tau,
    # #
    q_clip_low,
    q_clip_high,
):
    """Single jitted step for critic: forward + loss + grad."""

    r_observations = jnp.concatenate([next_vlm_output, next_base_actions], axis=1) # (1, pi0_hidden_dims + action_horizon * action_dim)
    r_samples, _ =  _sample_actions(sample_key, edit_actor_apply_fn, edit_actor_params, r_observations) # (n_edit_samples, action_horizon * action_dim)
    next_actions = r_samples * edit_action_scale + next_base_actions
    next_qs = compute_q(target_critic_apply_fn, target_params, next_vlm_output, next_actions) # (batch_size, )
    masks = 1.0 - terminals
    target_q = rewards + discount * masks * next_qs # (batch_size, )
    target_q = jax.lax.stop_gradient(target_q)

    def loss_fn(critic_params):
        qs = critic_apply_fn(
            {"params": critic_params},
            vlm_output,
            actions,
            False,
            rngs={"dropout": key},
        )
        # jax.debug.breakpoint()
        # jax.debug.print(f"qs: {qs.shape}")
        # jax.debug.print("target_q: {target_q.shape}")
        # clipped_qs = jnp.clip(qs, q_clip_low, q_clip_high)
        # clipped_target_q = jnp.clip(target_q, q_clip_low, q_clip_high)
        
        # critic_loss_td = ((clipped_qs - clipped_target_q) ** 2).mean()
        # critic_loss_mc = ((clipped_qs - mc_target) ** 2).mean()

        # critic_loss = (critic_loss_td + 0.2 * critic_loss_mc)
        # critic_loss = critic_loss_td
        critic_loss_td = ((qs - target_q) ** 2).mean()
        critic_loss_mc = ((qs - mc_target) ** 2).mean()
        critic_loss = critic_loss_td # + 0.2 * critic_loss_mc


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

        num_heads = qs.shape[0]  # static at compile time
        for i in range(num_heads):
            qi = qs[i]
            metrics[f"q{i+1}_mean"] = qi.mean()
            metrics[f"q{i+1}_std"]  = qi.std()
            metrics[f"q{i+1}_max"]  = qi.max()
            metrics[f"q{i+1}_min"]  = qi.min()
        
        return critic_loss, metrics

    (loss, metrics), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(critic_params)
    critic = critic.apply_gradients(grads=grads)
    target_critic_params = optax.incremental_update(
        critic.params, target_critic_params, tau
    )
    return critic, target_critic_params, grads, metrics

@partial(jax.jit, static_argnames="temp_apply_fn")
def _temperature_loss_and_grad(temp, temp_params, entropy, target_entropy, temp_apply_fn):
    # jax.debug.print(
    #     "entropy={e}, target={t}, c={c}, log_temp={lt}, temp={a}",
    #     e=entropy,
    #     t=target_entropy,
    #     c=(entropy - target_entropy).mean(),
    #     lt=jnp.log(temp_apply_fn({"params": temp_params})),
    #     a=temp_apply_fn({"params": temp_params}),
    # )
    def loss_fn(temp_params):
        temperature = temp_apply_fn({"params": temp_params})
        log_temp = jnp.log(temperature)
        temp_loss = temperature * jax.lax.stop_gradient((entropy - target_entropy).mean())
        return temp_loss, {"temp_loss": temp_loss, "log_temp": log_temp, "temperature": temperature}

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(temp_params)
    temp = temp.apply_gradients(grads=grads)

    return temp, grads, metrics

@partial(jax.jit, static_argnames="apply_fn")
def _sample_actions(rng, apply_fn, params, observations: np.ndarray) -> np.ndarray:
    key, rng = jax.random.split(rng)
    dist = apply_fn({"params": params}, observations)
    return dist.sample(seed=key), rng


class ExpoPiLearnerCache(Agent):
    """
    Update temperature based computation based on SB3: https://stable-baselines3.readthedocs.io/en/v1.0/_modules/stable_baselines3/sac/sac.html?utm_source=chatgpt.com
    Original implementation seems to be buggy
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
    num_min_qs: Optional[int] = struct.field(
        pytree_node=False
    )  # See M in RedQ https://arxiv.org/abs/2101.05982
    backup_entropy: bool = struct.field(pytree_node=False)
    q_clip_low: float
    q_clip_high: float
    pi0_hidden_dims: int

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
        rng : PRNGKey | None = None,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        num_qs: int = 10,
        num_min_qs: Optional[int] = 2,
        critic_dropout_rate: Optional[float] = None,
        critic_weight_decay: Optional[float] = None,
        critic_layer_norm: bool = True,
        critic_params: Optional[at.Params] = None,
        edit_actor_params: Optional[at.Params] = None,
        target_entropy: Optional[float] = None,
        entropy_scale: float = 1.0, 
        init_temperature: float = 1.0,
        backup_entropy: bool = True,
        use_pnorm: bool = False,
        adjust_target_entropy: bool = True, # NOTE: Make it `True` so that outputs are valid 
        use_critic_resnet: bool = False,
        time_dim: int = 128,
        actor_drop: Optional[float] = None, 
        d_actor_drop: Optional[float] = None, 
        T: int = 10, 
        N: int = 4,
        batch_split: int = 1, 
        M: int = 0,
        n_edit_samples: int = 4, 
        edit_action_scale: float = 0.5,
        actor_layer_norm: bool = True,
        clip_sampler: bool = True,
        decay_steps: Optional[int] = int(3e6),
        actor_tau: float = 0.003,
        actor_dropout_rate: Optional[float] = None,
        actor_num_blocks: int = 3,
        ddpm_temperature: float = 1.0,
        beta_schedule: str = 'vp',
        batch_size_dict_key: str = 'actions',
        q_clip_low: Optional[float] = -10000.0,
        q_clip_high: Optional[float] = 10000.0,
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
                target_entropy = -action_dim / 2 + action_dim * jnp.log(edit_action_scale)

        # batch_size = observations[batch_size_dict_key].shape[0]


        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        # Init Pi0 model #
        # Initialize target_actor as a PiPolicy object with is_target=True; This ensures optimizer states are not created for target_actor #
        # target_actor = PiPolicy(rng=rng, config=config, is_target=True)
        actor = PiPolicy(rng=rng, config=config, is_target=False)
        target_actor = actor
        
        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        # Init edit actor #
        # Edit actor for now will take in pi0 VLM output hidden states, predicted base actions, concatenate them and compute residual action
        # dummy_observations = jnp.ones((batch_size, pi0_hidden_dims)) # For initializing the edit actor
        dummy_observations = jnp.ones((batch_size, pi0_hidden_dims + state_dim)) # Inlcude state concatenation as well
        dummy_actions = jnp.ones((batch_size, action_dim)) # For initializing the critic
        edit_actor_base_cls = partial(
            MLP, hidden_dims=hidden_dims, dropout_rate=actor_drop, activate_final=True, use_pnorm=use_pnorm
        )
        edit_actor_def = TanhNormal(edit_actor_base_cls, action_dim)
        edit_observations = jnp.concatenate([dummy_observations, jnp.ones((batch_size, action_dim))], axis=1)

        if edit_actor_params is None:
            print("\n\n\nInitializing edit actor parameters from scratch...\n\n\n")
            edit_actor_params = edit_actor_def.init(actor_key, edit_observations)["params"]
        else:
            print("\n\n\nInitializing edit actor parameters loaded from checkpoint...\n\n\n")

        edit_actor = TrainState.create(
            apply_fn=edit_actor_def.apply, 
            params=edit_actor_params, 
            # tx=optax.adam(learning_rate=actor_lr),
            tx = optax.chain(
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
            # activations=nn.swish,
            activations=nn.relu,
        )
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=num_qs)

        if critic_params is None:
            print("\n\n\nInitializing critic parameters loaded from checkpoint...\n\n\n")
            critic_params = critic_def.init(critic_key, dummy_observations, dummy_actions)["params"]
        
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
        target_critic_def = Ensemble(critic_cls, num=num_min_qs or num_qs)
        target_critic = TrainState.create(
            apply_fn=target_critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx = optax.chain(
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
            # target_actor_params=target_actor_params,
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
            q_clip_low=q_clip_low,
            q_clip_high=q_clip_high,
            pi0_hidden_dims=pi0_hidden_dims,
        )

    def sample_batch_actions(self, obs, is_target=False, *args, **kwargs):
        # Optional RNG + timer from kwargs (similar to sample_actions)
        seed = kwargs.pop("seed", None)
        timer = kwargs.pop("timer", None)
        return_first_action = kwargs.pop("return_first_action", False)
        output_only_base_actions = kwargs.pop("output_only_base_actions", False)
        output_all_sampled_actions = kwargs.pop("output_all_sampled_actions", False)
        info_dict = {}

        batch_size = obs.state.shape[0]

        # if seed is None:
        #     rng = self.rng
        # else:
        assert seed is not None, "seed must be provided"
        seed, rng = jax.random.split(seed)
        
        # # Sample `N` actions from base policy #
        # # Repeat observations to sample `N` actions
        # timer.tick("repeat_observations_openpi_time")
        # observations_repeated = repeat_observations_openpi(obs, self.N, axis=0) # (batch_size * N, ...)
        # timer.tock("repeat_observations_openpi_time")

        # # Get actions and vlm output for all observations above # (batch_size * N,)
        # actor_to_sample = self.actor
        # if is_target:
        #     actor_to_sample = self.target_actor
        # # breakpoint()
        # timer.tick("sample_actions_with_vlm_output_time")
        # pi0_actions, vlm_output = actor_to_sample.sample_actions_with_vlm_output(rng, observations_repeated, timer=timer) # (batch_size * N, action_horizon, action_dim)
        # timer.tock("sample_actions_with_vlm_output_time")
        # # breakpoint()
        # vlm_output = jnp.mean(vlm_output[0][:, :512, :], axis=1) # Take mean representation across tokens, (batch_size * N, pi0_hidden_dims) #
        # actions = pi0_actions
        
        # state = observations_repeated.state[:, :8]
        # vlm_output = jnp.concatenate([vlm_output, state], axis=1) # (batch_size * N, pi0_hidden_dims + state_dim)
        


        if self.N > 1:
            key, rng = jax.random.split(rng)
            target_params = subsample_ensemble(
                key, self.target_critic.params, self.num_min_qs, self.num_qs
            )

            # Slice out vlm_output to keep only `batch_size` items #
            vlm_output_sliced = vlm_output[:batch_size, :]

            timer.tick("sample_actions_with_vlm_output_time")
            if self.n_edit_samples > 0:
                key, rng = jax.random.split(rng, 2)

                r_observations = repeat_observations_batched(vlm_output_sliced, self.n_edit_samples, axis=0) # (batch_size * n_edit_samples, pi0_hidden_dims)
                d_actions = pi0_actions.copy().reshape(batch_size, self.N, self.action_horizon, self.action_dim // self.action_horizon)
                d_actions = d_actions[:, :self.n_edit_samples, :, :].reshape(-1, self.action_dim) # (batch_size * n_edit_samples, actions_horizon * action_dim)
                # d_actions = d_actions.reshape(self.n_edit_samples, -1)
                r_observations = jnp.concatenate([r_observations, d_actions], axis=1)
                r_samples, rng = _sample_actions(
                    key, self.edit_actor.apply_fn, self.edit_actor.params, r_observations
                )
                r_samples = r_samples.reshape(-1, self.action_horizon, self.action_dim // self.action_horizon)

                # Concatenate actions #
                actions = actions.reshape(batch_size, self.N, self.action_horizon, self.action_dim // self.action_horizon)
                r_samples = r_samples.reshape(batch_size, self.n_edit_samples, self.action_horizon, self.action_dim // self.action_horizon)
                actions = jnp.concatenate([actions, r_samples], axis=1) # (batch_size, N + n_edit_samples, action_horizon, action_dim)
                observations_repeated = repeat_observations_batched(vlm_output_sliced, self.N + self.n_edit_samples, axis=0) # (batch_size * (N + n_edit_samples), pi0_hidden_dims)
            else:
                observations_repeated = repeat_observations_batched(vlm_output_sliced, self.N, axis=0) # (batch_size * N, pi0_hidden_dims)
            timer.tock("sample_actions_with_vlm_output_time")

            # Log action statistics #
            actions_base = actions[:, :self.N, :, :]
            actions_edit = actions[:, self.N:, :, :]
            for __idx in range(self.action_dim // self.action_horizon):
                info_dict[f"actions_base{__idx}/mean"] = actions_base[:, :, :, __idx].mean()
                info_dict[f"actions_base{__idx}/min"] = actions_base[:, :, :, __idx].min()
                info_dict[f"actions_base{__idx}/max"] = actions_base[:, :, :, __idx].max()
                info_dict[f"actions_edit{__idx}/mean"] = actions_edit[:, :, :, __idx].mean()
                info_dict[f"actions_edit{__idx}/min"] = actions_edit[:, :, :, __idx].min()
                info_dict[f"actions_edit{__idx}/max"] = actions_edit[:, :, :, __idx].max()

            actions = actions.reshape(-1, self.action_dim) # (batch_size * (N + n_edit_samples), action_horizon * action_dim)

            timer.tick("compute_q_time")
            qs = compute_q(self.target_critic.apply_fn, target_params, observations_repeated, actions)
            timer.tock("compute_q_time")
            qs = qs.reshape(batch_size, self.N + self.n_edit_samples) # (batch_size, N + n_edit_samples)

            # Get q values for logging #
            qs_base = qs[:, :self.N]
            qs_edit = qs[:, self.N:]
            info_dict["qs_base/mean"] = qs_base.mean()
            info_dict["qs_base/std"] = qs_base.std()
            info_dict["qs_base/max"] = qs_base.max()
            info_dict["qs_base/min"] = qs_base.min()
            info_dict["qs_edit/mean"] = qs_edit.mean()
            info_dict["qs_edit/std"] = qs_edit.std()
            info_dict["qs_edit/max"] = qs_edit.max()
            info_dict["qs_edit/min"] = qs_edit.min()

            idx = jnp.argmax(qs, axis=1) # (batch_size, )
            batch_idx = jnp.arange(batch_size)
            actions = actions.reshape(batch_size, self.N + self.n_edit_samples, self.action_horizon, self.action_dim // self.action_horizon)

            if return_first_action:
                action = actions[batch_idx, idx, 0, :] # (batch_size, action_dim)
            else:
                action = actions[batch_idx, idx, :, :] # (batch_size, action_horizon, action_dim)
        else:
            raise ValueError(f"N must be greater than 1, got {self.N}")

        rng, _ = jax.random.split(rng, 2)
        # We *don’t* mutate self.rng here; caller controls RNG via `seed`
        return action, vlm_output_sliced[:, :self.pi0_hidden_dims], info_dict
    
    

    def sample_actions(self, _observations: Data, is_target=False,*args, **kwargs):
        '''
        For given state/observation, samles self.N actions from base policy; For first self.n_edit_samples actions, samples from edit_actor;
        Combine all (N + n_edit_samples) actions and compute Q-values; Return the action with the highest Q-value;
        '''
        out_dict = {}

        seed = kwargs.pop("seed", None)
        seed, rng = jax.random.split(seed)
        output_action_chunk = kwargs.pop("output_action_chunk", True)
        # Repeat observations to sample `N` actions#
        # observations = repeat_observations(_observations, self.N, axis=0)
        observations = _observations

        # breakpoint()
        # if not is_target:
        #     actions = self.actor.sample_actions(observations, seed=rng) # (N, action_horizon, action_dim)
        # else:
        #     # actions = self.actor.sample_actions(observations, seed=rng, params=self.target_actor.train_state.params) # (N, action_horizon, action_dim)
        #     actions = self.target_actor.sample_actions(observations, seed=rng) # (N, action_horizon, action_dim)

        # actions = self.actor.sample_actions(observations, seed=rng)
        # diffusion_actions = actions.copy() # (1, action_horizon, action_dim)

        # Do a forward pass to get VLM output #
        seed, rng = jax.random.split(seed)
        # if not is_target:
        #     vlm_output, _, processed_obs = self.actor.get_vlm_output(rng, _observations, processed_obs=False, infer=True, return_processed_obs=True)
        # else:
        #     vlm_output, _, processed_obs = self.target_actor.get_vlm_output(rng, _observations, processed_obs=False, infer=True, return_processed_obs=True)

        # _model.Observation.from_dict(observations)
        actions, vlm_output, processed_obs = self.actor.sample_actions_with_vlm_output(rng, observations)
        diffusion_actions = actions.copy() # (1, action_horizon, action_dim)
        
        
        # Take mean across tokens as representation from VLM #
        vlm_output = jnp.mean(vlm_output[0][:, :512, :], axis=1) # (1, pi0_hidden_dims)
        state = processed_obs['state'][0, :8][None, :8] # State dimension
        vlm_output = jnp.concatenate([vlm_output, state], axis=1) # (1, pi0_hidden_dims + state_dim)

        # NOTE: In all the computation in this class, self.action_dim = action_horizon * action_dim, so keep that semantic in mind #
        # This is done so that without any code modification, edit_actor outputs an action chunk #
        # TODO: Think of a better design #
        # if self.N > 1:
        seed, rng = jax.random.split(seed)
        # target_params = subsample_ensemble(
        #     key, self.target_critic.params, self.num_min_qs, self.num_qs
        # )
        actions = actions.reshape(1, self.action_dim)
        r_observations = jnp.concatenate([vlm_output, actions], axis=1) # (1, pi0_hidden_dims + action_horizon * action_dim)
        r_samples, rng =  _sample_actions(seed, self.edit_actor.apply_fn, self.edit_actor.params, r_observations) # (n_edit_samples, action_horizon * action_dim)
        r_samples = r_samples * self.edit_action_scale + actions
        actions = r_samples
        action = actions.reshape(1, self.action_horizon, self.action_dim // self.action_horizon)

        rng, _ = jax.random.split(rng, 2)
        out_dict = {
            "actions": action.squeeze(), # (action_horizon, action_dim)
            "vlm_output": vlm_output[0], # (pi0_hidden_dims,)
            "diffusion_actions": diffusion_actions # (N, action_horizon, action_dim)
        }
        
        return out_dict
    
    def sample_base_actions(self, _observations: Data, is_target=False,*args, **kwargs):
        '''
        For given state/observation, samles self.N actions from base policy; For first self.n_edit_samples actions, samples from edit_actor;
        Combine all (N + n_edit_samples) actions and compute Q-values; Return the action with the highest Q-value;
        '''
        out_dict = {}

        seed = kwargs.pop("seed", None)
        seed, rng = jax.random.split(seed)
        output_action_chunk = kwargs.pop("output_action_chunk", True)
        # Repeat observations to sample `N` actions#
        # observations = repeat_observations(_observations, self.N, axis=0)
        observations = _observations

        seed, rng = jax.random.split(seed)
        actions, vlm_output, processed_obs = self.actor.sample_actions_with_vlm_output(rng, observations)
        diffusion_actions = actions.copy() # (1, action_horizon, action_dim)
        
        
        # Take mean across tokens as representation from VLM #
        vlm_output = jnp.mean(vlm_output[0][:, :512, :], axis=1) # (1, pi0_hidden_dims)
        state = processed_obs['state'][0, :8][None, :8] # State dimension
        vlm_output = jnp.concatenate([vlm_output, state], axis=1) # (1, pi0_hidden_dims + state_dim)

        action = diffusion_actions

        rng, _ = jax.random.split(rng, 2)
        out_dict = {
            "actions": action.squeeze(), # (action_horizon, action_dim)
            "vlm_output": vlm_output[0], # (pi0_hidden_dims,)
            "diffusion_actions": diffusion_actions # (N, action_horizon, action_dim)
        }
        
        return out_dict
    
    def get_vlm_output(self, _observations: Data, is_target=False,*args, **kwargs):
        '''
        For given state/observation, samles self.N actions from base policy; For first self.n_edit_samples actions, samples from edit_actor;
        Combine all (N + n_edit_samples) actions and compute Q-values; Return the action with the highest Q-value;
        '''
        seed = kwargs.pop("seed", None)
        seed, rng = jax.random.split(seed)
        # Do a forward pass to get VLM output #
        seed, rng = jax.random.split(rng)
        if not is_target:
            vlm_output, _, processed_obs = self.actor.get_vlm_output(rng, _observations, processed_obs=False, infer=True, return_processed_obs=True)
        else:
            vlm_output, _, processed_obs = self.target_actor.get_vlm_output(rng, _observations, processed_obs=False, infer=True, return_processed_obs=True)

        # Take mean across tokens as representation from VLM #
        vlm_output = jnp.mean(vlm_output[0][:, :512, :], axis=1) # (N, pi0_hidden_dims)
        state = processed_obs['state'][0, :8][None, :8] # State dimension
        vlm_output = jnp.concatenate([vlm_output, state], axis=1) # (1, pi0_hidden_dims + state_dim)
        
        return vlm_output[0]


    def update_actor(self, batch: Batch, *args, **kwargs) -> Tuple[Agent, Dict[str, float]]:
        seed = kwargs.pop("seed", None)
        timer = kwargs.pop("timer", None)
        infer = kwargs.pop("infer", False)
        obs_key = kwargs.pop("obs_key", "observations")

        # if seed is None:
        #     rng = self.rng
        # else:
        assert seed is not None, "seed must be provided"
        seed, rng = jax.random.split(seed)

        actor_update_info = self.actor.update(batch)
        # breakpoint()

        # Update target actor train_state with incremental parameter update #
        # target_score_params = optax.incremental_update(
        #     self.actor.train_state.params, self.target_actor.train_state.params, self.actor_tau
        # )
        # self.target_actor.train_state = self.target_actor.train_state.replace(params=target_score_params)

        new_agent = self.replace(actor=self.actor, target_actor=self.target_actor, rng=rng)
        # new_agent = self.replace(actor=self.actor, rng=rng)
        return new_agent, actor_update_info
    
    def update_edit_actor(self, batch: Batch, *args, **kwargs) -> Tuple[Agent, Dict[str, float]]:
        seed = kwargs.pop("seed", None)

        # if seed is None:
        #     rng = self.rng
        # else:
        assert seed is not None, "seed must be provided"
        seed, rng = jax.random.split(seed)

        key, rng = jax.random.split(rng)
        key2, rng = jax.random.split(rng)
        dropout_key, rng = jax.random.split(rng)

        # batch['actions'] = batch['actions'].reshape(-1, self.action_dim)
        base_actions = batch['diffusion_actions']
        batch_size = base_actions.shape[0]
        key, rng = jax.random.split(key)
        # action_indices = jax.random.randint(rng, shape=(batch_size,), minval=0, maxval=self.N)
        # base_actions = base_actions[jnp.arange(batch_size), action_indices, :, :]
        base_actions = base_actions.reshape(-1, self.action_dim)
        
        # Get vlm output for observations #
        seed, rng = jax.random.split(rng)
        vlm_output = batch['vlm_output']
        # Use JITted version #
        edit_actor, grads, actor_info = _edit_actor_loss_and_grad(
            self.edit_actor,
            self.edit_actor.params,
            self.critic.params,
            self.temp.params,
            vlm_output,
            base_actions,
            dropout_key,
            key,
            key2,
            self.entropy_scale,
            self.edit_action_scale,
            self.edit_actor.apply_fn,
            self.critic.apply_fn,
            self.temp.apply_fn,
        )

        actor_info["edit_actor_grad_norm"] = optax.global_norm(grads)
        # edit_actor = self.edit_actor.apply_gradients(grads=grads)
        actor_info["edit_actor_param_norm"] = optax.global_norm(edit_actor.params)
        actor_info["target_entropy"] = self.target_entropy

        return self.replace(edit_actor=edit_actor, rng=rng), actor_info
        
    def update_temperature(self, entropy: float) -> Tuple[Agent, Dict[str, float]]:
        # temp, grads, temp_info = temperature_loss_fn(self.temp.params, entropy, self.target_entropy, self.temp.apply_fn)
        temp, grads, temp_info = _temperature_loss_and_grad(self.temp, self.temp.params, entropy, self.target_entropy, self.temp.apply_fn)
        # temp = self.temp.apply_gradients(grads=grads)

        temp_info['temp_grad_norm'] = optax.global_norm(grads)
        temp_info['temp_param_norm'] = optax.global_norm(temp.params)

        return self.replace(temp=temp), temp_info

    def update_critic(self, batch, *args, **kwargs) -> Tuple[TrainState, Dict[str, float]]:
        seed = kwargs.pop("seed", None)
        timer = kwargs.pop("timer", None)
        current_state = batch['state']
        next_state = batch['next_state']

        # if seed is None:
        #     rng = self.rng
        # else:
        assert seed is not None, "seed must be provided"
        key, rng = jax.random.split(seed)
        
        batch_size = batch['actions'].shape[0]
        # next_actions = batch['next_actions'] # (batch_size, action_horizon, action_dim)

        # Sample next_actions by sampling from current edit policy #
        next_base_actions = batch['next_diffusion_actions']
        next_vlm_output = batch['next_vlm_output'] # (batch_size, pi0_hidden_dims)
        # next_base_actions = next_base_actions.reshape(-1, self.action_dim)
        # r_observations = jnp.concatenate([next_vlm_output, next_base_actions], axis=1) # (1, pi0_hidden_dims + action_horizon * action_dim)
        # r_samples, _ =  _sample_actions(rng, self.edit_actor.apply_fn, self.edit_actor.params, r_observations) # (n_edit_samples, action_horizon * action_dim)
        # next_actions = r_samples * self.edit_action_scale + next_base_actions

        # next_actions = next_actions.reshape(1, self.action_horizon, self.action_dim // self.action_horizon)
        # next_actions = next_actions.reshape(-1, self.action_dim) # (batch_size, action_horizon * action_dim)
        # Subsample from `N` actions on next state
        # key, rng = jax.random.split(key)
        # action_indices = jax.random.randint(rng, shape=(batch_size,), minval=0, maxval=self.N)
        # next_actions = next_actions[jnp.arange(batch_size), action_indices, :, :]
        # next_actions = next_actions.reshape(-1, self.action_dim) # (batch_size, action_horizon * action_dim)

        # No need to append state here as it is already appended in the batch #

        # key, rng = jax.random.split(key)
        current_vlm_output = batch['vlm_output']
        # No need to append state here as it is already appended in the batch #
        actions = batch["actions"].reshape(-1, self.action_dim) # (batch_size, action_horizon * action_dim)
        next_base_actions = next_base_actions.reshape(-1, self.action_dim)
        # seed, rng = jax.random.split(rng)
        target_params = subsample_ensemble(
            rng, self.target_critic.params, self.num_min_qs, self.num_qs
        )
        
        # r_observations = jnp.concatenate([next_vlm_output, next_base_actions], axis=1) # (1, pi0_hidden_dims + action_horizon * action_dim)
        # r_samples, _ =  _sample_actions(rng, self.edit_actor.apply_fn, self.edit_actor.params, r_observations) # (n_edit_samples, action_horizon * action_dim)
        # next_actions = r_samples * self.edit_action_scale + next_base_actions
        # next_qs = compute_q(self.target_critic.apply_fn, target_params, next_vlm_output, next_actions) # (batch_size, )
        # masks = 1.0 - batch['terminals']
        # target_q = batch["rewards"] + self.discount * masks * next_qs # (batch_size, )

        # Use JITted version #
        timer.tick("critic_loss_and_grad_time")
        key, rng = jax.random.split(rng)
        key, sample_key = jax.random.split(key)
        critic, target_critic_params, grads, info = _critic_loss_and_grad(
            self.critic.params,
            self.target_critic.params,
            current_vlm_output,
            actions,
            # target_q,
            batch["mc_returns"],
            rng,
            self.critic.apply_fn,
            # #
            target_params,
            next_vlm_output,
            next_base_actions,
            sample_key,
            self.edit_actor.apply_fn,
            self.edit_actor.params,
            self.edit_action_scale,
            self.target_critic.apply_fn,
            batch["terminals"],
            batch["rewards"],
            self.discount,
            # #
            self.critic,
            self.tau,
            # #
            self.q_clip_low,
            self.q_clip_high,
        )
        timer.tock("critic_loss_and_grad_time")

        timer.tick("apply_gradients_time")
        info["critic_grad_norm"] = optax.global_norm(grads)
        # critic = self.critic.apply_gradients(grads=grads)
        timer.tock("apply_gradients_time")
        # timer.tock("critic_loss_and_grad_time")

        timer.tick("incremental_update_time")
        # target_critic_params = optax.incremental_update(
        #     critic.params, self.target_critic.params, self.tau
        # )
        target_critic = self.target_critic.replace(params=target_critic_params)
        timer.tock("incremental_update_time")

        info["critic_param_norm"] = optax.global_norm(critic.params)
        info["target_critic_param_norm"] = optax.global_norm(target_critic.params)

        return self.replace(critic=critic, target_critic=target_critic, rng=key), info

    def preproess_batch(self, batch: Batch, *args, **kwargs) -> Batch:
        action_dim = self.action_dim // self.action_horizon
        if batch['actions'].shape[-1] > action_dim:
            if batch['actions'].ndim == 3:
                batch['actions'] = batch['actions'][:, :, :action_dim]
            elif batch['actions'].ndim == 2:
                batch['actions'] = batch['actions'][:, :action_dim]
            else:
                raise ValueError(f"Actions must be 3 or 2 dimensional, got {batch['actions'].shape}")
        
        return batch

    # @partial(jax.jit, static_argnames="utd_ratio")
    def update(self, _observations: Data, utd_ratio: int, *args, **kwargs):
        timer = kwargs.pop("timer", None)
        update_only_critic = kwargs.pop("update_only_critic", False) # For warmstarting critic before starting online training #

        new_agent = self
        # breakpoint()
        timer.tick("total_update_time")
        timer.tick("preprocess_time")

        # Preprocess batch at once to save computation #
        batch = self.preproess_batch(_observations.copy())
        state = batch['observations']['proprio'][:, :8].copy()
        next_state = batch['next_observations']['proprio'][:, :8].copy()
        # breakpoint()
        # Normalize state and actions #
        # Observations #
        # observations = self.actor.convert_to_openpi_format_infer(batch, obs_key="observations")
        # obs = self.actor.input_data_transforms(observations)
        # next_observations = self.actor.convert_to_openpi_format_infer(batch, obs_key="next_observations")
        # next_obs = self.actor.input_data_transforms(next_observations)
        # state = obs['state'][:, :8].copy()
        # next_state = next_obs['state'][:, :8].copy()
        # Filter batch to only keep relevant information #
        relevant_keys = [
            "actions", "rewards", "masks", "mc_returns", 
            "terminals", "truncates", "vlm_output", "next_vlm_output", "diffusion_actions", "next_diffusion_actions",
            "next_actions",
        ]
        batch = {k: v for k, v in batch.items() if k in relevant_keys}
        # Normalization of state has already happened in image_replay_buffer_pi.py #
        batch['state'] = state
        batch['next_state'] = next_state

        timer.tock("preprocess_time")
        timer.tick("update_critic_time")

        batch_size = state.shape[0]
        print("---")
        print("Batch size for update: ", batch_size)

        seed = kwargs.pop("seed", None)
        assert seed is not None, "Seed must be provided"
        seed, rng = jax.random.split(seed)

        for i in range(utd_ratio):
            def slice(x):
                assert x.shape[0] % utd_ratio == 0
                batch_size = x.shape[0] // utd_ratio
                return x[batch_size * i : batch_size * (i + 1)]

            mini_batch = jax.tree_util.tree_map(slice, batch)
            seed, rng = jax.random.split(seed)

            break
        
        new_agent, critic_info = new_agent.update_critic(batch, timer=timer, seed=rng)

        critic_info = append_substr_to_dict_keys(critic_info, "critic")
        timer.tock("update_critic_time")
        # breakpoint()
        
        if update_only_critic:
            return new_agent, {**critic_info}

        # timer.tick("update_actor_time")
        # new_agent, actor_update_info = new_agent.update_actor(mini_batch_original_obs)
        # timer.tock("update_actor_time")
        # actor_update_info = append_substr_to_dict_keys(actor_update_info, "actor")
        actor_update_info = {}
        temp_info = {}
        # # breakpoint()

        timer.tick("update_edit_actor_time")
        if self.n_edit_samples > 0:
            seed, rng = jax.random.split(seed)
            new_agent, actor_info = new_agent.update_edit_actor(mini_batch, seed=rng)
            entropy = actor_info["entropy"]
            actor_info = append_substr_to_dict_keys(actor_info, "edit_actor")
            # breakpoint()
            new_agent, temp_info = new_agent.update_temperature(entropy)
            temp_info = append_substr_to_dict_keys(temp_info, "temp")
            # breakpoint()
            # actor_info.update(temp_info)
        timer.tock("update_edit_actor_time")
        timer.tock("total_update_time")
        print(timer.get_total_times(reset=False))

        return new_agent, {**actor_info, **critic_info, **actor_update_info, **temp_info}
