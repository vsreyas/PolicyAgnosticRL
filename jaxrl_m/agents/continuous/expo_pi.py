"""Starter code from the RLPD repository https://github.com/ikostrikov/rlpd"""

from functools import partial
from typing import Dict, Optional, Sequence, Tuple

import flax
import gym
import jax
import jax.numpy as jnp
import optax
from flax import struct
import flax.linen as nn
from flax.training.train_state import TrainState

import numpy as np

# from expo.agents.agent import Agent
# from expo.agents.sac.temperature import Temperature
# from expo.data.dataset import Batch
# from expo.distributions import TanhNormal
# from expo.networks import (
#     MLP,
#     Ensemble,
#     MLPResNetV2,
#     StateActionValue,
#     subsample_ensemble,
# )
# from expo.networks import DiffusionMLP, DDPM, FourierFeatures, cosine_beta_schedule, ddpm_sampler, ddpm_train_sampler, DiffusionMLPResNet, get_weight_decay_mask, vp_beta_schedule

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

    q_values = critic_fn({'params': critic_params}, observations, actions)
    q_values = q_values.min(axis=0)
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
            training=True,
            rngs={"dropout": dropout_key},
        )
        actions = dist.sample(seed=key)

        log_probs = dist.log_prob(actions)
        actions = actions * edit_action_scale
        log_probs -= actions.shape[-1] * jnp.log(edit_action_scale)

        actions = actions + batch_actions

        qs = critic_apply_fn(
            {"params": critic_params},
            vlm_output,
            actions,
            True,
            rngs={"dropout": key2},
        )
        q = qs.mean(axis=0)

        temperature = temp_apply_fn({"params": temp_params})
        edit_actor_loss = (entropy_scale * log_probs * temperature - q).mean()

        metrics = {
            "edit_q": q.mean(),
            "edit_actor_loss": edit_actor_loss,
            "entropy": -log_probs.mean(),
        }
        return edit_actor_loss, metrics

    (loss, metrics), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(actor_params)
    return grads, metrics


@partial(jax.jit, static_argnames=("critic_apply_fn",))
def _critic_loss_and_grad(
    critic_params,
    vlm_output,
    actions,
    target_q,
    key,
    critic_apply_fn,
):
    """Single jitted step for critic: forward + loss + grad."""

    def loss_fn(critic_params):
        qs = critic_apply_fn(
            {"params": critic_params},
            vlm_output,
            actions,
            True,
            rngs={"dropout": key},
        )
        critic_loss = ((qs - target_q) ** 2).mean()
        metrics = {
            "critic_loss": critic_loss,
            "q": qs.mean(),
        }
        return critic_loss, metrics

    (loss, metrics), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(critic_params)
    return grads, metrics


@partial(jax.jit, static_argnames="apply_fn")
def _sample_actions(rng, apply_fn, params, observations: np.ndarray) -> np.ndarray:
    key, rng = jax.random.split(rng)
    dist = apply_fn({"params": params}, observations)
    return dist.sample(seed=key), rng


class ExpoPiLearner(Agent):
    critic: TrainState
    target_critic: TrainState
    target_actor: PiPolicy
    edit_actor: TrainState
    temp: TrainState
    action_dim: int = struct.field(pytree_node=False)
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

    @classmethod
    def create(
        cls,
        config: TrainConfig,
        seed: int,
        observations: Data,
        action_dim: int = 7,
        action_horizon: int = 10,
        pi0_hidden_dims: int = 4096,
        rng : PRNGKey | None = None,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        num_qs: int = 2,
        num_min_qs: Optional[int] = None,
        critic_dropout_rate: Optional[float] = None,
        critic_weight_decay: Optional[float] = None,
        critic_layer_norm: bool = False,
        target_entropy: Optional[float] = None,
        entropy_scale: float = 1.0, 
        init_temperature: float = 1.0,
        backup_entropy: bool = True,
        use_pnorm: bool = False,
        adjust_target_entropy: bool = False, 
        use_critic_resnet: bool = False,
        time_dim: int = 128,
        actor_drop: Optional[float] = None, 
        d_actor_drop: Optional[float] = None, 
        T: int = 10, 
        N: int = 4,
        batch_split: int = 1, 
        M: int = 0,
        n_edit_samples: int = 4, 
        edit_action_scale: float = 1.0, 
        actor_layer_norm: bool = True,
        clip_sampler: bool = True,
        decay_steps: Optional[int] = int(3e6),
        actor_tau: float = 0.001,
        actor_dropout_rate: Optional[float] = None,
        actor_num_blocks: int = 3,
        ddpm_temperature: float = 1.0,
        beta_schedule: str = 'vp',
    ):
        # breakpoint()

        # Assertions
        assert N >= n_edit_samples, f"N must be greater than or equal to n_edit_samples, got N={N} and n_edit_samples={n_edit_samples}"
        
        if target_entropy is None:
            target_entropy = -action_dim / 2
            
            if adjust_target_entropy:
                target_entropy = -action_dim / 2 + action_dim * jnp.log(edit_action_scale)
        
        paligemma_config = _gemma.get_config(config.model.paligemma_variant)
        pi0_hidden_dims = paligemma_config.width
        action_horizon = config.model.action_horizon
        action_dim = action_dim * action_horizon

        batch_size = observations['actions'].shape[0]


        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        # Init Pi0 model #
        # breakpoint()
        # TODO: Create actor and target actor with same params #
        actor = PiPolicy(rng=rng, config=config, is_target=False)
        # target_actor = PiPolicy(rng=rng, config=config, is_target=True)
        target_actor = None
        
        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        # Init edit actor #
        # Edit actor for now will take in pi0 VLM output hidden states, predicted base actions, concatenate them and compute residual action
        dummy_observations = jnp.ones((batch_size, pi0_hidden_dims)) # For initializing the edit actor
        dummy_actions = jnp.ones((batch_size, action_dim)) # For initializing the critic
        edit_actor_base_cls = partial(
            MLP, hidden_dims=hidden_dims, dropout_rate=actor_drop, activate_final=True, use_pnorm=use_pnorm
        )
        edit_actor_def = TanhNormal(edit_actor_base_cls, action_dim)
        edit_observations = jnp.concatenate([dummy_observations, jnp.ones((batch_size, action_dim))], axis=1)
        edit_actor_params = edit_actor_def.init(actor_key, edit_observations)["params"]
        edit_actor = TrainState.create(
            apply_fn=edit_actor_def.apply, 
            params=edit_actor_params, 
            tx=optax.adam(learning_rate=actor_lr),
        )

        # Init critic #
        critic_base_cls = partial(
            MLP,
            hidden_dims=hidden_dims,
            activate_final=True,
            dropout_rate=critic_dropout_rate,
            use_layer_norm=critic_layer_norm,
            use_pnorm=use_pnorm,
        )
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=num_qs)
        critic_params = critic_def.init(critic_key, dummy_observations, dummy_actions)["params"]
        if critic_weight_decay is not None:
            tx = optax.adamw(
                learning_rate=critic_lr,
                weight_decay=critic_weight_decay,
                mask=decay_mask_fn,
            )
        else:
            tx = optax.adam(learning_rate=critic_lr)
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
            tx=optax.adam(learning_rate=temp_lr),
        )

        # breakpoint()

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
        )
    

    def eval_actions(self, observations):
        rng = self.rng
        observations = jnp.squeeze(observations)
        assert len(observations.shape) == 1
        observations = jax.device_put(observations)
        observations = jnp.expand_dims(observations, axis = 0).repeat(self.N, axis = 0)

        actor_params = self.target_actor.params
        actions, rng = ddpm_sampler(self.actor.apply_fn, actor_params, self.T, rng, self.action_dim, observations, self.alphas, self.alpha_hats, self.betas, self.ddpm_temperature, self.M, self.clip_sampler)

        diffusion_actions = actions

        if self.N > 1:
            key, rng = jax.random.split(rng)
            target_params = subsample_ensemble(
                key, self.target_critic.params, self.num_min_qs, self.num_qs
            )

            if self.n_edit_samples > 0:
                key, rng = jax.random.split(rng, 2)

                observations = jnp.concatenate([observations, jnp.expand_dims(observations[0], axis = 0).repeat(self.n_edit_samples, axis = 0)], axis=0)

                r_observations = jnp.repeat(jnp.expand_dims(observations[0], axis = 0), self.n_edit_samples, axis=0)
                d_actions = diffusion_actions.copy()[:self.n_edit_samples]
                r_observations = jnp.concatenate([r_observations, d_actions], axis=1)
                r_samples, rng =  _sample_actions(key, self.edit_actor.apply_fn, self.edit_actor.params, r_observations)
                r_samples = r_samples * self.edit_action_scale + d_actions
                actions = jnp.concatenate([actions, r_samples], axis=0)
                
            qs = compute_q(self.target_critic.apply_fn, target_params, observations, actions)
            idx = jnp.argmax(qs)
            action = actions[idx]

        else:
        
            action = actions[0]

        rng, _ = jax.random.split(rng, 2)
        return np.array(action.squeeze()), self.replace(rng=rng)


    def sample_batch_actions(self, _observations: Data | Batch, *args, **kwargs):
        # Optional RNG + timer from kwargs (similar to sample_actions)
        seed = kwargs.pop("seed", None)
        timer = kwargs.pop("timer", None)
        infer = kwargs.pop("infer", False)
        obs_key = kwargs.pop("obs_key", "observations")
        return_first_action = kwargs.pop("return_first_action", False)

        if seed is None:
            rng = self.rng
        else:
            seed, rng = jax.random.split(seed)

        if timer is not None:
            timer.tick("sample_batch_actions_time")

        if 'observations' in _observations:
            batch_size = _observations['observations']['state'].shape[0] if 'state' in _observations['observations'] else _observations['observations']['proprio'].shape[0]
        else:
            batch_size = _observations['state'].shape[0] if 'state' in _observations else _observations['proprio'].shape[0]

        # Sample `N` actions from base policy #
        # Repeat observations to sample `N` actions
        observations_repeated = repeat_observations_batched(_observations, self.N, axis=0) # (batch_size * N, ...)

        actions = self.actor.sample_actions(observations_repeated, seed=rng, infer=infer, obs_key=obs_key) # (batch_size * N, action_horizon, action_dim)
        diffusion_actions = actions

        # Do a forward pass to get VLM output for all observations in the batch #
        seed, rng = jax.random.split(rng)
        vlm_output, _ = self.actor.get_vlm_output(seed, _observations, infer=infer, obs_key=obs_key)
        # vlm_output: (B, num_tokens, hidden_dim) or similar
        vlm_output = jnp.mean(vlm_output[0], axis=1)  # (B, pi0_hidden_dims)
        # vlm_output_repeated = repeat_observations(vlm_output, self.N, axis=0) # (batch_size * N, pi0_hidden_dims)

        if self.N > 1:
            key, rng = jax.random.split(rng)
            target_params = subsample_ensemble(
                key, self.target_critic.params, self.num_min_qs, self.num_qs
            )

            if self.n_edit_samples > 0:
                key, rng = jax.random.split(rng, 2)

                r_observations = repeat_observations_batched(vlm_output, self.n_edit_samples, axis=0) # (batch_size * n_edit_samples, pi0_hidden_dims)
                d_actions = diffusion_actions.copy().reshape(batch_size, self.n_edit_samples, self.action_horizon, self.action_dim // self.action_horizon)
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
                observations_repeated = repeat_observations_batched(vlm_output, self.N + self.n_edit_samples, axis=0) # (batch_size * (N + n_edit_samples), pi0_hidden_dims)
            else:
                observations_repeated = repeat_observations_batched(vlm_output, self.N, axis=0) # (batch_size * N, pi0_hidden_dims)

            actions = actions.reshape(-1, self.action_dim) # (batch_size * (N + n_edit_samples), action_horizon * action_dim)
            qs = compute_q(self.target_critic.apply_fn, target_params, observations_repeated, actions)
            qs = qs.reshape(batch_size, self.N + self.n_edit_samples) # (batch_size, N + n_edit_samples)
            idx = jnp.argmax(qs, axis=1) # (batch_size, )
            batch_idx = jnp.arange(batch_size)
            actions = actions.reshape(batch_size, self.N + self.n_edit_samples, self.action_horizon, self.action_dim // self.action_horizon)

            if return_first_action:
                action = actions[batch_idx, idx, 0, :] # (batch_size, action_dim)
            else:
                action = actions[batch_idx, idx, :, :] # (batch_size, action_horizon, action_dim)
        else:
            raise ValueError(f"N must be greater than 1, got {self.N}")

        if timer is not None:
            timer.tock("sample_batch_actions_time")

        rng, _ = jax.random.split(rng, 2)
        # We *don’t* mutate self.rng here; caller controls RNG via `seed`
        return np.array(action.squeeze())
    
    

    def sample_actions(self, _observations: Data, *args, **kwargs):
        '''
        For given state/observation, samles self.N actions from base policy; For first self.n_edit_samples actions, samples from edit_actor;
        Combine all (N + n_edit_samples) actions and compute Q-values; Return the action with the highest Q-value;
        '''
        # Extract rng from kwargs
        # seed = kwargs.pop("seed")
        # # Split seed #
        # seed, rng = jax.random.split(seed)
        seed = kwargs.pop("seed", None)
        seed, rng = jax.random.split(seed)
        timer = kwargs.pop("timer", None)
        # timer.tick("total_sample_actions_time")
        # timer.tick("sample_actions_time")
        # Repeat observations to sample `N` actions#
        observations = repeat_observations(_observations, self.N, axis=0)

        # breakpoint()
        actions = self.target_actor.sample_actions(observations, seed=rng) # (N, action_horizon, action_dim)
        # timer.tock("sample_actions_time")
        # breakpoint()
        diffusion_actions = actions
        
        # # DEBUG #
        # action = actions[0, 0, :]
        # #########

        # Do a forward pass to get VLM output #
        seed, rng = jax.random.split(rng)
        vlm_output, _ = self.target_actor.get_vlm_output(rng, observations)
        # vlm_output: (N, 256 * num_images + 200 (tokens), pi0_hidden_dims)
        # breakpoint()
        # Take mean across tokens as representation from VLM #
        vlm_output = jnp.mean(vlm_output[0], axis=1) # (N, pi0_hidden_dims)

        # NOTE: In all the computation in this class, self.action_dim = action_horizon * action_dim, so keep that semantic in mind #
        # This is done so that without any code modification, edit_actor outputs an action chunk #
        # TODO: Think of a better design #
        if self.N > 1:
            key, rng = jax.random.split(rng)
            target_params = subsample_ensemble(
                key, self.target_critic.params, self.num_min_qs, self.num_qs
            )

            if self.n_edit_samples > 0:
                key, rng = jax.random.split(rng, 2)

                observations = jnp.concatenate([vlm_output, jnp.expand_dims(vlm_output[0], axis = 0).repeat(self.n_edit_samples, axis = 0)], axis=0)
                # (N + n_edit_samples, pi0_hidden_dims)

                r_observations = jnp.repeat(jnp.expand_dims(vlm_output[0], axis = 0), self.n_edit_samples, axis=0) # (n_edit_samples, pi0_hidden_dims)
                d_actions = diffusion_actions.copy()[:self.n_edit_samples] # (n_edit_samples, action_horizon, action_dim)
                d_actions = d_actions.reshape(self.n_edit_samples, -1) # (n_edit_samples, action_horizon * action_dim)
                r_observations = jnp.concatenate([r_observations, d_actions], axis=1) # (n_edit_samples, pi0_hidden_dims + action_horizon * action_dim)
                r_samples, rng =  _sample_actions(key, self.edit_actor.apply_fn, self.edit_actor.params, r_observations) # (n_edit_samples, action_horizon * action_dim)
                r_samples = r_samples * self.edit_action_scale + d_actions
                # Need to divide by action_horizon as self.action_dim = action_horizon * action_dim in this class #
                r_samples = r_samples.reshape(-1, self.action_horizon, self.action_dim // self.action_horizon) # (n_edit_samples, action_horizon, action_dim)
                actions = jnp.concatenate([actions, r_samples], axis=0) # (N + n_edit_samples, action_horizon, action_dim)

            actions = actions.reshape(-1, self.action_dim) # (N + n_edit_samples, action_horizon * action_dim) # Accounting for the fact that self.action_dim = action_horizon * action_dim in this class #
            qs = compute_q(self.target_critic.apply_fn, target_params, observations, actions)
            # Keep only first action in chunk for choosing next action #
            actions = actions.reshape(self.N + self.n_edit_samples, self.action_horizon, self.action_dim // self.action_horizon)
            actions = actions[:, 0, :] # (N + n_edit_samples, action_dim)
            idx = jnp.argmax(qs)
            action = actions[idx]

            # breakpoint()

        else:
            raise ValueError(f"N must be greater than 1, got {self.N}")
        
        # timer.tock("total_sample_actions_time")
        # print(timer.get_total_times(reset=False))

        rng, _ = jax.random.split(rng, 2)
        return np.array(action.squeeze())
    

    def update_edit_actor(self, _batch: Batch, *args, **kwargs) -> Tuple[Agent, Dict[str, float]]:
        batch = self.preproess_batch(_batch.copy())

        seed = kwargs.pop("seed", None)
        timer = kwargs.pop("timer", None)
        infer = kwargs.pop("infer", False)
        obs_key = kwargs.pop("obs_key", "observations")

        if seed is None:
            rng = self.rng
        else:
            seed, rng = jax.random.split(seed)

        key, rng = jax.random.split(rng)
        key2, rng = jax.random.split(rng)
        dropout_key, rng = jax.random.split(rng)

        # Get base action dim out from given actions #
        base_action_dim = self.action_dim // self.action_horizon
        if batch['actions'].shape[-1] > base_action_dim:
            if batch['actions'].ndim == 3:
                batch['actions'] = batch['actions'][:, :, :base_action_dim]
            elif batch['actions'].ndim == 2:
                batch['actions'] = batch['actions'][:, :base_action_dim]
            else:
                raise ValueError(f"Actions must be 3 or 2 dimensional, got {batch['actions'].shape}")
            

        batch['actions'] = batch['actions'].reshape(-1, self.action_dim)
        
        # Get vlm output for observations #
        seed, rng = jax.random.split(rng)
        vlm_output, _ = self.actor.get_vlm_output(rng, batch, obs_key=obs_key)
        vlm_output = jnp.mean(vlm_output[0], axis=1) # (batch_size, pi0_hidden_dims)

        # def edit_actor_loss_fn(actor_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        #     edit_observations = jnp.concatenate([vlm_output, batch["actions"]], axis=1)
        #     dist = self.edit_actor.apply_fn({"params": actor_params}, edit_observations, training=True, rngs={"dropout": dropout_key},)
        #     actions = dist.sample(seed=key)

        #     log_probs = dist.log_prob(actions)
        #     actions = actions * self.edit_action_scale
        #     log_probs -= actions.shape[-1] * jnp.log(self.edit_action_scale)

        #     actions += batch["actions"]

        
        #     qs = self.critic.apply_fn(
        #         {"params": self.critic.params},
        #         vlm_output,
        #         actions,
        #         True,
        #         rngs={"dropout": key2},
        #     )  # training=True
        #     q = qs.mean(axis=0)
        #     edit_actor_loss = (
        #         self.entropy_scale * log_probs * self.temp.apply_fn({"params": self.temp.params}) - q
        #     ).mean()
        #     return edit_actor_loss, {"edit_q": q.mean(), "edit_actor_loss": edit_actor_loss, "entropy": -log_probs.mean()}

        # grads, actor_info = jax.grad(edit_actor_loss_fn, has_aux=True)(self.edit_actor.params)

        # Use JITted version #
        grads, actor_info = _edit_actor_loss_and_grad(
            self.edit_actor.params,
            self.critic.params,
            self.temp.params,
            vlm_output,
            batch["actions"],
            dropout_key,
            key,
            key2,
            self.entropy_scale,
            self.edit_action_scale,
            self.edit_actor.apply_fn,
            self.critic.apply_fn,
            self.temp.apply_fn,
        )
        edit_actor = self.edit_actor.apply_gradients(grads=grads)

        return self.replace(edit_actor=edit_actor, rng=rng), actor_info
    

    def update_actor(self, batch: Batch, *args, **kwargs) -> Tuple[Agent, Dict[str, float]]:
        seed = kwargs.pop("seed", None)
        timer = kwargs.pop("timer", None)
        infer = kwargs.pop("infer", False)
        obs_key = kwargs.pop("obs_key", "observations")

        if seed is None:
            rng = self.rng
        else:
            seed, rng = jax.random.split(seed)

        actor_update_info = self.actor.update(batch)
        # breakpoint()

        # Update target actor train_state with incremental parameter update #
        # target_score_params = optax.incremental_update(
        #     self.actor.train_state.params, self.target_actor.train_state.params, self.actor_tau
        # )
        # self.target_actor.train_state = self.target_actor.train_state.replace(params=target_score_params)

        # new_agent = self.replace(actor=self.actor, target_actor=self.target_actor, rng=rng)
        new_agent = self.replace(actor=self.actor, rng=rng)
        return new_agent, actor_update_info
        
    def update_temperature(self, entropy: float) -> Tuple[Agent, Dict[str, float]]:
        def temperature_loss_fn(temp_params):
            temperature = self.temp.apply_fn({"params": temp_params})
            temp_loss = temperature * (entropy - self.target_entropy).mean()
            return temp_loss, {}

        grads, temp_info = jax.grad(temperature_loss_fn, has_aux=True)(self.temp.params)
        temp = self.temp.apply_gradients(grads=grads)

        return self.replace(temp=temp), temp_info

    def update_critic(self, _batch: Batch, *args, **kwargs) -> Tuple[TrainState, Dict[str, float]]:
        batch = self.preproess_batch(_batch.copy())

        seed = kwargs.pop("seed", None)
        timer = kwargs.pop("timer", None)
        infer = kwargs.pop("infer", False)
        obs_key = kwargs.pop("obs_key", "observations")

        if seed is None:
            rng = self.rng
        else:
            seed, rng = jax.random.split(seed)

        # Sample next actions from base policy #
        next_actions = self.sample_batch_actions(batch, infer=True, obs_key=obs_key, return_first_action=False) # (batch_size, action_horizon, action_dim)
        next_actions = next_actions.reshape(-1, self.action_dim) # (batch_size, action_horizon * action_dim)
        # Get vlm output for next observations #
        seed, rng = jax.random.split(rng)
        next_vlm_output, _ = self.target_actor.get_vlm_output(rng, batch, infer=infer, obs_key="next_observations")
        next_vlm_output = jnp.mean(next_vlm_output[0], axis=1) # (batch_size, pi0_hidden_dims)
        # Get current vlm_output #
        seed, rng = jax.random.split(rng)
        current_vlm_output, _ = self.target_actor.get_vlm_output(rng, batch, infer=infer, obs_key="observations")
        current_vlm_output = jnp.mean(current_vlm_output[0], axis=1) # (batch_size, pi0_hidden_dims)
        actions = batch["actions"].reshape(-1, self.action_dim) # (batch_size, action_horizon * action_dim)

        seed, rng = jax.random.split(rng)

        # Used only for REDQ.
        key, rng = jax.random.split(rng)
        target_params = subsample_ensemble(
            key, self.target_critic.params, self.num_min_qs, self.num_qs
        )

        key, rng = jax.random.split(rng)
        next_qs = compute_q(self.target_critic.apply_fn, target_params, next_vlm_output, next_actions) # (batch_size, )

        target_q = batch["rewards"] + self.discount * batch["masks"] * next_qs # (batch_size, )

        # key, rng = jax.random.split(rng)

        # def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        #     qs = self.critic.apply_fn(
        #         {"params": critic_params},
        #         current_vlm_output,
        #         actions,
        #         True,
        #         rngs={"dropout": key},
        #     )  # training=True
        #     critic_loss = ((qs - target_q) ** 2).mean()
        #     return critic_loss, {"critic_loss": critic_loss, "q": qs.mean()}

        # grads, info = jax.grad(critic_loss_fn, has_aux=True)(self.critic.params)

        # Use JITted version #
        key, rng = jax.random.split(rng)
        grads, info = _critic_loss_and_grad(
            self.critic.params,
            current_vlm_output,
            actions,
            target_q,
            key,
            self.critic.apply_fn,
        )
        critic = self.critic.apply_gradients(grads=grads)

        target_critic_params = optax.incremental_update(
            critic.params, self.target_critic.params, self.tau
        )
        target_critic = self.target_critic.replace(params=target_critic_params)

        return self.replace(critic=critic, target_critic=target_critic, rng=rng), info
    

    # @partial(jax.jit, static_argnames=("utd_ratio", "pretrain_q", "pretrain_edit"))
    # def update_offline(self, batch: Batch, utd_ratio: int, pretrain_q: bool, pretrain_edit: bool):

    #     new_agent = self
    #     for i in range(utd_ratio):

    #         def slice(x):
    #             assert x.shape[0] % utd_ratio == 0
    #             batch_size = x.shape[0] // utd_ratio
    #             return x[batch_size * i : batch_size * (i + 1)]

    #         mini_batch = jax.tree_util.tree_map(slice, batch)
    #         critic_info = {}
    #         if pretrain_q:
    #             new_agent, critic_info = new_agent.update_critic(mini_batch)

    #     new_agent, actor_info = new_agent.update_actor(mini_batch)

    #     if pretrain_edit:

    #         if self.n_edit_samples > 0:
    #             new_agent, actor_info = new_agent.update_edit_actor(mini_batch)
    #             new_agent, temp_info = new_agent.update_temperature(actor_info["entropy"])

    #             actor_info.update(temp_info)

    #     return new_agent, {**actor_info, **critic_info}
    

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
        new_agent = self
        # breakpoint()
        timer.tick("total_update_time")
        timer.tick("update_critic_time")
        for i in range(utd_ratio):
            # breakpoint()

            def slice(x):
                assert x.shape[0] % utd_ratio == 0
                batch_size = x.shape[0] // utd_ratio
                return x[batch_size * i : batch_size * (i + 1)]

            mini_batch = jax.tree_util.tree_map(slice, _observations)
            # breakpoint()
            # new_agent, critic_info = new_agent.update_critic(mini_batch, infer=True, obs_key="next_observations")
        timer.tock("update_critic_time")
        # breakpoint()
        timer.tick("update_actor_time")
        new_agent, actor_update_info = new_agent.update_actor(mini_batch)
        timer.tock("update_actor_time")
        # breakpoint()

        timer.tick("update_edit_actor_time")
        # if self.n_edit_samples > 0:
        #     new_agent, actor_info = new_agent.update_edit_actor(mini_batch)
        #     # breakpoint()
        #     new_agent, temp_info = new_agent.update_temperature(actor_info["entropy"])
        #     # breakpoint()
        #     actor_info.update(temp_info)
        timer.tock("update_edit_actor_time")
        timer.tock("total_update_time")
        print(timer.get_total_times(reset=False))

        # return new_agent, {**actor_info, **critic_info, **actor_update_info}
        return new_agent, actor_update_info
