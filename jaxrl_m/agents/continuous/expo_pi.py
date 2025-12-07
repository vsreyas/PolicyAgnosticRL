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
        target_actor = PiPolicy(rng=rng, config=config, is_target=True)
        
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


    def sample_batch_actions(self, _observations: Data, *args, **kwargs):
        # rng = self.rng

        # observations = jnp.squeeze(observations)
        # observations = jax.device_put(observations)
        
        # batch_size = observations.shape[0]
        # observations_repeated = jnp.repeat(observations, self.N, axis=0)

        # actor_params = self.actor.params
        # actions_flat, rng = ddpm_train_sampler(
        #     self.actor.apply_fn, actor_params, self.T, rng, self.action_dim, 
        #     observations_repeated, self.alphas, self.alpha_hats, self.betas, 
        #     self.ddpm_temperature, self.M, self.clip_sampler
        # )

        # actions = actions_flat.reshape(batch_size, self.N, -1)
        # observations_repeated = jnp.repeat(observations, self.N + self.n_edit_samples, axis=0)
        
        # if self.n_edit_samples > 0:
        #     key, rng = jax.random.split(rng, 2)
        #     r_observations = jnp.repeat(observations, self.n_edit_samples, axis=0) # (batch_size * n_edit_samples, obs_dim) repeats
        #     d_actions = actions.copy()[:, :self.n_edit_samples].reshape(-1, actions.shape[-1])
        #     r_observations = jnp.concatenate([r_observations, d_actions], axis=1) # self.n_edit_samples actions for each observation
        #     r_samples, rng =  _sample_actions(key, self.edit_actor.apply_fn, self.edit_actor.params, r_observations)
        #     r_samples = r_samples * self.edit_action_scale + d_actions
        #     actions = jnp.concatenate([actions, r_samples.reshape(batch_size, self.n_edit_samples, -1)], axis=1)
        #     actions_flat = actions.reshape(-1, actions.shape[-1])


        
        # if self.N > 1:
        #     key, rng = jax.random.split(rng)
        #     target_params = subsample_ensemble(
        #         key, self.target_critic.params, self.num_min_qs, self.num_qs
        #     )
            
        #     obs_flat = observations_repeated
        #     actions_flat = actions_flat

        #     qs = compute_q(self.target_critic.apply_fn, target_params, obs_flat, actions_flat)
        #     qs = qs.reshape(batch_size, self.N + self.n_edit_samples)

        #     best_indices = jnp.argmax(qs, axis=1) 
        #     batch_indices = jnp.arange(batch_size)
        #     best_actions = actions[batch_indices, best_indices] 
        # else:
        #     best_actions = actions[:, 0] 
        
        # rng, _ = jax.random.split(rng, 2)
        # return jnp.array(best_actions.squeeze())

        # Extract rng from kwargs
        # seed = kwargs.pop("seed")
        # # Split seed #
        # seed, rng = jax.random.split(seed)
        seed = kwargs.pop("seed")
        seed, rng = jax.random.split(seed)
        timer = kwargs.pop("timer")
        # timer.tick("total_sample_actions_time")
        # timer.tick("sample_actions_time")
        # Repeat observations to sample `N` actions#
        observations = repeat_observations(_observations, self.N, axis=0)

        # breakpoint()
        actions = self.actor.sample_actions(observations, seed=rng) # (N, action_horizon, action_dim)
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
    
    

    def sample_actions(self, _observations: Data, *args, **kwargs):
        # Extract rng from kwargs
        # seed = kwargs.pop("seed")
        # # Split seed #
        # seed, rng = jax.random.split(seed)
        seed = kwargs.pop("seed")
        seed, rng = jax.random.split(seed)
        timer = kwargs.pop("timer")
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
    

    def update_edit_actor(self, batch: Batch) -> Tuple[Agent, Dict[str, float]]:
        key, rng = jax.random.split(self.rng)
        key2, rng = jax.random.split(rng)
        dropout_key, rng = jax.random.split(rng)

        def edit_actor_loss_fn(actor_params) -> Tuple[jnp.ndarray, Dict[str, float]]:

            edit_observations = jnp.concatenate([batch["observations"], batch["actions"]], axis=1)
            dist = self.edit_actor.apply_fn({"params": actor_params}, edit_observations, training=True, rngs={"dropout": dropout_key},)
            actions = dist.sample(seed=key)

            log_probs = dist.log_prob(actions)
            actions = actions * self.edit_action_scale
            log_probs -= actions.shape[-1] * jnp.log(self.edit_action_scale)

            actions += batch["actions"]

        
            qs = self.critic.apply_fn(
                {"params": self.critic.params},
                batch["observations"],
                actions,
                True,
                rngs={"dropout": key2},
            )  # training=True
            q = qs.mean(axis=0)
            edit_actor_loss = (
                self.entropy_scale * log_probs * self.temp.apply_fn({"params": self.temp.params}) - q
            ).mean()
            return edit_actor_loss, {"edit_q": q.mean(), "edit_actor_loss": edit_actor_loss, "entropy": -log_probs.mean()}

        grads, actor_info = jax.grad(edit_actor_loss_fn, has_aux=True)(self.edit_actor.params)
        edit_actor = self.edit_actor.apply_gradients(grads=grads)

        return self.replace(edit_actor=edit_actor, rng=rng), actor_info
    

    def update_actor(self, batch: Batch) -> Tuple[Agent, Dict[str, float]]:
        rng = self.rng
        key, rng = jax.random.split(rng, 2)
        time = jax.random.randint(key, (batch['actions'].shape[0], ), 0, self.T)
        key, rng = jax.random.split(rng, 2)
        noise_sample = jax.random.normal(
            key, (batch['actions'].shape[0], self.action_dim))
        
        alpha_hats = self.alpha_hats[time]
        time = jnp.expand_dims(time, axis=1)
        alpha_1 = jnp.expand_dims(jnp.sqrt(alpha_hats), axis=1)
        alpha_2 = jnp.expand_dims(jnp.sqrt(1 - alpha_hats), axis=1)
        noisy_actions = alpha_1 * batch['actions'] + alpha_2 * noise_sample

        key, rng = jax.random.split(rng, 2)

        def actor_loss_fn(score_model_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            eps_pred = self.actor.apply_fn({'params': score_model_params},
                                       batch['observations'],
                                       noisy_actions,
                                       time,
                                       rngs={'dropout': key},
                                       training=True)
            
            actor_loss = (((eps_pred - noise_sample) ** 2).sum(axis = -1)).mean()
            return actor_loss, {'actor_loss': actor_loss}

        grads, info = jax.grad(actor_loss_fn, has_aux=True)(self.actor.params)
        actor = self.actor.apply_gradients(grads=grads)

        agent = self.replace(actor=actor)
        target_score_params = optax.incremental_update(
            actor.params, self.target_actor.params, self.actor_tau
        )

        target_score_model = self.target_actor.replace(params=target_score_params)
        new_agent = self.replace(actor=actor, target_actor=target_score_model, rng=rng)

        return new_agent, info

    def update_temperature(self, entropy: float) -> Tuple[Agent, Dict[str, float]]:
        def temperature_loss_fn(temp_params):
            temperature = self.temp.apply_fn({"params": temp_params})
            temp_loss = temperature * (entropy - self.target_entropy).mean()
            return temp_loss, {}

        grads, temp_info = jax.grad(temperature_loss_fn, has_aux=True)(self.temp.params)
        temp = self.temp.apply_gradients(grads=grads)

        return self.replace(temp=temp), temp_info

    def update_critic(self, batch: Batch) -> Tuple[TrainState, Dict[str, float]]:
        breakpoint()

        next_actions = self.sample_batch_actions(batch["next_observations"])
        rng = self.rng

        # Used only for REDQ.
        key, rng = jax.random.split(rng)
        target_params = subsample_ensemble(
            key, self.target_critic.params, self.num_min_qs, self.num_qs
        )

        key, rng = jax.random.split(rng)
        next_qs = self.target_critic.apply_fn(
            {"params": target_params},
            batch["next_observations"],
            next_actions,
            True,
            rngs={"dropout": key},
        )  # training=True
        next_q = next_qs.min(axis=0)

        target_q = batch["rewards"] + self.discount * batch["masks"] * next_q

        key, rng = jax.random.split(rng)

        def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qs = self.critic.apply_fn(
                {"params": critic_params},
                batch["observations"],
                batch["actions"],
                True,
                rngs={"dropout": key},
            )  # training=True
            critic_loss = ((qs - target_q) ** 2).mean()
            return critic_loss, {"critic_loss": critic_loss, "q": qs.mean()}

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(self.critic.params)
        critic = self.critic.apply_gradients(grads=grads)

        target_critic_params = optax.incremental_update(
            critic.params, self.target_critic.params, self.tau
        )
        target_critic = self.target_critic.replace(params=target_critic_params)

        return self.replace(critic=critic, target_critic=target_critic, rng=rng), info
    

    @partial(jax.jit, static_argnames=("utd_ratio", "pretrain_q", "pretrain_edit"))
    def update_offline(self, batch: Batch, utd_ratio: int, pretrain_q: bool, pretrain_edit: bool):

        new_agent = self
        for i in range(utd_ratio):

            def slice(x):
                assert x.shape[0] % utd_ratio == 0
                batch_size = x.shape[0] // utd_ratio
                return x[batch_size * i : batch_size * (i + 1)]

            mini_batch = jax.tree_util.tree_map(slice, batch)
            critic_info = {}
            if pretrain_q:
                new_agent, critic_info = new_agent.update_critic(mini_batch)

        new_agent, actor_info = new_agent.update_actor(mini_batch)

        if pretrain_edit:

            if self.n_edit_samples > 0:
                new_agent, actor_info = new_agent.update_edit_actor(mini_batch)
                new_agent, temp_info = new_agent.update_temperature(actor_info["entropy"])

                actor_info.update(temp_info)

        return new_agent, {**actor_info, **critic_info}
    


    # @partial(jax.jit, static_argnames="utd_ratio")
    def update(self, _observations: Data, utd_ratio: int):

        new_agent = self
        for i in range(utd_ratio):
            breakpoint()

            def slice(x):
                assert x.shape[0] % utd_ratio == 0
                batch_size = x.shape[0] // utd_ratio
                return x[batch_size * i : batch_size * (i + 1)]

            mini_batch = jax.tree_util.tree_map(slice, _observations)
            new_agent, critic_info = new_agent.update_critic(mini_batch)

        new_agent, actor_info = new_agent.update_actor(mini_batch)

        if self.n_edit_samples > 0:
            new_agent, actor_info = new_agent.update_edit_actor(mini_batch)
            new_agent, temp_info = new_agent.update_temperature(actor_info["entropy"])

            actor_info.update(temp_info)

        return new_agent, {**actor_info, **critic_info}
