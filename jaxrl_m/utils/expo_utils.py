# All EXPO related utilities #

import functools
from typing import Optional, Type
from typing import Any, Dict, Optional
from dataclasses import field

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

import flax.linen as nn
import jax.numpy as jnp
import numpy as np

import jax
import jax.numpy as jnp
from flax import struct
from flax.training.train_state import TrainState
from jaxrl_m.common.initialization import init_fns

from functools import partial

from jaxrl_m.common.traj import calc_return_to_go

import openpi.models.model as _model

# from jaxrl_m.types import PRNGKey

from typing import Any, Optional, Type, Callable, Sequence

# Inspired by
# https://github.com/deepmind/acme/blob/300c780ffeb88661a41540b99d3e25714e2efd20/acme/jax/networks/distributional.py#L163
# but modified to only compute a mode.


# TODO: Make changes to all these networks based on input shapes #

class TanhTransformedDistribution(tfd.TransformedDistribution):
    def __init__(self, distribution: tfd.Distribution, validate_args: bool = False):
        super().__init__(
            distribution=distribution, bijector=tfb.Tanh(), validate_args=validate_args
        )

    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())

    @classmethod
    def _parameter_properties(cls, dtype: Optional[Any], num_classes=None):
        td_properties = super()._parameter_properties(dtype, num_classes=num_classes)
        del td_properties["bijector"]
        return td_properties


class Normal(nn.Module):
    base_cls: Type[nn.Module]
    action_dim: int
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    state_dependent_std: bool = True
    squash_tanh: bool = False

    @nn.compact
    def __call__(self, inputs, *args, **kwargs) -> tfd.Distribution:
        x = self.base_cls()(inputs, *args, **kwargs)

        means = nn.Dense(
            self.action_dim, kernel_init=default_init(), name="OutputDenseMean"
        )(x)
        if self.state_dependent_std:
            log_stds = nn.Dense(
                self.action_dim, kernel_init=default_init(), name="OutputDenseLogStd"
            )(x)
        else:
            log_stds = self.param(
                "OutpuLogStd", nn.initializers.zeros, (self.action_dim,), jnp.float32
            )

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = tfd.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds)
        )

        if self.squash_tanh:
            return TanhTransformedDistribution(distribution)
        else:
            return distribution


TanhNormal = functools.partial(Normal, squash_tanh=True)


default_init = nn.initializers.xavier_uniform


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.swish
    activate_final: bool = False
    use_layer_norm: bool = False
    scale_final: Optional[float] = None
    dropout_rate: Optional[float] = None
    use_pnorm: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:

        for i, size in enumerate(self.hidden_dims):
            if i + 1 == len(self.hidden_dims) and self.scale_final is not None:
                x = nn.Dense(size, kernel_init=default_init(self.scale_final))(x)
            else:
                x = nn.Dense(size, kernel_init=default_init())(x)

            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training
                    )
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
        if self.use_pnorm:
            x /= jnp.linalg.norm(x, axis=-1, keepdims=True).clip(1e-10)
        return x

class MLPResnetEncoding(nn.Module):
    encoder: nn.Module
    network: nn.Module

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        train: bool = False,
    ) -> jnp.ndarray:
        obs_enc = self.encoder(observations, train=train)
        # jax.debug.breakpoint()
        mlp_input = jnp.concatenate([obs_enc, actions], -1)
        outputs = self.network()(mlp_input, training=train)
        return outputs

        # if self.encoder is None:
        #     obs_enc = observations
        # else:
        #     obs_enc = self.encoder(observations, train=train)

        # if self.network_separate_action_input:
        #     outputs = self.network(obs_enc, actions, train)
        # else:
        #     inputs = jnp.concatenate([obs_enc, actions], -1)
        #     outputs = self.network(inputs, train)
        # if self.init_final is not None:
        #     value = nn.Dense(
        #         1,
        #         kernel_init=nn.initializers.uniform(-self.init_final, self.init_final),
        #     )(outputs)
        # else:
        #     value = nn.Dense(1, kernel_init=self.init_fn(**self.kernel_init_params))(
        #         outputs
        #     )
        # return jnp.squeeze(value, -1)


class StateValue(nn.Module):
    base_cls: nn.Module

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, *args, **kwargs
    ) -> jnp.ndarray:
        inputs = jnp.concatenate([observations], axis=-1)
        outputs = self.base_cls()(inputs, *args, **kwargs)

        value = nn.Dense(1, kernel_init=default_init())(outputs)

        return jnp.squeeze(value, -1)


class StateActionValue(nn.Module):
    base_cls: nn.Module

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray, *args, **kwargs
    ) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], axis=-1)
        outputs = self.base_cls()(inputs, *args, **kwargs)

        value = nn.Dense(1, kernel_init=default_init())(outputs)

        return jnp.squeeze(value, -1)


class Ensemble(nn.Module):
    net_cls: Type[nn.Module]
    num: int = 2

    @nn.compact
    def __call__(self, *args):
        ensemble = nn.vmap(
            self.net_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num,
        )
        return ensemble()(*args)


def subsample_ensemble(key: jax.random.PRNGKey, params, num_sample: int, num_qs: int):
    if num_sample is not None:
        all_indx = jnp.arange(0, num_qs)
        indx = jax.random.choice(key, a=all_indx, shape=(num_sample,), replace=False)

        if "Ensemble_0" in params:
            ens_params = jax.tree_util.tree_map(
                lambda param: param[indx], params["Ensemble_0"]
            )
            params = params.copy(add_or_replace={"Ensemble_0": ens_params})
        else:
            params = jax.tree_util.tree_map(lambda param: param[indx], params)
    return params


@partial(jax.jit, static_argnames="apply_fn")
def _sample_actions(rng, apply_fn, params, observations: np.ndarray) -> np.ndarray:
    key, rng = jax.random.split(rng)
    dist = apply_fn({"params": params}, observations)
    return dist.sample(seed=key), rng


@partial(jax.jit, static_argnames="apply_fn")
def _eval_actions(apply_fn, params, observations: np.ndarray) -> np.ndarray:
    dist = apply_fn({"params": params}, observations)
    return dist.mode()


class Agent(struct.PyTreeNode):
    actor: TrainState
    rng: Any

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        actions = _eval_actions(self.actor.apply_fn, self.actor.params, observations)
        return np.asarray(actions)

    def sample_actions(self, observations: np.ndarray) -> np.ndarray:
        actions, new_rng = _sample_actions(
            self.rng, self.actor.apply_fn, self.actor.params, observations
        )
        return np.asarray(actions), self.replace(rng=new_rng)

class Temperature(nn.Module):
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param(
            "log_temp",
            init_fn=lambda key: jnp.full((), jnp.log(self.initial_temperature)),
        )
        return jnp.exp(log_temp)

def calc_mc_return_fn(rewards, masks, discount, reward_bias):
    # breakpoint()
    return calc_return_to_go(
        rewards,
        masks,
        discount,
        push_failed_to_min=False, # Use only for real robot case
        min_reward=reward_bias,
    )

def repeat_observations(observations, N, axis=0):
    if isinstance(observations, dict):
        return {k: repeat_observations(v, N) for k, v in observations.items()}
    elif isinstance(observations, np.ndarray):
        return jnp.repeat(jnp.expand_dims(observations, axis), N, axis=axis)
    elif isinstance(observations, str):
        return [observations] * N
    elif isinstance(observations, jax.Array):
        return jnp.repeat(jnp.expand_dims(observations, axis), N, axis=axis)
    else:
        raise ValueError(f"Unsupported type: {type(observations)}")

# def repeat_observations_batched(observations, N, axis=0):
#     if isinstance(observations, dict):
#         return {k: repeat_observations_batched(v, N) for k, v in observations.items()}
#     elif isinstance(observations, np.ndarray):
#         return jnp.repeat(observations, N, axis=axis)
#     elif isinstance(observations, str):
#         return observations
#     elif isinstance(observations, jax.Array):
#         return jnp.repeat(observations, N, axis=axis)
#     else:
#         raise ValueError(f"Unsupported type: {type(observations)}")
    
# def repeat_observations_openpi(obs: _model.Observation, N: int, axis: int = 0) -> _model.Observation:
#     """
#     Repeats all array fields within an Observation instance N times along the specified axis.
#     """
#     def _repeat_leaf(x):
#         # Handle None fields (e.g. optional masks)
#         if x is None:
#             return None
#         # Handle JAX and NumPy arrays
#         if isinstance(x, (jax.Array, np.ndarray)):
#             return jnp.repeat(x, N, axis=axis)
#         # Pass through other types (strings, etc. if any exist)
#         return x

#     return jax.tree.map(_repeat_leaf, obs)

def repeat_observations_batched(observations, N, axis=0):
    """
    Repeats the batch N times by tiling (stacking copies of the full batch).
    Input:  [A, B] (Batch size 2)
    Output: [A, B, A, B] (Batch size 4)
    """
    if isinstance(observations, dict):
        return {k: repeat_observations_batched(v, N, axis=axis) for k, v in observations.items()}
    elif isinstance(observations, (np.ndarray, jax.Array)):
        # Tiling: Concatenate N copies of the array
        return jnp.concatenate([observations] * N, axis=axis)
    elif isinstance(observations, str):
        return observations
    else:
        # Fallback for other scalar types or unexpected structures
        return observations

def repeat_observations_openpi(obs: _model.Observation, N: int, axis: int = 0) -> _model.Observation:
    """
    Repeats all array fields within an Observation instance N times by tiling.
    Input:  [A, B]
    Output: [A, B, A, B]
    """
    def _tile_leaf(x):
        # Handle None fields (e.g. optional masks)
        if x is None:
            return None
        # Handle JAX and NumPy arrays
        if isinstance(x, (jax.Array, np.ndarray)):
            return jnp.concatenate([x] * N, axis=axis)
        # Pass through other types (strings, etc.)
        return x

    return jax.tree.map(_tile_leaf, obs)

def append_substr_to_dict_keys(dict, substr):
    return {substr + '/' + k: v for k, v in dict.items()}
