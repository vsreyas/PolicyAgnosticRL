"""Factory interface for residual agents."""

import inspect
from .residual_td3 import PiResidualTD3Cache
from .residual_ppo import PiResidualPPOCache
from .residual_td3_grpo import PiResidualTD3GRPO

# Map agent names to their corresponding classes
agents_dict = {
    "pi_residual_td3": PiResidualTD3Cache,
    "pi_residual_ppo": PiResidualPPOCache,
    "pi_residual_td3_grpo": PiResidualTD3GRPO,
}


def create_agent(agent_name, config, seed, batch_size, rng, params_path=None, **kwargs):
    """Factory method to create residual agents.
    
    Args:
        agent_name: Name of the agent (e.g., 'pi_residual_td3', 'pi_residual_ppo')
        config: PI config object (from openpi.training.config)
        seed: Random seed
        batch_size: Batch size for training
        rng: JAX random key
        params_path: Optional path to checkpoint file
        **kwargs: Agent-specific kwargs (e.g., N, n_edit_samples, exploration_epsilon)
                 These typically come from FLAGS.config.agent_kwargs
    
    Returns:
        Agent instance (PiResidualTD3Cache or PiResidualPPOCache)
        
    Raises:
        ValueError: If agent_name is not in agents_dict
    
    Example:
        agent = create_agent(
            agent_name="pi_residual_td3",
            config=pi_config,
            seed=42,
            batch_size=256,
            rng=jax.random.PRNGKey(0),
            params_path="/path/to/checkpoint.pkl",
            N=4,
            n_edit_samples=4,
            exploration_epsilon=0.05,
        )
    """
    if agent_name not in agents_dict:
        raise ValueError(
            f"Unknown agent: {agent_name}. Available agents: {list(agents_dict.keys())}"
        )
    
    agent_class = agents_dict[agent_name]
    
    # Get the signature of the create method to filter valid kwargs
    create_signature = inspect.signature(agent_class.create)
    valid_params = set(create_signature.parameters.keys())
    
    # Filter kwargs to only include parameters accepted by create()
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    
    # Warn about ignored parameters (for debugging)
    ignored_params = set(kwargs.keys()) - valid_params
    if ignored_params:
        print(f"Warning: Ignoring parameters not accepted by {agent_name}: {ignored_params}")
    
    return agent_class.create(
        config=config,
        seed=seed,
        batch_size=batch_size,
        rng=rng,
        params_path=params_path,
        **filtered_kwargs
    )


__all__ = [
    "PiResidualTD3Cache",
    "PiResidualPPOCache",
    "PiResidualTD3GRPO",
    "agents_dict",
    "create_agent",
]
