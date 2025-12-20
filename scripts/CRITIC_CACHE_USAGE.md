# Critic Update Cache - Usage Guide

## Overview

The `dump_vlm_actions.py` script has been updated to cache **all data needed for `update_critic` function** with the assumption that `output_only_base_actions=True`.

This allows you to:
1. Pre-compute VLM outputs for the entire dataset
2. Cache all necessary data to disk
3. Load from cache and run `update_critic` updates efficiently without recomputing VLM outputs

## What Data is Cached?

For each (episode_id, timestep) pair, the cache stores:

| Data Field | Description | Usage in update_critic |
|------------|-------------|----------------------|
| `obs` | Current observations (OpenPI format) | Pass to `sample_batch_actions` for next_actions |
| `next_obs` | Next observations (OpenPI format) | Pass to `sample_batch_actions` for next_actions |
| `current_states` | Extracted current state (8D) | Concatenate with current_vlm_output |
| `next_states` | Extracted next state (8D) | Concatenate with next_vlm_output |
| `current_vlm_outputs` | VLM outputs from base policy | Used in critic loss computation |
| `actions` | Actions taken | Used in critic loss computation |
| `rewards` | Rewards received | Used in target Q computation |
| `masks` | Masks (1 - done) | Used in target Q computation |
| `terminals` | Terminal flags | Additional info (optional) |
| `truncates` | Truncation flags | Additional info (optional) |

## Step 1: Generate the Cache

```bash
python scripts/dump_vlm_actions.py \
    --config=configs/your_config.py \
    --task_name="put both moka pots on the stove" \
    --replay_buffer_path="/path/to/replay_buffer/*.tfrecord" \
    --output_path="./critic_cache.pkl" \
    --num_actions_to_sample=4 \
    --seed=0
```

**Test with small dataset first:**
```bash
python scripts/dump_vlm_actions.py \
    --config=configs/your_config.py \
    --task_name="your_task" \
    --replay_buffer_path="/path/to/buffer/*.tfrecord" \
    --output_path="./test_critic_cache.pkl" \
    --max_batches=10 \
    --seed=0
```

## Step 2: Load and Use the Cache

### Basic Loading

```python
import pickle
import numpy as np
import jax.numpy as jnp

# Load the cache
with open('critic_cache.pkl', 'rb') as f:
    cache = pickle.load(f)

# Cache structure:
# - cache['episode_ids']: (N,) array of episode IDs
# - cache['episode_timesteps']: (N,) array of timesteps
# - cache['obs']: dict of observation arrays
# - cache['next_obs']: dict of next observation arrays
# - cache['current_vlm_outputs']: (N, vlm_dim) array
# - cache['actions']: (N, action_dim) array
# - cache['rewards']: (N,) array
# - cache['masks']: (N,) array
# - cache['terminals']: (N,) array
# - cache['truncates']: (N,) array
```

### Query Specific Episode/Timestep

```python
def get_cached_data(episode_id, timestep):
    """Get all cached data for a specific (episode_id, timestep)."""
    mask = (cache['episode_ids'] == episode_id) & (cache['episode_timesteps'] == timestep)
    indices = np.where(mask)[0]
    
    if len(indices) == 0:
        return None
    
    idx = indices[0]
    return {
        'obs': {k: v[idx] for k, v in cache['obs'].items()},
        'next_obs': {k: v[idx] for k, v in cache['next_obs'].items()},
        'current_vlm_output': cache['current_vlm_outputs'][idx],
        'actions': cache['actions'][idx],
        'rewards': cache['rewards'][idx],
        'masks': cache['masks'][idx],
    }

# Example usage
data = get_cached_data(episode_id=0, timestep=10)
```

### Batch Processing

```python
def get_batch_from_cache(indices):
    """Get a batch of data from cache by indices."""
    return {
        'obs': {k: v[indices] for k, v in cache['obs'].items()},
        'next_obs': {k: v[indices] for k, v in cache['next_obs'].items()},
        'current_vlm_outputs': cache['current_vlm_outputs'][indices],
        'actions': cache['actions'][indices],
        'rewards': cache['rewards'][indices],
        'masks': cache['masks'][indices],
    }

# Random batch
batch_size = 256
indices = np.random.choice(len(cache['episode_ids']), size=batch_size, replace=False)
batch = get_batch_from_cache(indices)
```

## Step 3: Use Cache with update_critic

Here's how to use the cached data with the `update_critic` function:

```python
import jax
from openpi import _model
from jaxrl_m.utils.timer_utils import Timer

# Load agent (assuming you have it set up)
# agent = ExpoPiLearner.create(...)

# Load cache
with open('critic_cache.pkl', 'rb') as f:
    cache = pickle.load(f)

# Create timer
timer = Timer()

# Sample random batch from cache
batch_size = 256
indices = np.random.choice(len(cache['episode_ids']), size=batch_size, replace=False)

# Extract batch data
obs_batch = {k: v[indices] for k, v in cache['obs'].items()}
next_obs_batch = {k: v[indices] for k, v in cache['next_obs'].items()}

# Convert to OpenPI Observation format
obs = _model.Observation.from_dict(obs_batch)
next_obs = _model.Observation.from_dict(next_obs_batch)

# Prepare batch dict with cached data
batch = {
    'current_vlm_output': cache['current_vlm_outputs'][indices],
    'actions': cache['actions'][indices],
    'rewards': cache['rewards'][indices],
    'masks': cache['masks'][indices],
}

# Call update_critic (with output_only_base_actions=True)
rng = jax.random.PRNGKey(0)
agent, info = agent.update_critic(
    obs=obs_batch,
    next_obs=next_obs_batch,
    batch=batch,
    seed=rng,
    timer=timer,
    output_only_base_actions=True,  # Use base policy actions only
)

print(f"Critic loss: {info['critic_loss']}")
print(f"Next Q values: {info['next_qs_mean']}")
```

## Step 4: Training Loop with Cache

```python
def train_critic_from_cache(agent, cache, num_updates=1000, batch_size=256):
    """Train critic using cached data."""
    from tqdm import tqdm
    import jax
    
    rng = jax.random.PRNGKey(0)
    timer = Timer()
    
    for step in tqdm(range(num_updates)):
        # Sample batch
        rng, sample_rng = jax.random.split(rng)
        indices = jax.random.choice(
            sample_rng, 
            len(cache['episode_ids']), 
            shape=(batch_size,), 
            replace=False
        )
        
        # Extract batch
        obs_batch = {k: v[indices] for k, v in cache['obs'].items()}
        next_obs_batch = {k: v[indices] for k, v in cache['next_obs'].items()}
        
        batch = {
            'current_vlm_output': cache['current_vlm_outputs'][indices],
            'actions': cache['actions'][indices],
            'rewards': cache['rewards'][indices],
            'masks': cache['masks'][indices],
        }
        
        # Update critic
        rng, update_rng = jax.random.split(rng)
        agent, info = agent.update_critic(
            obs=obs_batch,
            next_obs=next_obs_batch,
            batch=batch,
            seed=update_rng,
            timer=timer,
            output_only_base_actions=True,
        )
        
        # Log metrics
        if step % 100 == 0:
            print(f"Step {step}: critic_loss={info['critic_loss']:.4f}, "
                  f"next_qs_mean={info['next_qs_mean']:.4f}")
    
    return agent

# Usage
trained_agent = train_critic_from_cache(agent, cache, num_updates=1000)
```

## Implementation Details

### What happens in update_critic with cached data?

1. **Receives cached data:**
   - `obs` and `next_obs` (OpenPI format observations)
   - `batch['current_vlm_output']` (pre-computed VLM outputs)
   - `batch['actions']`, `batch['rewards']`, `batch['masks']`

2. **Computes next actions** (with `output_only_base_actions=True`):
   ```python
   next_actions, next_vlm_output = agent.sample_batch_actions(
       next_obs, 
       is_target=True, 
       output_only_base_actions=True,
       seed=rng
   )
   ```
   This returns base policy actions without editing (faster!)

3. **Computes target Q:**
   ```python
   next_qs = compute_q(target_critic, next_vlm_output, next_actions)
   target_q = rewards + discount * masks * next_qs
   ```

4. **Updates critic:**
   ```python
   grads, info = _critic_loss_and_grad(
       critic.params,
       current_vlm_output,  # From cache!
       actions,              # From cache!
       target_q,
       key,
       critic.apply_fn,
   )
   ```

### Why output_only_base_actions=True?

- **Faster**: Skips the editing step (local optimization)
- **Simpler**: Only uses base policy actions
- **Good for warmup**: Reliable actions for critic warmup phase
- **Matches assumption**: The cache was created with this assumption

## Benefits

✅ **Pre-compute once**: VLM forward passes are expensive  
✅ **Train critic offline**: No need for environment interaction  
✅ **Fast iteration**: Load from cache and train immediately  
✅ **Reproducible**: Same cached data across experiments  
✅ **Memory efficient**: Numpy arrays on disk  

## Notes

- Cache is indexed by `(episode_id, timestep)` for easy lookup
- All arrays are stored in numpy format for efficiency
- Metadata includes task info, seed, and data descriptions
- Compatible with EXPO-PI architecture
- Assumes 8D state (change if your env uses different dims)

## Troubleshooting

**Q: Cache file is too large**  
A: Use `--max_batches` to limit the amount of data cached

**Q: Out of memory when loading cache**  
A: Use memory-mapped arrays or load in batches

**Q: Data shapes don't match**  
A: Check that your config matches the one used to create the cache

**Q: VLM outputs look wrong**  
A: Verify the base policy is loaded correctly and using the right checkpoint

Happy training! 🚀




