# Quick Start: Critic Update Cache

## TL;DR

Generate a cache of all data needed for `update_critic` (VLM outputs, observations, actions, rewards, masks), then train the critic efficiently using the cache.

## 1. Generate Cache (One-time)

```bash
python scripts/dump_vlm_actions.py \
    --config=configs/your_config.py \
    --task_name="put both moka pots on the stove" \
    --replay_buffer_path="/path/to/replay_buffer/*.tfrecord" \
    --output_path="./critic_cache.pkl" \
    --seed=0
```

**Test first:**
```bash
python scripts/dump_vlm_actions.py \
    --config=configs/your_config.py \
    --task_name="your_task" \
    --output_path="./test_cache.pkl" \
    --max_batches=10 \
    --seed=0
```

## 2. Use Cache for Critic Training

```python
import pickle
import jax
import numpy as np
from openpi import _model
from jaxrl_m.utils.timer_utils import Timer

# Load cache
with open('critic_cache.pkl', 'rb') as f:
    cache = pickle.load(f)

# Training loop
rng = jax.random.PRNGKey(0)
timer = Timer()
batch_size = 256

for step in range(1000):
    # Sample batch from cache
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
        output_only_base_actions=True,  # IMPORTANT!
    )
    
    if step % 100 == 0:
        print(f"Step {step}: loss={info['critic_loss']:.4f}")
```

## 3. What's in the Cache?

```python
cache = {
    'episode_ids': (N,),           # Episode IDs
    'episode_timesteps': (N,),     # Timesteps
    'obs': dict,                    # Current observations
    'next_obs': dict,               # Next observations
    'current_vlm_outputs': (N, D),  # Pre-computed VLM outputs ← KEY!
    'actions': (N, A),              # Actions
    'rewards': (N,),                # Rewards
    'masks': (N,),                  # Masks (1 - done)
}
```

## Key Points

✅ **output_only_base_actions=True**: Always use this flag  
✅ **Pre-computed VLM outputs**: Cached once, used many times  
✅ **Fast critic training**: No VLM recomputation  
✅ **Simple workflow**: Generate → Train  

## Documentation

- **CRITIC_CACHE_USAGE.md**: Detailed guide with examples
- **FINAL_SUMMARY.md**: Implementation details
- **train_expo_pi.py**: Reference code

## Command-Line Args

```bash
--config: Training config file (required)
--task_name: Task name (default: "put both moka pots on the stove")
--replay_buffer_path: Glob pattern to tfrecords (optional)
--output_path: Where to save cache (default: "./critic_update_cache.pkl")
--num_actions_to_sample: N for sampling (default: 4)
--max_batches: Limit processing (default: None = all)
--seed: Random seed (default: 0)
--use_wrist_view: Use wrist camera (default: True)
--use_lang: Use language conditioning (default: True)
```

## Troubleshooting

**Cache too large?**  
→ Use `--max_batches=100` to limit

**Out of memory?**  
→ Reduce batch size or load cache in chunks

**Data shapes don't match?**  
→ Check config matches cache generation config

Done! 🚀




