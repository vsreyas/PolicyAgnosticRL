# Changes Summary: Next Actions Support

## Overview
Updated all scripts to dump and load **next_actions** (actions for next observations) instead of current actions, since current actions are already available in the batch data.

## Modified Files

### 1. `dump_vlm_actions.py`
**Key Changes:**
- Now extracts `next_observations` from batch instead of current observations
- Constructs next_observations dict by finding keys with `next_` prefix and removing the prefix
- Samples actions using `agent.sample_actions()` on next_observations
- Stores data as `next_actions` and `next_vlm_output` instead of `actions` and `vlm_output`
- Updated all variable names throughout: `next_actions`, `next_vlm_output`, `next_actions_array`, `next_vlm_outputs_array`
- Added note in metadata: `'note': 'Actions are for NEXT observations (current actions already in batch)'`

**Data Structure Output:**
```python
{
    'episode_ids': np.ndarray,
    'episode_timesteps': np.ndarray,
    'next_actions': np.ndarray,        # NEW: Actions for next observations
    'next_vlm_outputs': np.ndarray,    # NEW: VLM outputs for next observations
    'metadata': dict
}
```

### 2. `load_vlm_actions.py`
**Key Changes:**
- Updated `VLMActionsLoader` class to load `next_actions` and `next_vlm_outputs`
- Updated `__init__` to access correct keys from pickle file
- Updated `get()` method to return `(next_action, next_vlm_output)`
- Updated `get_episode()` method to return `(timesteps, next_actions, next_vlm_outputs)`
- Updated all docstrings and print statements to reflect next_actions
- Updated example in `main()` function

### 3. `example_usage.py`
**Key Changes:**
- Updated all example functions to use `next_actions` and `next_vlm_outputs`
- Updated variable names in:
  - `example_basic_usage()`: `next_action`, `next_vlm_output`
  - `example_episode_processing()`: `next_actions`, `next_vlm_outputs`
  - `example_batch_processing()`: `all_next_actions`, `all_next_vlm_outputs`
  - `example_manual_indexing()`: `next_actions`, `next_vlm_outputs`
  - `example_save_subset()`: Updated subset data structure

### 4. `README.md`
**Key Changes:**
- Added note explaining that next_actions are dumped (not current actions)
- Updated data structure documentation
- Updated all code examples to use `next_actions` and `next_vlm_outputs`
- Updated implementation details to clarify next_observations are extracted

## Rationale

**Why dump next_actions instead of current actions?**
- Current actions are already available in the batch data (in the `'actions'` key)
- Next actions are not available and need to be sampled from the VLM
- This makes the dumped data more useful by providing information that's not already present
- Aligns with typical RL workflows where you want to evaluate what action the policy would take at the next state

## Usage Example

```python
from scripts.load_vlm_actions import VLMActionsLoader

# Load data
loader = VLMActionsLoader("vlm_actions_dump.pkl")

# Get next action for specific episode and timestep
# This returns what action the VLM would take at the NEXT observation
# after the observation at (episode_id=0, timestep=10)
next_action, next_vlm_output = loader.get(episode_id=0, timestep=10)

# Get all next actions for an episode
timesteps, next_actions, next_vlm_outputs = loader.get_episode(episode_id=0)
```

## Backward Compatibility

⚠️ **Breaking Change**: This is a breaking change from the original implementation.
- Old pickle files with `'actions'` and `'vlm_outputs'` keys will NOT work with updated loader
- New pickle files with `'next_actions'` and `'next_vlm_outputs'` keys will NOT work with old loader
- If you have existing dumps, you'll need to regenerate them with the updated script

## Testing

Before running on full dataset, test with a small batch:

```bash
python scripts/dump_vlm_actions.py \
    --config=configs/your_config.py \
    --task_name="your_task" \
    --replay_buffer_path="/path/to/buffer/*.tfrecord" \
    --output_path="./test_dump.pkl" \
    --max_batches=10 \
    --seed=0

# Then verify:
python scripts/load_vlm_actions.py ./test_dump.pkl
```




