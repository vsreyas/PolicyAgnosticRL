# VLM Actions Dumping Scripts

This directory contains scripts for dumping and loading VLM actions from offline datasets or replay buffers.

## Files

- **`dump_vlm_actions.py`**: Main script to sample trajectories, iterate through batches, sample N actions per batch, get VLM outputs, and dump to disk
- **`load_vlm_actions.py`**: Utility script to load and query the dumped VLM actions data

## Usage

### 1. Dump VLM Actions

The `dump_vlm_actions.py` script processes a dataset (offline or replay buffer), samples actions using the VLM, and saves them to disk.

```bash
python scripts/dump_vlm_actions.py \
    --config=path/to/config.py \
    --task_name="put both moka pots on the stove" \
    --replay_buffer_path="/path/to/replay_buffer/*.tfrecord" \
    --output_path="./vlm_actions_dump.pkl" \
    --num_actions_to_sample=4 \
    --num_edit_samples=4 \
    --seed=0
```

**Key Arguments:**
- `--config`: Path to training configuration file (required)
- `--task_name`: Name of the task to process
- `--replay_buffer_path`: Path to replay buffer tfrecords (glob pattern supported). If not provided, uses offline dataset from config
- `--output_path`: Where to save the dumped actions (default: `./vlm_actions_dump.pkl`)
- `--num_actions_to_sample`: Number of actions to sample per observation (default: 4)
- `--num_edit_samples`: Number of edit samples for action editing (default: 4)
- `--max_batches`: Optional limit on number of batches to process
- `--use_wrist_view`: Whether to use wrist view camera (default: True)
- `--use_lang`: Whether to use language conditioning (default: True)

### 2. Load and Query VLM Actions

The dumped data is saved as a pickle file with the following structure:

```python
{
    'episode_ids': np.ndarray,        # Shape: (N,) - episode IDs
    'episode_timesteps': np.ndarray,  # Shape: (N,) - timesteps within episodes
    'next_actions': np.ndarray,       # Shape: (N, action_dim) - sampled actions for NEXT observations
    'next_vlm_outputs': np.ndarray,   # Shape: (N, vlm_output_dim) - VLM outputs for next observations
    'metadata': dict                  # Metadata about the dump
}
```

**Note**: The script dumps `next_actions` (actions for next observations) rather than current actions, since current actions are already available in the batch data.

#### Using the VLMActionsLoader class:

```python
from scripts.load_vlm_actions import VLMActionsLoader

# Load data
loader = VLMActionsLoader("vlm_actions_dump.pkl")

# Get next_action and next_vlm_output for specific episode and timestep
next_action, next_vlm_output = loader.get(episode_id=0, timestep=10)

# Get all data for an episode
timesteps, next_actions, next_vlm_outputs = loader.get_episode(episode_id=0)

# Get all unique episode IDs
episode_ids = loader.get_all_episode_ids()

# Get episode length
length = loader.get_episode_length(episode_id=0)
```

#### Manual loading:

```python
import pickle
import numpy as np

# Load data
with open('vlm_actions_dump.pkl', 'rb') as f:
    data = pickle.load(f)

episode_ids = data['episode_ids']
episode_timesteps = data['episode_timesteps']
next_actions = data['next_actions']  # Actions for NEXT observations
next_vlm_outputs = data['next_vlm_outputs']

# Query by episode_id and timestep
def get_next_action_for_episode_step(episode_id, timestep):
    mask = (episode_ids == episode_id) & (episode_timesteps == timestep)
    indices = np.where(mask)[0]
    if len(indices) > 0:
        idx = indices[0]
        return next_actions[idx], next_vlm_outputs[idx]
    return None, None

# Example: Get action for NEXT observation at (episode_id=0, timestep=10)
next_action, next_vlm_out = get_next_action_for_episode_step(episode_id=0, timestep=10)
```

#### Command-line inspection:

```bash
python scripts/load_vlm_actions.py vlm_actions_dump.pkl
```

This will print summary statistics and example queries.

## Implementation Details

### dump_vlm_actions.py

1. **Setup**: Loads configuration, initializes EXPO agent with VLM
2. **Dataset Loading**: Creates iterator from either:
   - Offline dataset (using `libero_tfrecord_regexp` from config)
   - Replay buffer (using `--replay_buffer_path` flag)
3. **Batch Processing**: Iterates through all batches with progress bar
4. **Action Sampling**: For each batch:
   - Extracts **next_observations** (not current observations, as current actions are already in batch)
   - Samples N actions using `agent.sample_actions()` for next observations
   - Gets VLM outputs for next observations
   - Stores with (episode_id, timestep) as key
5. **Data Conversion**: Converts dictionary to numpy arrays for efficient storage
6. **Saving**: Dumps to pickle file with metadata

### load_vlm_actions.py

- Provides `VLMActionsLoader` class for convenient data access
- Builds internal index for O(1) lookup by (episode_id, timestep)
- Supports querying individual timesteps or entire episodes
- Includes command-line tool for data inspection

## Notes

- The script handles both offline datasets and online replay buffers
- Episode IDs and timesteps are extracted from batch data if available, otherwise inferred from batch indices
- Data is stored in a memory-efficient format using numpy arrays
- The loader class builds an index for fast lookups
- Progress is shown with tqdm progress bar during processing

