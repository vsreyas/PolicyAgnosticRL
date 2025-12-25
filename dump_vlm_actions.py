"""Script to dump VLM outputs and batch data for critic updates.

This script caches all data needed for update_critic function:
- Current observations (for extracting state)
- Next observations (for computing next actions/VLM outputs)
- Current VLM outputs (from base policy)
- Actions, rewards, masks from batch
- Episode IDs and timesteps for indexing
"""

import os
import pickle
from typing import Dict, List

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from absl import app, flags, logging
from ml_collections import config_flags
from tqdm import tqdm

from jaxrl_m.agents.continuous.expo_pi import ExpoPiLearner
from jaxrl_m.data.image_replay_buffer_pi import ImageReplayBufferPi
from jaxrl_m.data.bridge_dataset import glob_to_path_list
from jaxrl_m.utils.timer_utils import Timer
from openpi.training.config import get_config
import openpi.models.model as _model

FLAGS = flags.FLAGS

flags.DEFINE_string("environment_name", "libero", "Environment name.")
config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string(
    "task_name",
    "put both moka pots on the stove",
    "Name of fixed task"
)
flags.DEFINE_bool("use_wrist_view", True, "Use Wrist view camera.")
flags.DEFINE_bool("use_lang", True, "Use language conditioning.")
flags.DEFINE_integer(
    "num_actions_to_sample", 4, "Number of actions to sample for the policy."
)
flags.DEFINE_integer(
    "num_edit_samples", 4, "Number of edit samples to use for editing the actions."
)
flags.DEFINE_string(
    "output_path", "./critic_update_cache.pkl", "Path to save the cached data for critic updates."
)
flags.DEFINE_string(
    "replay_buffer_path", "", "Path to replay buffer tfrecords (can be a glob pattern)."
)
flags.DEFINE_integer(
    "max_batches", None, "Maximum number of batches to process (None for all)."
)


def main(_):
    # Prevent TensorFlow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    # Setup devices
    devices = jax.local_devices()
    num_devices = len(devices)
    logging.info(f"Found {num_devices} JAX devices")
    
    assert FLAGS.config.batch_size % num_devices == 0, \
        f"Batch size {FLAGS.config.batch_size} must be divisible by num_devices {num_devices}"

    # Get PI config
    pi_config = get_config("pi05_libero_custom_low_mem")
    pi_config.fsdp_devices = 2
    pi_config.exp_name = "dump_vlm_actions"
    pi_config.overwrite = True

    # Load dataset
    logging.info("Loading dataset...")
    if FLAGS.environment_name == "libero":
        from jaxrl_m.envs.libero import get_libero_tfrecord_dataset
        
        # Use replay buffer path if provided, otherwise use config path
        if FLAGS.replay_buffer_path:
            data_paths = glob_to_path_list(FLAGS.replay_buffer_path)
            logging.info(f"Found {len(data_paths)} tfrecord files from replay buffer")
            
            # Create replay buffer iterator
            replay_buffer = ImageReplayBufferPi(
                data_paths=data_paths,
                seed=FLAGS.seed,
                train=False,
                task_name=FLAGS.task_name,
                use_wrist_view=FLAGS.use_wrist_view,
                use_language=FLAGS.use_lang,
                config=pi_config,
                final_step_sparse_reward=False,
                **FLAGS.config.image_replay_buffer_kwargs,
            )
            dataset_iterator = replay_buffer.iterator(
                batch_size=FLAGS.config.agent_kwargs.batch_size
            )
        else:
            # Load from offline dataset
            dataset = get_libero_tfrecord_dataset(
                tfrecord_regexp=FLAGS.config.libero_tfrecord_regexp,
                use_wrist_view=FLAGS.use_wrist_view,
                use_language=FLAGS.use_lang,
                config=pi_config,
                is_pi=True,
                task_name=FLAGS.task_name,
                train=False,
                final_step_sparse_reward=True,
                **FLAGS.config.dataset_kwargs,
            )
            dataset_iterator = dataset.iterator(
                batch_size=FLAGS.config.agent_kwargs.batch_size
            )
    else:
        raise NotImplementedError(f"Environment {FLAGS.environment_name} not supported")

    # Get example batch to initialize agent
    logging.info("Getting example batch...")
    example_batch = next(dataset_iterator)

    # breakpoint()
    
    # Create EXPO agent
    logging.info("Creating EXPO agent...")
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, construct_rng = jax.random.split(rng)
    
    agent = ExpoPiLearner.create(
        config=pi_config,
        seed=FLAGS.seed,
        observations=example_batch,
        rng=construct_rng,
        N=FLAGS.num_actions_to_sample,
        n_edit_samples=FLAGS.num_edit_samples,
    )
    
    logging.info("Agent created successfully")
    del example_batch

    # Reset data iterator
    if FLAGS.replay_buffer_path:
        dataset_iterator = replay_buffer.iterator(
            batch_size=FLAGS.config.agent_kwargs.batch_size
        )
    else:
        dataset_iterator = dataset.iterator(
            batch_size=FLAGS.config.agent_kwargs.batch_size
        )
    
    # Create timer for performance tracking
    timer = Timer()

    # Data structure to store results
    # Format: {(episode_id, episode_timestep): {all data needed for update_critic}}
    critic_cache_data = {}
    
    # Iterate through all batches
    logging.info("Processing batches...")
    batch_count = 0
    
    try:
        # Reset iterator by creating a new one
        if FLAGS.replay_buffer_path:
            dataset_iterator = replay_buffer.iterator(
                batch_size=FLAGS.config.agent_kwargs.batch_size
            )
        else:
            dataset_iterator = dataset.iterator(
                batch_size=FLAGS.config.agent_kwargs.batch_size
            )
        
        with tqdm(desc="Processing batches") as pbar:
            while True:
                if FLAGS.max_batches and batch_count >= FLAGS.max_batches:
                    logging.info(f"Reached max_batches limit: {FLAGS.max_batches}")
                    break
                
                try:
                    batch = next(dataset_iterator)
                except StopIteration:
                    logging.info("Reached end of dataset")
                    break
                    
                # breakpoint()
                
                batch_size = batch['actions'].shape[0]
                
                # Preprocess batch (similar to train_expo_pi.py)
                batch_processed = agent.preproess_batch(batch.copy())
                
                # Convert observations to OpenPI format
                observations = agent.actor.convert_to_openpi_format_infer(
                    batch_processed, obs_key="observations"
                )
                obs = agent.actor.input_data_transforms(observations)
                
                next_observations_dict = agent.actor.convert_to_openpi_format_infer(
                    batch_processed, obs_key="next_observations"
                )
                next_obs = agent.actor.input_data_transforms(next_observations_dict)
                
                # Extract states
                current_state = obs['state'][:, :8]  # State dimension (8D)
                next_state = next_obs['state'][:, :8]
                
                # Sample VLM outputs from current observations using base policy
                # This is what batch['current_vlm_output'] would contain
                rng, sample_rng = jax.random.split(rng)
                _obs = _model.Observation.from_dict(obs)
                _next_obs = _model.Observation.from_dict(next_obs)
                
                # Sample actions and VLM output from base policy (actor)
                # Using output_only_base_actions=True paradigm
                # timer.tick("sample_current_vlm_output_time")
                # _, current_vlm_output_full = agent.actor.sample_actions_with_vlm_output(
                #     rng=sample_rng,
                #     observation=_obs,
                #     timer=timer,
                # )
                # timer.tock("sample_current_vlm_output_time")
                next_actions, next_vlm_output = agent.sample_batch_actions(_next_obs, is_target=True, return_first_action=False, timer=timer, output_only_base_actions=True, output_all_sampled_actions=True, seed=sample_rng)

                rng, sample_rng = jax.random.split(rng)
                _, current_vlm_output = agent.sample_batch_actions(_obs, is_target=True, return_first_action=False, timer=timer, output_only_base_actions=True, output_all_sampled_actions=True, seed=sample_rng)
                
                current_actions = batch['actions'][:, :, :7].reshape(batch_size, 1, pi_config.model.action_horizon, 7)
                # breakpoint()

                # Extract VLM output (mean across tokens)
                # current_vlm_output = jnp.mean(current_vlm_output_full[0][:, :512, :], axis=1)
                # next_vlm_output = jnp.mean(next_vlm_output_full[0][:, :512, :], axis=1)
                
                # Convert to numpy
                current_vlm_output = jax.device_get(current_vlm_output)
                current_state = jax.device_get(current_state)
                next_state = jax.device_get(next_state)
                # breakpoint()
                
                # Store all data needed for update_critic
                for i in range(batch_size):
                    # Try to get episode_id and timestep from batch
                    # If not available, use batch_count and index within batch
                    episode_id = batch.get('episode_id', np.array([batch_count] * batch_size))[i]
                    episode_timestep = int(
                        batch.get("episode_timestep", batch.get("timesteps", np.arange(batch_size)))[i]
                    )
                    
                    # Convert to python int for dictionary key
                    episode_id = int(episode_id)
                    episode_timestep = int(episode_timestep)
                    
                    key = (episode_id, episode_timestep)
                    critic_cache_data[key] = {
                        # Observations (needed for sample_batch_actions in update_critic)
                        'obs': {k: v[i] for k, v in obs.items() if 'image' not in k},
                        'next_obs': {k: v[i] for k, v in next_obs.items() if 'image' not in k},
                        # States (extracted for convenience)
                        'current_state': current_state[i],
                        'next_state': next_state[i],
                        # VLM output from base policy
                        'current_actions': current_actions[i],
                        'current_vlm_output': current_vlm_output[i],
                        # Batch data needed for critic update
                        'actions': batch['actions'][i],
                        'rewards': batch['rewards'][i],
                        'masks': batch['masks'][i],
                        # Optional: terminals and truncates if available
                        'terminals': batch['terminals'][i],
                        'truncates': batch['truncates'][i],
                        'next_actions': next_actions[i],
                        'next_vlm_output': next_vlm_output[i],
                        'mc_returns': batch['mc_returns'][i],
                    }
                
                batch_count += 1
                pbar.update(1)
                pbar.set_postfix({'batches': batch_count, 'entries': len(critic_cache_data)})
                
                # Periodically log timing info
                if batch_count % 10 == 0:
                    logging.info(f"Timing info (last 10 batches): {timer.get_average_times()}")
                
    except Exception as e:
        logging.error(f"Error during processing: {e}")
        raise
    
    logging.info(f"Processed {batch_count} batches, collected {len(critic_cache_data)} entries")
    
    # Convert to more efficient storage format
    # Create separate arrays for all data needed by update_critic
    logging.info("Converting to numpy arrays...")

    # breakpoint()
    
    keys_list = sorted(critic_cache_data.keys())
    episode_ids = np.array([k[0] for k in keys_list], dtype=np.int32)
    episode_timesteps = np.array([k[1] for k in keys_list], dtype=np.int32)
    
    # Stack all observation data
    obs_dict = {}
    next_obs_dict = {}
    for key in critic_cache_data[keys_list[0]]['obs'].keys():
        obs_dict[key] = np.stack([critic_cache_data[k]['obs'][key] for k in keys_list])
        next_obs_dict[key] = np.stack([critic_cache_data[k]['next_obs'][key] for k in keys_list])
    
    current_states = np.stack([critic_cache_data[k]['current_state'] for k in keys_list])
    next_states = np.stack([critic_cache_data[k]['next_state'] for k in keys_list])
    current_actions = np.stack([critic_cache_data[k]['current_actions'] for k in keys_list])
    current_vlm_outputs = np.stack([critic_cache_data[k]['current_vlm_output'] for k in keys_list])
    next_actions = np.stack([critic_cache_data[k]['next_actions'] for k in keys_list])
    next_vlm_outputs = np.stack([critic_cache_data[k]['next_vlm_output'] for k in keys_list])
    actions = np.stack([critic_cache_data[k]['actions'] for k in keys_list])
    rewards = np.stack([critic_cache_data[k]['rewards'] for k in keys_list])
    masks = np.stack([critic_cache_data[k]['masks'] for k in keys_list])
    terminals = np.stack([critic_cache_data[k]['terminals'] for k in keys_list])
    truncates = np.stack([critic_cache_data[k]['truncates'] for k in keys_list])
    mc_returns = np.stack([critic_cache_data[k]['mc_returns'] for k in keys_list])

    # breakpoint()
    
    # Create final data structure
    final_data = {
        'episode_ids': episode_ids,
        'episode_timesteps': episode_timesteps,
        'obs': obs_dict,
        'next_obs': next_obs_dict,
        'current_states': current_states,
        'current_actions': current_actions,
        'next_states': next_states,
        'current_vlm_outputs': current_vlm_outputs,
        'next_actions': next_actions,
        'next_vlm_outputs': next_vlm_outputs,
        'actions': actions,
        'rewards': rewards,
        'masks': masks,
        'terminals': terminals,
        'truncates': truncates,
        'mc_returns': mc_returns,
        'metadata': {
            'num_entries': len(keys_list),
            'num_batches_processed': batch_count,
            'task_name': FLAGS.task_name,
            'seed': FLAGS.seed,
            'num_actions_sampled': FLAGS.num_actions_to_sample,
            'note': 'Data cached for update_critic with output_only_base_actions=True',
            'data_description': {
                'obs': 'Current observations (for sample_batch_actions)',
                'next_obs': 'Next observations (for sample_batch_actions)',
                'current_states': 'Extracted current state (8D)',
                'next_states': 'Extracted next state (8D)',
                'current_vlm_outputs': 'VLM outputs from base policy on current obs',
                'actions': 'Actions taken (from batch)',
                'rewards': 'Rewards (from batch)',
                'masks': 'Masks for bootstrapping (1 - done)',
                'terminals': 'Terminal flags (done)',
                'truncates': 'Truncation flags',
                'mc_returns': 'MC returns',
            }
        }
    }
    
    # Save to disk
    logging.info(f"Saving to {FLAGS.output_path}...")
    output_dir = os.path.dirname(FLAGS.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(FLAGS.output_path, 'wb') as f:
        pickle.dump(final_data, f)
    
    logging.info(f"Successfully saved critic update cache to {FLAGS.output_path}")
    logging.info(f"Data shapes:")
    logging.info(f"  Observations: {obs_dict['state'].shape if 'state' in obs_dict else 'N/A'}")
    logging.info(f"  Current VLM outputs: {current_vlm_outputs.shape}")
    logging.info(f"  Actions: {actions.shape}")
    logging.info(f"  Rewards: {rewards.shape}")
    logging.info(f"  Masks: {masks.shape}")
    logging.info(f"  MC returns: {mc_returns.shape}")
    # Print usage example
    print("\n" + "="*80)
    print("CRITIC UPDATE CACHE SAVED SUCCESSFULLY!")
    print("="*80)
    print(f"\nTo load and use this data for critic updates:")
    print(f"""
import pickle
import numpy as np
import jax.numpy as jnp

# Load data
with open('{FLAGS.output_path}', 'rb') as f:
    data = pickle.load(f)

episode_ids = data['episode_ids']
episode_timesteps = data['episode_timesteps']

# Get all data for a specific episode and timestep
def get_data_for_critic_update(episode_id, timestep):
    mask = (episode_ids == episode_id) & (episode_timesteps == timestep)
    indices = np.where(mask)[0]
    if len(indices) > 0:
        idx = indices[0]
        return {{
            'obs': {{k: v[idx] for k, v in data['obs'].items()}},
            'next_obs': {{k: v[idx] for k, v in data['next_obs'].items()}},
            'current_vlm_output': data['current_vlm_outputs'][idx],
            'actions': data['actions'][idx],
            'rewards': data['rewards'][idx],
            'masks': data['masks'][idx],
        }}
    return None

# Example: Get data for update_critic
critic_data = get_data_for_critic_update(episode_id=0, timestep=10)

if critic_data:
    # Now you can call update_critic with this cached data:
    # - Use critic_data['next_obs'] to compute next_actions (with output_only_base_actions=True)
    # - Use critic_data['current_vlm_output'], actions, rewards, masks for critic loss
    print("Retrieved data for critic update")
    """)
    print("="*80)


if __name__ == "__main__":
    app.run(main)

