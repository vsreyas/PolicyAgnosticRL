"""Example script demonstrating how to use the dumped VLM actions data."""

import numpy as np
from load_vlm_actions import VLMActionsLoader


def example_basic_usage():
    """Basic usage example."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Usage")
    print("="*80)
    
    # Load the data
    loader = VLMActionsLoader("vlm_actions_dump.pkl")
    
    # Get a specific next_action and next_vlm_output
    next_action, next_vlm_output = loader.get(episode_id=0, timestep=10)
    
    if next_action is not None:
        print(f"\nRetrieved data for episode 0, timestep 10:")
        print(f"  Next Action shape: {next_action.shape}")
        print(f"  Next VLM output shape: {next_vlm_output.shape}")
        print(f"  Next Action (first 5 dims): {next_action[:5]}")
    else:
        print("\nNo data found for episode 0, timestep 10")


def example_episode_processing():
    """Process all data for a specific episode."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Process Entire Episode")
    print("="*80)
    
    loader = VLMActionsLoader("vlm_actions_dump.pkl")
    
    # Get all episode IDs
    episode_ids = loader.get_all_episode_ids()
    print(f"\nFound {len(episode_ids)} unique episodes")
    
    if len(episode_ids) > 0:
        # Process first episode
        ep_id = int(episode_ids[0])
        timesteps, next_actions, next_vlm_outputs = loader.get_episode(ep_id)
        
        print(f"\nEpisode {ep_id}:")
        print(f"  Length: {len(timesteps)} timesteps")
        print(f"  Next Actions shape: {next_actions.shape}")
        print(f"  Next VLM outputs shape: {next_vlm_outputs.shape}")
        
        # Compute some statistics
        print(f"\nNext Action statistics:")
        print(f"  Mean: {np.mean(next_actions, axis=0)[:5]}... (first 5 dims)")
        print(f"  Std: {np.std(next_actions, axis=0)[:5]}... (first 5 dims)")
        print(f"  Min: {np.min(next_actions, axis=0)[:5]}... (first 5 dims)")
        print(f"  Max: {np.max(next_actions, axis=0)[:5]}... (first 5 dims)")


def example_batch_processing():
    """Process data in batches for all episodes."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Batch Processing All Episodes")
    print("="*80)
    
    loader = VLMActionsLoader("vlm_actions_dump.pkl")
    
    episode_ids = loader.get_all_episode_ids()
    
    all_next_actions = []
    all_next_vlm_outputs = []
    
    for ep_id in episode_ids[:5]:  # Process first 5 episodes
        timesteps, next_actions, next_vlm_outputs = loader.get_episode(int(ep_id))
        all_next_actions.append(next_actions)
        all_next_vlm_outputs.append(next_vlm_outputs)
        print(f"Episode {ep_id}: {len(timesteps)} timesteps")
    
    # Concatenate all next_actions
    if all_next_actions:
        all_next_actions_concat = np.concatenate(all_next_actions, axis=0)
        all_next_vlm_outputs_concat = np.concatenate(all_next_vlm_outputs, axis=0)
        
        print(f"\nCombined data shape:")
        print(f"  Next Actions: {all_next_actions_concat.shape}")
        print(f"  Next VLM outputs: {all_next_vlm_outputs_concat.shape}")


def example_filtering():
    """Filter and query specific data."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Filtering and Querying")
    print("="*80)
    
    loader = VLMActionsLoader("vlm_actions_dump.pkl")
    
    # Find episodes with length > 50
    episode_ids = loader.get_all_episode_ids()
    long_episodes = []
    
    for ep_id in episode_ids:
        length = loader.get_episode_length(int(ep_id))
        if length > 50:
            long_episodes.append((int(ep_id), length))
    
    print(f"\nFound {len(long_episodes)} episodes with length > 50")
    if long_episodes:
        print("First 5 long episodes:")
        for ep_id, length in long_episodes[:5]:
            print(f"  Episode {ep_id}: {length} timesteps")


def example_manual_indexing():
    """Manually work with the underlying arrays."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Manual Array Indexing")
    print("="*80)
    
    loader = VLMActionsLoader("vlm_actions_dump.pkl")
    
    # Access raw arrays
    episode_ids = loader.episode_ids
    timesteps = loader.episode_timesteps
    next_actions = loader.next_actions
    next_vlm_outputs = loader.next_vlm_outputs
    
    print(f"\nRaw data shapes:")
    print(f"  Episode IDs: {episode_ids.shape}")
    print(f"  Timesteps: {timesteps.shape}")
    print(f"  Next Actions: {next_actions.shape}")
    print(f"  Next VLM outputs: {next_vlm_outputs.shape}")
    
    # Find all timesteps for episode 0
    mask = episode_ids == 0
    ep0_timesteps = timesteps[mask]
    ep0_next_actions = next_actions[mask]
    
    if len(ep0_timesteps) > 0:
        print(f"\nEpisode 0 data:")
        print(f"  Timesteps: {ep0_timesteps[:10]}..." if len(ep0_timesteps) > 10 else f"  Timesteps: {ep0_timesteps}")
        print(f"  Next Actions shape: {ep0_next_actions.shape}")


def example_save_subset():
    """Save a subset of the data."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Save Subset of Data")
    print("="*80)
    
    import pickle
    
    loader = VLMActionsLoader("vlm_actions_dump.pkl")
    
    # Get data for first 3 episodes
    episode_ids = loader.get_all_episode_ids()[:3]
    
    subset_data = {
        'episode_ids': [],
        'episode_timesteps': [],
        'next_actions': [],
        'next_vlm_outputs': [],
    }
    
    for ep_id in episode_ids:
        timesteps, next_actions, next_vlm_outputs = loader.get_episode(int(ep_id))
        subset_data['episode_ids'].extend([int(ep_id)] * len(timesteps))
        subset_data['episode_timesteps'].extend(timesteps.tolist())
        subset_data['next_actions'].append(next_actions)
        subset_data['next_vlm_outputs'].append(next_vlm_outputs)
    
    # Convert to arrays
    subset_data['episode_ids'] = np.array(subset_data['episode_ids'])
    subset_data['episode_timesteps'] = np.array(subset_data['episode_timesteps'])
    subset_data['next_actions'] = np.concatenate(subset_data['next_actions'], axis=0)
    subset_data['next_vlm_outputs'] = np.concatenate(subset_data['next_vlm_outputs'], axis=0)
    subset_data['metadata'] = loader.metadata.copy()
    subset_data['metadata']['subset'] = True
    subset_data['metadata']['num_episodes'] = len(episode_ids)
    
    # Save subset
    output_path = "vlm_actions_subset.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(subset_data, f)
    
    print(f"\nSaved subset to {output_path}")
    print(f"  Episodes: {len(episode_ids)}")
    print(f"  Total entries: {len(subset_data['episode_ids'])}")


def main():
    """Run all examples."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python example_usage.py <path_to_vlm_actions.pkl>")
        print("\nThis script demonstrates various ways to use the dumped VLM actions data.")
        sys.exit(1)
    
    # Update the path in the examples
    data_path = sys.argv[1]
    
    # Replace hardcoded path with provided path
    import load_vlm_actions
    original_init = load_vlm_actions.VLMActionsLoader.__init__
    
    def new_init(self, path=None):
        if path is None:
            path = data_path
        original_init(self, path)
    
    load_vlm_actions.VLMActionsLoader.__init__ = new_init
    
    try:
        example_basic_usage()
        example_episode_processing()
        example_batch_processing()
        example_filtering()
        example_manual_indexing()
        # example_save_subset()  # Uncomment if you want to save a subset
        
        print("\n" + "="*80)
        print("All examples completed successfully!")
        print("="*80 + "\n")
        
    except FileNotFoundError:
        print(f"\nError: Could not find data file at {data_path}")
        print("Please run dump_vlm_actions.py first to generate the data.")
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

