"""Utility script to load and query VLM actions dumped by dump_vlm_actions.py"""

import pickle
from typing import Optional, Tuple

import numpy as np


class VLMActionsLoader:
    """Utility class to load and query VLM actions data."""
    
    def __init__(self, data_path: str):
        """
        Initialize the loader with a path to dumped VLM actions.
        
        Args:
            data_path: Path to the pickle file containing dumped VLM actions
        """
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        self.episode_ids = self.data['episode_ids']
        self.episode_timesteps = self.data['episode_timesteps']
        self.next_actions = self.data['next_actions']
        self.next_vlm_outputs = self.data['next_vlm_outputs']
        self.metadata = self.data['metadata']
        
        # Create index for faster lookup
        self._build_index()
        
        print(f"Loaded VLM actions data:")
        print(f"  Total entries: {len(self.episode_ids)}")
        print(f"  Next Actions shape: {self.next_actions.shape}")
        print(f"  Next VLM outputs shape: {self.next_vlm_outputs.shape}")
        print(f"  Metadata: {self.metadata}")
    
    def _build_index(self):
        """Build an index mapping (episode_id, timestep) -> array index for fast lookup."""
        self.index = {}
        for i, (ep_id, timestep) in enumerate(zip(self.episode_ids, self.episode_timesteps)):
            key = (int(ep_id), int(timestep))
            self.index[key] = i
    
    def get(self, episode_id: int, timestep: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Retrieve next_action and next_vlm_output for a specific episode and timestep.
        
        Args:
            episode_id: The episode ID
            timestep: The timestep within the episode
            
        Returns:
            Tuple of (next_action, next_vlm_output), or (None, None) if not found
        """
        key = (episode_id, timestep)
        if key in self.index:
            idx = self.index[key]
            return self.next_actions[idx], self.next_vlm_outputs[idx]
        return None, None
    
    def get_episode(self, episode_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieve all next_actions and next_vlm_outputs for a specific episode.
        
        Args:
            episode_id: The episode ID
            
        Returns:
            Tuple of (timesteps, next_actions, next_vlm_outputs) for the episode
        """
        mask = self.episode_ids == episode_id
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            return np.array([]), np.array([]), np.array([])
        
        timesteps = self.episode_timesteps[indices]
        next_actions = self.next_actions[indices]
        next_vlm_outputs = self.next_vlm_outputs[indices]
        
        # Sort by timestep
        sort_idx = np.argsort(timesteps)
        return timesteps[sort_idx], next_actions[sort_idx], next_vlm_outputs[sort_idx]
    
    def get_all_episode_ids(self) -> np.ndarray:
        """Get list of all unique episode IDs."""
        return np.unique(self.episode_ids)
    
    def get_episode_length(self, episode_id: int) -> int:
        """Get the number of timesteps for a specific episode."""
        mask = self.episode_ids == episode_id
        return int(np.sum(mask))
    
    def __len__(self) -> int:
        """Return total number of entries."""
        return len(self.episode_ids)
    
    def __repr__(self) -> str:
        return (f"VLMActionsLoader(entries={len(self)}, "
                f"episodes={len(self.get_all_episode_ids())}, "
                f"task='{self.metadata.get('task_name', 'unknown')}')")


def main():
    """Example usage of VLMActionsLoader."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python load_vlm_actions.py <path_to_vlm_actions.pkl>")
        sys.exit(1)
    
    data_path = sys.argv[1]
    
    # Load data
    loader = VLMActionsLoader(data_path)
    print("\n" + "="*80)
    print(loader)
    print("="*80)
    
    # Get all episode IDs
    episode_ids = loader.get_all_episode_ids()
    print(f"\nFound {len(episode_ids)} unique episodes")
    print(f"Episode IDs (first 10): {episode_ids[:10]}")
    
    # Example: Get data for first episode
    if len(episode_ids) > 0:
        ep_id = int(episode_ids[0])
        print(f"\n--- Example: Episode {ep_id} ---")
        timesteps, next_actions, next_vlm_outputs = loader.get_episode(ep_id)
        print(f"Episode length: {len(timesteps)} timesteps")
        print(f"Timesteps: {timesteps[:5]}..." if len(timesteps) > 5 else f"Timesteps: {timesteps}")
        print(f"Next Actions shape: {next_actions.shape}")
        print(f"Next VLM outputs shape: {next_vlm_outputs.shape}")
        
        # Get specific timestep
        if len(timesteps) > 0:
            ts = int(timesteps[0])
            next_action, next_vlm_out = loader.get(ep_id, ts)
            print(f"\n--- Example: Episode {ep_id}, Timestep {ts} ---")
            print(f"Next Action shape: {next_action.shape if next_action is not None else 'Not found'}")
            print(f"Next VLM output shape: {next_vlm_out.shape if next_vlm_out is not None else 'Not found'}")


if __name__ == "__main__":
    main()

