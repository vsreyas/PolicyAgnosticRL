
import os
import pickle
import time
import cv2
from typing import Dict

import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
import tensorflow as tf
import wandb
from absl import app, flags, logging
from ml_collections import config_flags
from tqdm import tqdm
from matplotlib import pyplot as plt
import io
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Any


from jaxrl_m.agents.continuous.expo_pi import ExpoPiLearner
from jaxrl_m.common.wandb import WandBLogger
from jaxrl_m.utils.timer_utils import Timer
from openpi.training.config import get_config
from jaxrl_m.envs.libero import (
    get_libero_config,
    get_libero_env,
    get_libero_tfrecord_dataset,
)
from jaxrl_m.common.evaluation import evaluate_with_trajectories_libero, save_rollout_gif, supply_rng
from jaxrl_m.agents.continuous.expo_pi import ExpoPiLearner, compute_q, compute_q_all
# from train_expo_pi import get_policy_fn
from jaxrl_m.common.traj import calc_return_to_go
from jaxrl_m.common.typing import Batch, Data

try:
    from jax_smi import initialise_tracking  # type: ignore
    initialise_tracking()
except ImportError:
    pass

def filter_cache(cache: Dict) -> Dict:
    """Filter the cache to only include entries where the current state is not None."""
    # import pickle
    # import pandas as pd
    # import numpy as np
    # file=open("outputs/vlm_actions_replay_ep30_v1_clean_v3.pkl","rb"); cache=pickle.load(file);

    # Find keys
    action_horizon = cache['current_actions'].shape[2]
    episode_ids = cache['episode_ids']
    episode_timesteps = cache['episode_timesteps']
    terminals = cache['terminals']
    df = pd.DataFrame({
        'episode_id': episode_ids,
        'episode_timestep': episode_timesteps,
        'terminals': terminals,
    })

    pos = df.groupby("episode_id").cumcount()
    ep_len = df.groupby("episode_id")["episode_timestep"].transform("size")
    terminals = df["terminals"].astype(bool)
    keep_in_sorted = terminals | (pos < (ep_len - action_horizon))
    keep_idx = df.index[keep_in_sorted]
    filtered_df = df.loc[df.index.isin(keep_idx)].copy()

    filtered_index = np.array(filtered_df.index)

    for key in cache.keys():
        if key == "metadata":
            continue
        
        if type(cache[key]) == dict:
            for subkey in cache[key].keys():
                cache[key][subkey] = cache[key][subkey][filtered_index]
        else:
            cache[key] = cache[key][filtered_index]
    
    return cache


def load_vlm_cache(cache_path: str) -> Dict:
    """Load the VLM action cache from disk.
    
    Args:
        cache_path: Path to the .pkl cache file
        
    Returns:
        Dictionary containing:
            - episode_ids: (num_entries,) - Episode IDs
            - episode_timesteps: (num_entries,) - Timesteps
            - current_vlm_outputs: (num_entries, N, vlm_dim) - N VLM outputs for current state
            - next_vlm_outputs: (num_entries, N, vlm_dim) - N VLM outputs for next state
            - current_actions: (num_entries, N, action_horizon, action_dim) - N actions for current state
            - next_actions: (num_entries, N, action_horizon, action_dim) - N actions for next state
            - current_states: (num_entries, 8) - Current states
            - next_states: (num_entries, 8) - Next states
            - rewards: (num_entries,) - Rewards
            - masks: (num_entries,) - Masks
            - actions: (num_entries, action_horizon, action_dim) - Ground truth actions
            - metadata: Dictionary with cache metadata
    """
    logging.info(f"Loading VLM cache from {cache_path}")
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    
    return cache

if __name__ == "__main__":
    cache = load_vlm_cache("pkl_files/CLEAN_traj30_v1.pkl")
    cache = filter_cache(cache)

    pickle.dump(cache, open("pkl_files/CLEAN_traj30_v1_filtered.pkl", "wb"))
    # breakpoint()