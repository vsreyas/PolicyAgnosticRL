import os
import cv2
import gzip
import pickle
import gym
import torch
import numpy as np
from typing import Optional, Dict
from scipy.spatial.transform import Rotation as R
import random
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import benchmark

from jaxrl_m.data.dataset import Dataset
from jaxrl_m.data.image_replay_buffer import ImageReplayBuffer
from jaxrl_m.data.image_replay_buffer_pi import ImageReplayBufferPi
from jaxrl_m.data.bridge_dataset import glob_to_path_list
import numpy as np
from typing import Dict, Any
import torch
import clip

import json
from tqdm import tqdm


def load_language_embeddings(path: str) -> dict[int, np.ndarray]:
    """
    Loads language embeddings from a JSON file and converts them to float32 numpy arrays.

    Returns:
        Dict[int, np.ndarray] mapping task_id → embedding (shape [512], dtype float32)
    """
    with open(path, "r") as f:
        data = json.load(f)

    # Convert keys to int and lists to np.float32 arrays
    embeddings = {
        int(k): np.array(v, dtype=np.float32)
        for k, v in data.items()
    }

    # Optionally verify dimensions
    for k, v in embeddings.items():
        assert v.shape == (512,), f"Task {k} has wrong shape {v.shape}"

    return embeddings

def get_dataset(
    dataset: Dict[str, np.ndarray],
    clip_to_eps: bool = True,
    eps: float = 1e-5,
    filter_terminals: bool = False,
    obs_dtype=np.float32,
):
    """Convert a LIBERO dataset dictionary to Dataset structure (CALVIN-style),
    including language field if present.
    """
    
    # Clip actions to avoid saturation near ±1
    if clip_to_eps:
        lim = 1 - eps
        dataset["actions"] = np.clip(dataset["actions"], -lim, lim)

    # Ensure final transition is marked terminal
    dataset["terminals"][-1] = 1

    # Optionally filter out terminal transitions (except penultimate)
    if filter_terminals:
        non_last_idx = np.nonzero(~dataset["terminals"])[0]
        last_idx = np.nonzero(dataset["terminals"])[0]
        penult_idx = last_idx - 1

        new_dataset = {}
        for k, v in dataset.items():
            v = v.copy()
            if k == "terminals":
                v[penult_idx] = 1
            new_dataset[k] = v[non_last_idx]
        dataset = new_dataset

    dones_float = dataset["terminals"].astype(np.float32)
    observations = dataset["observations"].astype(obs_dtype)
    next_observations = dataset["next_observations"].astype(obs_dtype)

    # --- handle language if present ---
    language = None
    if "language" in dataset:
        lang_field = dataset["language"]
        if isinstance(lang_field, np.ndarray):
            # if language is stored per transition (array of strings or bytes)
            language = np.array([
                s if isinstance(s, bytes) else s.encode("utf-8")
                for s in lang_field
            ])
        else:
            # single string/bytes for all transitions
            language = (
                lang_field
                if isinstance(lang_field, bytes)
                else lang_field.encode("utf-8")
            )

    # Build dataset dictionary
    dataset_dict: Dict[str, Any] = {
        "observations": observations,
        "actions": dataset["actions"].astype(np.float32),
        "rewards": dataset["rewards"].astype(np.float32),
        "masks": 1.0 - dones_float,
        "dones_float": dones_float,
        "next_observations": next_observations,
    }

    # Include language if available
    if language is not None:
        dataset_dict["language"] = language

    return Dataset(dataset_dict)



# ======================================================
# === Config loader
# ======================================================

def get_libero_config(env_name="libero_10"):
    """Loads Hydra config for Libero (same structure as Calvin configs)."""
    import hydra
    hydra.initialize(config_path="libero_config", version_base=None)
    return hydra.compose(config_name=env_name)


# ======================================================
# === Environment factory
# ======================================================
def get_libero_env(
    cfg=None,
    task_id = 0,
    goal_conditioned: bool = False,
    libero_path: str = "",
    device_id: int = 0,
    **kwargs,
):
    """
    Creates a LIBERO OffScreenRenderEnv wrapped in Calvin-style Gym wrappers.

    The returned environment has the same structure and key format as CALVIN:
    - GymWrapper (for unified observation/action space)
    - wrap_env (normalization, seeding, etc.)
    - Optional goal-conditioning wrapper
    """

    # --- Config setup ---
    if cfg is None:
        cfg = get_libero_config()

    # --- Load benchmark suite and task ---
    benchmark_dict = benchmark.get_benchmark_dict()
    suite = benchmark_dict[cfg.name]()

    num_envs = 10 if "10" in cfg.name else 90
    task_id = task_id % num_envs
    task = suite.get_task(task_id)

    # --- Paths for BDDL and init states ---
    bddl_file = os.path.join(
        f"{cfg.libero_path}/libero/libero/bddl_files",
        task.problem_folder,
        task.bddl_file,
    )
    init_states_path = os.path.join(
        f"{cfg.libero_path}/libero/libero/init_files",
        task.problem_folder,
        task.init_states_file,
    )

    # --- Base LIBERO environment ---
    env_args = dict(
        bddl_file_name=bddl_file,
        camera_heights=cfg.screen_size[0],
        camera_widths=cfg.screen_size[1],
        # render_gpu_device_id=device_id,
        ignore_done=False,
        reward_shaping=True

    )
    base_env = OffScreenRenderEnv(**env_args)
    
    # --- Wrap LIBERO base env in Calvin-style stack ---
    env = LiberoEnvWrapper(base_env, init_states_path=init_states_path, camera_dims=cfg.screen_size, max_steps=cfg.max_episode_steps, cfg=cfg, task_id = task_id,
                           suite=suite)
    

    # # Convert to dict-style obs (like Calvin)
    # env = DictWrapper(env)

    # Wrap into Gym-like interface (Calvin compatibility)
    # env = GymWrapper(
    #     env=env,
    #     from_pixels=cfg.pixel_ob,
    #     from_state=cfg.state_ob,
    #     height=cfg.screen_size[0],
    #     width=cfg.screen_size[1],
    #     channels_first=False,
    #     frame_skip=cfg.action_repeat,
    #     return_state=False,
    # )

    # Apply common wrappers (normalization, flattening, etc.)
    # env = wrap_env(env, cfg)

    # Add goal conditioning or proprio wrapper (same as Calvin)
    # if goal_conditioned:
    #     env = GCLiberoWrapper(env, goal_image_size=cfg.screen_size[0], **kwargs)
    # else:
    #     env = AddProprioWrapper(env)

    return env
# ======================================================
# === TFRecord + Pickle dataset loading
# ======================================================

def get_libero_tfrecord_dataset(tfrecord_regexp: str, 
                                goal_relabeling_strategy: Optional[str] = None,
                                goal_relabeling_kwargs: dict = {},
                                cache: bool = False,
                                train: bool = True,
                                seed: int = 0,
                                is_pi: bool = False,
                                **kwargs):
    assert tfrecord_regexp.endswith(".tfrecord")
    paths = glob_to_path_list(tfrecord_regexp)
    if is_pi:
        return ImageReplayBufferPi(data_paths=paths,
        seed=seed,
        goal_relabeling_strategy=goal_relabeling_strategy,
        goal_relabeling_kwargs=goal_relabeling_kwargs,
        cache=cache,
        train=train,
        **kwargs,
    )
    return ImageReplayBuffer(
        data_paths=paths,
        seed=seed,
        goal_relabeling_strategy=goal_relabeling_strategy,
        goal_relabeling_kwargs=goal_relabeling_kwargs,
        cache=cache,
        train=train,
        **kwargs,
    )


def get_libero_env_and_dataset(dataset_path: str):
    assert dataset_path.endswith(".gz")
    env = get_libero_env(goal_conditioned=False)
    data = pickle.load(gzip.open(dataset_path, "rb"))

    ds = []
    for traj in data:
        obs = traj["observations"]
        next_obs = traj["next_observations"]
        actions = traj["actions"]
        rewards = np.zeros(len(actions), np.float32)
        rewards[-1] = 1
        terminals = np.zeros(len(actions), bool)
        terminals[-1] = True
        ds.append(
            dict(
                observations=obs[:-1],
                next_observations=next_obs[:-1],
                actions=actions[:-1],
                rewards=rewards[:-1],
                terminals=terminals[:-1],
            )
        )

    dataset = {k: np.concatenate([d[k] for d in ds], 0) for k in ds[0]}
    dataset = get_dataset(dataset)
    return env, dataset


def _get_task_index(suite, name: str):
    for i, task in enumerate(suite.tasks):
        if name.lower() in task.name.lower():
            return i
    raise ValueError(f"Task {name} not found in Libero suite.")



class LiberoEnvWrapper(gym.Wrapper):
    """
    Wraps OffScreenRenderEnv to return Calvin-compatible keys:

        {
            "observations/state": np.ndarray,
            "observations/images0": np.uint8[H,W,3],
            "observations/images1": np.uint8[H,W,3],
        }

    This ensures the downstream pipeline (TFRecord conversion, training)
    works exactly as it does for Calvin.
    """

    def __init__(self, env: OffScreenRenderEnv, init_states_path: Optional[str] = None, gripper_width=False, camera_dims = (224,224), max_steps = 999, cfg=None, 
                 task_id = None, suite=None):
        super().__init__(env)
        self.metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
        self.env = env
        self.gripper_width = gripper_width
        self.init_states = (
            torch.load(init_states_path, map_location="cpu", weights_only=False)
            if init_states_path and os.path.exists(init_states_path)
            else None
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32,
        )
        image_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(camera_dims[0], camera_dims[1], 3),
            dtype=np.uint8,
        )
        proprio_dim = 15 + (2 if gripper_width else 0)
        proprio_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(proprio_dim,),
            dtype=np.float32,
        )
        lang_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(512,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "image": image_space,
                "wrist_image": image_space,
                "proprio": proprio_space,
                "language": lang_space,
            }
        )

        self.__step = 0
        self.max_steps = max_steps

        self.cfg = cfg
        self.task_id = task_id
        self.num_envs = 10 if "10" in cfg.name else 90
        self.suite = suite
        self.id2embedding = load_language_embeddings("data_info/libero_id2embeddings_normalised.json")
        self.language_embedding = self.id2embedding[self.task_id]
        self.reset()



    def reset(self):
        self.env.close()
        del self.env, self.language_embedding
        
        self.task_id = (self.task_id + 1) % self.num_envs
        task = self.suite.get_task(self.task_id)
        self.language_embedding = self.id2embedding[self.task_id]
        # --- Paths for BDDL and init states ---
        bddl_file = os.path.join(
            f"{self.cfg.libero_path}/libero/libero/bddl_files",
            task.problem_folder,
            task.bddl_file,
        )
        init_states_path = os.path.join(
            f"{self.cfg.libero_path}/libero/libero/init_files",
            task.problem_folder,
            task.init_states_file,
        )

        # --- Base LIBERO environment ---
        env_args = dict(
            bddl_file_name=bddl_file,
            camera_heights=self.cfg.screen_size[0],
            camera_widths=self.cfg.screen_size[1],
            # render_gpu_device_id=device_id,
            ignore_done=False,
            reward_shaping=True

        )
        self.env = OffScreenRenderEnv(**env_args)

        del env_args 
        import gc; gc.collect()

        obs = self.env.reset()
        self.__step = 0
        return self._format_obs(obs)

    def reset_to_state(self, idx: int):
        """Reset to a specific saved state (to be implemented later)."""
        raise NotImplementedError("reset_to_state not implemented for Libero yet.")

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.__step += 1
        if self.env.check_success():
            done = True
            reward = 1.0
        if not done and self.__step > self.max_steps:
            done = True
        
        return self._format_obs(obs, action), reward, done, info

    def _format_obs(self, raw_obs: Dict[str, np.ndarray], action=None) -> Dict[str, np.ndarray]:
        """Converts raw Libero obs to Calvin key format."""
        img0 = raw_obs["agentview_image"]
        img1 = raw_obs["robot0_eye_in_hand_image"]

        # Compose robot state (15 + optional 2 + 24 zeros)
        pos = raw_obs["robot0_eef_pos"]
        quat = raw_obs["robot0_eef_quat"]
        euler = R.from_quat(quat).as_euler("xyz", degrees=False)
        gripper = action[-1] if action is not None else -1.0
        if isinstance(gripper, np.ndarray):
            gripper = gripper[0]
        proprio = raw_obs["robot0_joint_pos"]
        robot_vec = np.zeros(15 + (2 if self.gripper_width else 0), np.float32)
        robot_vec[:3] = pos
        robot_vec[3:6] = euler
        if not self.gripper_width:
            robot_vec[-1] = gripper
        else:
            robot_vec[-2: ] = raw_obs['robot0_gripper_qpos']
        robot_vec[7:14] = proprio
        # scene_obs = np.zeros(24, np.float32)
        # state = np.concatenate([robot_vec, scene_obs], axis=0)
        state = robot_vec
        obs = {
            "proprio": state,
            "image": img0,
            "wrist_image": img1,
            "language": self.language_embedding,
        }
        return obs

# # ======================================================
# # === Dict and Goal Wrappers
# # ======================================================

# class DictWrapper(gym.Wrapper):
#     """Simple dict passthrough for Calvin compatibility."""

#     def reset(self):
#         return self.env.reset()

#     def step(self, action):
#         obs, reward, done, info = self.env.step(action)
#         return obs, reward, done, info


# class GCLiberoWrapper(gym.Wrapper):
#     """Adds goal conditioning keys."""

#     def __init__(self, env: gym.Env, goal_image_size: int = 224):
#         super().__init__(env)
#         self.goal_image_size = goal_image_size
#         self.current_goal = None

#     def reset(self, **kwargs):
#         obs = self.env.reset(**kwargs)
#         goal_img = self._render_goal()
#         obs["goal_image"] = goal_img
#         obs["goal_state"] = obs["observations/state"]
#         self.current_goal = goal_img
#         return obs

#     def step(self, action):
#         obs, reward, done, info = self.env.step(action)
#         obs["goal_image"] = self.current_goal
#         obs["goal_state"] = obs["observations/state"]
#         return obs, reward, done, info

#     def _render_goal(self):
#         raise NotImplementedError("Goal rendering not implemented yet for Libero.")

def save_all_task_language_embeddings(cfg=None, output_path="libero_language_embeddings.json", key_val="id"):
    """
    Iterates over all tasks in the LIBERO benchmark suite and saves CLIP language embeddings.

    Args:
        cfg: Hydra config for the Libero benchmark (optional; defaults to get_libero_config()).
        output_path (str): Path to save the JSON file.
    """
    if cfg is None:
        cfg = get_libero_config()

    benchmark_dict = benchmark.get_benchmark_dict()
    suite = benchmark_dict[cfg.name]()

    # Load CLIP once
    clip_device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("ViT-B/32", device=clip_device)
    clip_model.eval()

    task_embeddings = {}

    print(f"Encoding language for {len(suite.tasks)} tasks...")
    for task_id, task in tqdm(enumerate(suite.tasks), total=len(suite.tasks)):
        if hasattr(task, "language") and isinstance(task.language, str):
            text = task.language
            with torch.no_grad():
                tokens = clip.tokenize([text]).to(clip_device)
                text_features = clip_model.encode_text(tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                text_features = text_features[0].cpu().numpy().astype(float).tolist()
            key = text if key_val == "lang" else task_id
            task_embeddings[key] = text_features
        else:
            print(f"⚠️  Skipping task {task_id}: no valid language description.")
            continue

    # Save as JSON
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(task_embeddings, f, indent=2)

    print(f"✅ Saved {len(task_embeddings)} language embeddings to {output_path}")
    return task_embeddings

def main():
    # import time

    # print("Initializing LIBERO environment...")
    # env = get_libero_env(goal_conditioned=False)

    # print("Resetting environment...")
    # obs = env.reset()
    # print("Initial observation keys:", obs.keys())
    # for k, v in obs.items():
    #     print(f"  {k}: shape={np.array(v).shape}, dtype={np.array(v).dtype}")

    # num_steps = 10
    # print(f"\nRunning {num_steps} random steps...")
    # done = False
    # step = 0
    # while not done:
    #     action = env.action_space.sample()
    #     next_obs, reward, done, info = env.step(action)
    #     print(f"Step {step+1}: reward={reward:.3f}, done={done}")

    #     # Show image sizes for quick sanity check
    #     if step == 0:
    #         print("Image0 shape:", np.array(next_obs['image']).shape)
    #         print("Image1 shape:", np.array(next_obs['images1']).shape)
    #         print("Proprio shape:", np.array(next_obs['proprio']).shape)

    #     if done:
    #         break
    #     step += 1
    #     time.sleep(0.1)
    save_all_task_language_embeddings()
    


if __name__ == "__main__":
    main()
