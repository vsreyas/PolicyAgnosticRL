"""
Libero -> TFRecord conversion (Calvin-style, per-episode, with rewards).
"""

from __future__ import annotations
import os, json, math, h5py, cv2, click, numpy as np, tensorflow as tf
from PIL import Image
from typing import Tuple, Optional
from functools import partial
from multiprocessing import Pool, cpu_count
from scipy.spatial.transform import Rotation as R

# ============================ TF Helpers ============================

def _tensor_feature(array: np.ndarray) -> tf.train.Feature:
    t = tf.convert_to_tensor(array)
    s = tf.io.serialize_tensor(t).numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[s]))

def _bytes_feature(value: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# ============================ Libero Loaders ============================

def _load_primary_rgb(dataset_path: str, episode_id: str, step_id: str, primary_mode: str) -> np.ndarray:
    path = f"{dataset_path}/episodes/{episode_id}/steps/{step_id}/{primary_mode}.jpg"
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)

def _load_wrist_rgb(dataset_path: str, episode_id: str, step_id: str) -> np.ndarray:
    path = f"{dataset_path}/episodes/{episode_id}/steps/{step_id}/image_wrist.jpg"
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)

def _load_language(other_file, filetype: str, key="language_instruction") -> str:
    if filetype == "h5":
        return other_file[key][()].decode("utf-8")
    elif filetype == "npz":
        return other_file[key].tobytes().decode("utf-8")
    else:
        raise NotImplementedError

def _load_action(other_file, filetype: str) -> np.ndarray:
    return other_file["action"][()] if filetype == "h5" else other_file["action"]

def _load_robot_obs(other_file, filetype: str, gripper_width: bool) -> np.ndarray:
    robot_obs = np.zeros(15 + (2 if gripper_width else 0), dtype=np.float32)
    if filetype == "h5":
        robot_obs[:6] = other_file["observation"]["tcp_pose"][:6]
        euler = R.from_euler("xyz", robot_obs[3:6], degrees=False).as_euler("xyz", degrees=False)
        robot_obs[3:6] = euler
        robot_obs[-1] = other_file["observation"]["gripper_state"][()]
        robot_obs[7:14] = other_file["observation"]["proprio"][()]
        if gripper_width:
            robot_obs[-2:] = other_file["observation"]["gripper_position"][()]
    else:
        robot_obs[:6] = other_file["observation_tcp_pose"][:6]
        euler = R.from_euler("xyz", robot_obs[3:6], degrees=False).as_euler("xyz", degrees=False)
        robot_obs[3:6] = euler
        robot_obs[-1] = other_file["observation_gripper_state"]
        robot_obs[7:14] = other_file["observation_proprio"]
        if gripper_width:
            robot_obs[-2:] = other_file["observation_gripper_position"]
    return robot_obs.astype(np.float32)

def _load_scene_obs() -> np.ndarray:
    return np.zeros(24, dtype=np.float32)

def _open_other(path: str, filetype: str):
    return h5py.File(path, "r") if filetype == "h5" else np.load(path, allow_pickle=True)

def _resize(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size)) if img.shape[0] != size else img

# ============================ Reward Computation ============================

def calculate_synthetic_rewards_for_libero_episode(
    num_steps: int,
    discount_factor: float = 0.99,
    mode: str = "exponential",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Synthetic reward function for Libero dataset without subtasks.
    Gives a single terminal reward of 1, optionally decayed over time.

    Args:
        num_steps (int): number of timesteps in the episode
        discount_factor (float): decay rate for exponential mode
        mode (str): "exponential" or "linear"

    Returns:
        rewards (np.ndarray): shape [num_steps-1]
        masks (np.ndarray): shape [num_steps-1]
        mc_returns (np.ndarray): shape [num_steps-1]
    """
    # --- create decaying rewards ---
    if mode == "exponential":
        # Exponentially increasing toward final step (simulate discounted terminal reward)
        rewards_full = np.array([discount_factor ** (num_steps - 1 - t) for t in range(num_steps)], dtype=np.float32)
    elif mode == "linear":
        # Linearly increasing reward from 0 to 1
        rewards_full = np.linspace(0.0, 1.0, num_steps, dtype=np.float32)
    else:
        raise ValueError("mode must be 'exponential' or 'linear'")

    # Normalize so final reward = 1
    rewards_full /= rewards_full[-1]

    # Use first T-1 rewards as transition rewards
    rewards = rewards_full[:-1]

    # --- masks (no early termination) ---
    masks = np.ones_like(rewards, dtype=np.float32)

    # --- Monte Carlo return-to-go ---
    mc_returns = np.zeros_like(rewards)
    acc = 0.0
    for t in reversed(range(len(rewards))):
        acc = rewards[t] + discount_factor * acc * masks[t]
        mc_returns[t] = acc

    return rewards, masks, mc_returns

# ============================ Core Conversion ============================

def convert_libero_episode_to_tfrecord(
    dataset_path: str,
    output_path: str,
    episode_id: str,
    num_steps: int,
    image_size: int = 224,
    primary_mode: str = "image_primary",
    include_next_observations: bool = False,
    filetype: str = "h5",
    gripper_width: bool = False,
    include_rewards: bool = False,
    reward_bias: float = 0.0,
) -> None:
    tf.config.set_visible_devices([], "GPU")
    T = num_steps
    assert T > 1, f"Episode {episode_id} must have >=2 steps"
    print(f"[Proc] episode {episode_id} ({T} steps)")

    images0 = np.zeros((T, image_size, image_size, 3), np.uint8)
    images1 = np.zeros((T, image_size, image_size, 3), np.uint8)
    states = None
    actions = None
    language = None

    for step in range(T):
        step_id = str(step).zfill(4)
        other_path = f"{dataset_path}/episodes/{episode_id}/steps/{step_id}/other.{filetype}"
        other = _open_other(other_path, filetype)

        if language is None:
            language = _load_language(other, filetype)

        img0 = _load_primary_rgb(dataset_path, episode_id, step_id, primary_mode)
        img1 = _load_wrist_rgb(dataset_path, episode_id, step_id)
        images0[step] = _resize(img0, image_size)
        images1[step] = _resize(img1, image_size)

        robot = _load_robot_obs(other, filetype, gripper_width)
        # scene = _load_scene_obs()
        # s = np.concatenate([robot, scene], 0)
        s = robot
        if states is None:
            states = np.zeros((T, s.shape[0]), np.float32)
        states[step] = s

        act = _load_action(other, filetype).astype(np.float32)
        if actions is None:
            actions = np.zeros((T, act.shape[0]), np.float32)
        actions[step] = act
        if filetype == "h5": other.close()

    actions = actions[:-1]
    rewards = masks = mc_returns = None
    if include_rewards:
        rewards, masks, mc_returns = calculate_synthetic_rewards_for_libero_episode(
            num_steps=num_steps,
            discount_factor=0.99,   # or tune as you wish
            mode="exponential",      # "linear" also available
        )

    features = {
        "observations/state": _tensor_feature(states if not include_next_observations else states[:-1]),
        "observations/images0": _tensor_feature(images0 if not include_next_observations else images0[:-1]),
        "observations/images1": _tensor_feature(images1 if not include_next_observations else images1[:-1]),
        "actions": _tensor_feature(actions),
        "language": _bytes_feature(language.encode("utf-8")),
    }
    if include_next_observations:
        features.update({
            "next_observations/state": _tensor_feature(states[1:]),
            "next_observations/images0": _tensor_feature(images0[1:]),
            "next_observations/images1": _tensor_feature(images1[1:]),
        })
    if include_rewards:
        features.update({
            "rewards": _tensor_feature(rewards),
            "masks": _tensor_feature(masks),
            "mc_returns": _tensor_feature(mc_returns),
        })

    example = tf.train.Example(features=tf.train.Features(feature=features))
    os.makedirs(output_path, exist_ok=True)
    out_file = os.path.join(output_path, f"episode_{episode_id}.tfrecord")
    if tf.io.gfile.exists(out_file): tf.io.gfile.remove(out_file)
    with tf.io.TFRecordWriter(out_file) as w:
        w.write(example.SerializeToString())
    print(f"[OK] {episode_id} -> {out_file}")

# ============================ Dataset Loop ============================

def _load_episode_index_list(path: str):
    with open(path, "r") as f:
        ep_list = json.load(f)
    return [(str(eid), int(nsteps)) for eid, nsteps in ep_list]

def _convert_one(args): convert_libero_episode_to_tfrecord(*args)

@click.command()
@click.option("--dataset_path", type=str, required=True, default="/data/hf_cache/datasets/LIBERO/libero_10_converted/")
@click.option("--output_path", type=str, required=True, default="/data/hf_cache/datasets/LIBERO/libero_10_tf/")
@click.option("--data_info_name", type=str, default="libero_10_converted")
@click.option("--image_size", type=int, default=224)
@click.option("--include_next_observations", is_flag=True, default=False)
@click.option("--filetype", type=click.Choice(["h5", "npz"]), default="h5")
@click.option("--gripper_width", is_flag=True, default=False)
@click.option("--include_rewards", is_flag=True, default=True)
@click.option("--reward_bias", type=float, default=0.0)
@click.option("--num_workers", type=int, default=max(1, cpu_count() // 2))
def convert_libero_dataset_to_tfrecord(
    dataset_path: str,
    output_path: str,
    data_info_name: str,
    image_size: int,
    include_next_observations: bool,
    filetype: str,
    gripper_width: bool,
    include_rewards: bool,
    reward_bias: float,
    num_workers: int,
):
    tf.config.set_visible_devices([], "GPU")
    data_info_path = os.path.join("./data_info", f"{data_info_name}.json")
    ep_list = _load_episode_index_list(data_info_path)
    print(f"[Info] {len(ep_list)} episodes in {data_info_path}")

    args = [
        (dataset_path, output_path, eid, n, image_size, "image_primary",
         include_next_observations, filetype, gripper_width,
         include_rewards, reward_bias)
        for eid, n in ep_list
    ]

    if num_workers <= 1:
        for a in args: _convert_one(a)
    else:
        with Pool(processes=num_workers, maxtasksperchild=1) as p:
            for _ in p.imap_unordered(_convert_one, args): pass
    print(f"[Done] Wrote TFRecords to {output_path}")

if __name__ == "__main__":
    convert_libero_dataset_to_tfrecord()
