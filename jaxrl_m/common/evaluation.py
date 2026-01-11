import math
from collections import defaultdict
from typing import Dict
from absl import logging

import gym
import jax
import numpy as np
import imageio.v3 as iio
import os


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def filter_info(info):
    filter_keys = [
        "object_names",
        "target_object",
        "initial_positions",
        "target_position",
        "goal",
    ]
    for k in filter_keys:
        if k in info:
            del info[k]
    return info


def add_to(dict_of_lists, single_dict):
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(policy_fn, env: gym.Env, num_episodes: int) -> Dict[str, float]:
    stats = defaultdict(list)
    for _ in range(num_episodes):
        observation, info = env.reset()
        add_to(stats, flatten(info))
        done = False
        while not done:
            action = policy_fn(observation)
            observation, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            add_to(stats, flatten(info))
        add_to(stats, flatten(info, parent_key="final"))

    for k, v in stats.items():
        stats[k] = np.mean(v)
    return stats


def evaluate_with_trajectories(
    policy_fn, env: gym.Env, num_episodes: int, clip_action: float = np.inf
) -> Dict[str, float]:
    trajectories = []
    stats = defaultdict(list)

    for _ in range(num_episodes):
        trajectory = defaultdict(list)
        observation, info = env.reset()
        add_to(stats, flatten(info))
        done = False
        while not done:
            action = policy_fn(observation)
            action = np.clip(action, -clip_action, clip_action)
            next_observation, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            transition = dict(
                observations=observation,
                next_observations=next_observation,
                actions=action,
                rewards=r,
                dones=done,
                infos=info,
                masks=1 - terminated,
            )
            add_to(trajectory, transition)
            add_to(stats, flatten(info))
            observation = next_observation
        add_to(stats, flatten(info, parent_key="final"))
        trajectories.append(trajectory)

    for k, v in stats.items():
        stats[k] = np.mean(v)
    return stats, trajectories


# def evaluate_with_trajectories_vectorized(
#     policy_fn,
#     env: gym.vector.VectorEnv,
#     num_episodes: int,
#     save_video: bool = False,
#     max_episodes_for_video: int = 2,
# ):
#     trajectories = [[defaultdict(list)] for _ in range(env.num_envs)]
#     num_envs = env.num_envs
#     assert num_episodes % num_envs == 0
#     episode_counts = np.zeros(num_envs, dtype=int)

#     observations = env.reset()
#     if isinstance(observations, tuple) and len(observations) == 2:
#         observations, _ = observations
#     step_indices = np.zeros(
#         num_envs, dtype=int
#     )  # Index of the current step in each env

#     while np.sum(episode_counts) < num_episodes:
#         actions = policy_fn(observations, step_indices=step_indices)
#         step_variables = env.step(actions)
#         if len(step_variables) == 4:
#             next_observations, rewards, dones, infos = step_variables
#         else:
#             next_observations, rewards, dones, truncated, infos = step_variables
#             dones = np.logical_or(dones, truncated)

#         if save_video and np.sum(episode_counts) < max_episodes_for_video:
#             images = env.render()

#         step_indices += 1
#         log_progress = False
#         for i in range(num_envs):
#             if episode_counts[i] < num_episodes // num_envs:
#                 obs = jax.tree_map(lambda x: x[i], observations)
#                 next_obs = jax.tree_map(lambda x: x[i], next_observations)
#                 transition = dict(
#                     observation=obs,
#                     next_observation=next_obs,
#                     action=actions[i],
#                     reward=rewards[i],
#                     done=dones[i],
#                     # info=infos[i],
#                     info={},
#                 )
#                 if save_video and np.sum(episode_counts) < max_episodes_for_video:
#                     transition["image"] = images[i].copy()
#                 add_to(trajectories[i][-1], transition)

#                 if dones[i]:
#                     episode_counts[i] += 1
#                     step_indices[i] = 0
#                     log_progress = True
#                     if episode_counts[i] < num_episodes // num_envs:
#                         trajectories[i].append(defaultdict(list))

#         if log_progress:
#             logging.info(
#                 f"Completed {np.sum(episode_counts)} out of {num_episodes} eval episodes..."
#             )

#         observations = next_observations

#     all_trajectories = []
#     for i in range(num_envs):
#         all_trajectories.extend(trajectories[i])

#     return all_trajectories

def evaluate_with_trajectories_vectorized(
    policy_fn,
    env: gym.vector.VectorEnv,
    num_episodes: int,
    save_video: bool = False,
    max_episodes_for_video: int = 2,
    action_horizon: int = 10,        # <--- ADD
):
    H = action_horizon
    half_H = max(1, H // 2)
    
    trajectories = [[defaultdict(list)] for _ in range(env.num_envs)]
    num_envs = env.num_envs
    assert num_episodes % num_envs == 0
    episode_counts = np.zeros(num_envs, dtype=int)

    observations = env.reset()
    if isinstance(observations, tuple) and len(observations) == 2:
        observations, _ = observations

    # tracks where each env is inside its horizon
    action_indices = np.zeros(num_envs, dtype=int)

    # cached actions: (num_envs, H, act_dim)
    cached_action_sequences = None

    while np.sum(episode_counts) < num_episodes:

        # -------------------------------------------------------------
        # 1. CALL POLICY WHENEVER ANY ENV NEEDS A REFILL
        # -------------------------------------------------------------
        # We need fresh actions IF:
        #    - first time OR
        #    - ANY env has already consumed half_H steps
        if cached_action_sequences is None or np.any(action_indices >= half_H):
            cached_action_sequences = policy_fn(observations)

            # If policy returned (num_envs, 1, D) or (num_envs, D)
            if cached_action_sequences.ndim == 2:
                # (num_envs, D) → make it (num_envs, 1, D)
                cached_action_sequences = cached_action_sequences[:, None, :]

            if cached_action_sequences.ndim == 3 and cached_action_sequences.shape[1] < H:
                raise ValueError(
                    f"Policy returned horizon={cached_action_sequences.shape[1]}, expected >= {H}"
                )

            action_indices[:] = 0
        # -------------------------------------------------------------
        # 2. SELECT ONE ACTION PER ENV FOR THIS STEP
        # -------------------------------------------------------------
        # actions_to_apply: (num_envs, act_dim)
        actions_to_apply = np.array([
            cached_action_sequences[i, action_indices[i]]
            for i in range(num_envs)
        ])

        # Update horizon counters
        action_indices += 1

        # -------------------------------------------------------------
        # 3. STEP ENVIRONMENTS
        # -------------------------------------------------------------
        step_variables = env.step(actions_to_apply)

        if len(step_variables) == 4:
            next_observations, rewards, dones, infos = step_variables
        else:
            next_observations, rewards, dones, truncated, infos = step_variables
            dones = np.logical_or(dones, truncated)

        if save_video and np.sum(episode_counts) < max_episodes_for_video:
            images = env.render()

        log_progress = False

        # -------------------------------------------------------------
        # 4. STORE TRAJECTORIES PER-ENV
        # -------------------------------------------------------------
        for i in range(num_envs):
            if episode_counts[i] < num_episodes // num_envs:
                obs_i = jax.tree_map(lambda x: x[i], observations)
                next_obs_i = jax.tree_map(lambda x: x[i], next_observations)

                transition = dict(
                    observation=obs_i,
                    next_observation=next_obs_i,
                    action=actions_to_apply[i],
                    reward=rewards[i],
                    done=dones[i],
                    info={},
                )

                if save_video and np.sum(episode_counts) < max_episodes_for_video:
                    transition["image"] = images[i].copy()

                add_to(trajectories[i][-1], transition)

                if dones[i]:
                    episode_counts[i] += 1
                    action_indices[:] = 0                   # reset horizon index
                    cached_action_sequences= None       # force refresh for this env
                    log_progress = True

                    if episode_counts[i] < num_episodes // num_envs:
                        trajectories[i].append(defaultdict(list))

        if log_progress:
            logging.info(
                f"Completed {np.sum(episode_counts)} out of {num_episodes} eval episodes..."
            )

        observations = next_observations

    # -------------------------------------------------------------
    # 5. FLATTEN TRAJECTORIES
    # -------------------------------------------------------------
    all_trajectories = []
    for i in range(num_envs):
        all_trajectories.extend(trajectories[i])

    return all_trajectories


def evaluate_gc(
    policy_fn,
    env: gym.Env,
    num_episodes: int,
    return_trajectories: bool = False,
) -> Dict[str, float]:
    stats = defaultdict(list)

    if return_trajectories:
        trajectories = []

    for _ in range(num_episodes):
        if return_trajectories:
            trajectory = defaultdict(list)

        observation, info = env.reset()
        goal = info["goal"]
        add_to(stats, flatten(filter_info(info)))
        done = False

        while not done:
            action = policy_fn(observation, goal)
            next_observation, r, terminated, truncated, info = env.step(action)
            goal = info["goal"]
            done = terminated or truncated
            transition = dict(
                observation=observation,
                next_observation=next_observation,
                goal=goal,
                action=action,
                reward=r,
                done=done,
                info=info,
            )

            add_to(stats, flatten(filter_info(info)))

            if return_trajectories:
                add_to(trajectory, transition)

            observation = next_observation

        add_to(stats, flatten(filter_info(info), parent_key="final"))
        if return_trajectories:
            trajectory["steps_remaining"] = list(
                np.arange(len(trajectory["action"]))[::-1]
            )
            trajectories.append(trajectory)

    stats = {k: np.mean(v) for k, v in stats.items() if not isinstance(v[0], str)}

    if return_trajectories:
        return stats, trajectories
    else:
        return stats


def bootstrap_std(arr, f=np.mean, n=30):
    arr = np.array(arr)
    return np.std([f(arr[np.random.choice(len(arr), len(arr))]) for _ in range(n)])


def parallel_evaluate(policy_fn, eval_envs, num_eval, verbose=True):
    n_envs = len(eval_envs.reset())
    eval_episode_rewards = []
    eval_episode_time_rewards = []
    counter = np.zeros(n_envs)

    obs = eval_envs.reset()
    if verbose:
        print(f"Evaluating Envs")
    n_per = int(math.ceil(num_eval / n_envs))
    n_to_eval = n_per * n_envs
    while len(eval_episode_rewards) < n_to_eval:
        action = policy_fn(obs)

        # Observe reward and next obs
        obs, _, done, infos = eval_envs.step(action)

        for n, info in enumerate(infos):
            if "episode" in info.keys() and counter[n] < n_per:
                eval_episode_rewards.append(info["episode"]["r"])
                eval_episode_time_rewards.append(info["episode"]["time_r"])
                counter[n] += 1
    if verbose:
        print(
            f"Evaluation using {len(eval_episode_rewards)} episodes: mean reward {np.mean(eval_episode_rewards):.5f} +- {bootstrap_std(eval_episode_rewards):.5f} \n"
        )
    return eval_episode_rewards, eval_episode_time_rewards


# def evaluate_with_trajectories_libero(
#     policy_fn,
#     env,
#     num_episodes: int,
#     save_video: bool = False,
#     max_episodes_for_video: int = 2,
#     action_horizon=1,
# ):
#     trajectories = [defaultdict(list)]
#     episode_count = 0

#     observations = env.reset()
#     if isinstance(observations, tuple) and len(observations) == 2:
#         observations, _ = observations
#     step_index = 0
#     half_horizon = max(1, action_horizon // 2)
#     current_action_index = 0
#     current_action_sequence = None

#     while episode_count < num_episodes:
#         if current_action_sequence is None or current_action_index >= half_horizon:
#             print("Calling policy check")
#             current_action_sequence = policy_fn(observations)

#             # Handle batched output
#             if isinstance(current_action_sequence, np.ndarray) and current_action_sequence.ndim == 3:
#                 # Shape (1, H, D) → remove batch dim
#                 current_action_sequence = current_action_sequence[0]

#             # If the policy returns only one action instead of a horizon
#             if current_action_sequence.ndim == 1:
#                 # shape (D,) → promote to horizon 1: (1, D)
#                 current_action_sequence = current_action_sequence[None, :]

#             current_action_index = 0
#         actions = current_action_sequence[current_action_index]
#         step_variables = env.step(actions)
#         if len(step_variables) == 4:
#             next_observations, rewards, dones, infos = step_variables
#         else:
#             next_observations, rewards, dones, truncated, infos = step_variables
#             dones = np.logical_or(dones, truncated)

#         if save_video and episode_count < max_episodes_for_video:
#             images = env.render()

#         step_index += 1
#         log_progress = False
#         obs = observations
#         next_obs = next_observations
#         transition = dict(
#             observation=obs,
#             next_observation=next_obs,
#             action=actions,
#             reward=rewards,
#             done=dones,
#             info={},  # match vectorized (info dropped / empty)
#         )
#         if save_video and episode_count < max_episodes_for_video:
#             transition["image"] = images.copy()
#         add_to(trajectories[-1], transition)

#         if dones:
#             episode_count += 1
#             step_index = 0
#             log_progress = True

#             if episode_count < num_episodes:
#                 # start a new episode: create a new trajectory and reset env
#                 trajectories.append(defaultdict(list))
#                 observations = env.reset()
#                 if isinstance(observations, tuple) and len(observations) == 2:
#                     observations, _ = observations
#             else:
#                 # don't reset if we're done with all episodes
#                 observations = next_observations
#         else:
#             observations = next_observations

#         if log_progress:
#             logging.info(
#                 f"Completed {episode_count} out of {num_episodes} eval episodes..."
#             )
#     return trajectories

def evaluate_with_trajectories_libero(
    policy_fn,
    env,
    num_episodes: int,
    save_video: bool = False,
    max_episodes_for_video: int = 2,
    action_horizon: int = 10,
    use_full_horizon_for_refill: bool = False,
):
    H = action_horizon
    half_H = max(1, H // 2)
    if use_full_horizon_for_refill:
        half_H = H

    trajectories = [defaultdict(list)]
    episode_count = 0

    observations = env.reset()
    if isinstance(observations, tuple) and len(observations) == 2:
        observations, _ = observations

    step_index = 0

    # horizon-related state for this single env
    current_action_index = 0          # where we are inside current horizon
    current_action_sequence = None    # shape (H, D) or (1, D)
    current_diffusion_actions = None

    q_vs_mc_returns_vals = []
    current_vlm_output = None
    curr_episode_q_vs_mc_returns_vals = []

    while episode_count < num_episodes:
        # print("q_vs_mc_returns_vals", len(q_vs_mc_returns_vals))
        # print("curr_episode_q_vs_mc_returns_vals", len(curr_episode_q_vs_mc_returns_vals))
        # print("--------------------------------")

        current_vlm_output = None
        current_actions_for_vlm_output = None
        # ---------------------------------------------------------
        # 1. Call policy when we need a refill
        # ---------------------------------------------------------
        if current_action_sequence is None or current_action_index >= half_H:
            # print("Calling policy check")
            # breakpoint()
            # try:
            # breakpoint()
            out_dict = policy_fn(observations)
            current_action_sequence = out_dict['actions']
            current_vlm_output = out_dict['vlm_output']
            current_diffusion_actions = out_dict['diffusion_actions']
            current_actions_for_vlm_output = out_dict['actions']
            
            # state = observations['proprio'][:, :8] # State dimension
            # breakpoint()
            curr_episode_q_vs_mc_returns_vals.append((current_vlm_output, current_action_sequence))
            # except Exception as e:
            #     print(f"Error in policy_fn: {e}")
            #     current_action_sequence = policy_fn(observations)

            # Expect one env. If policy returns (1, H, D) → (H, D)
            if isinstance(current_action_sequence, np.ndarray):
                if current_action_sequence.ndim == 3:
                    # assume (1, H, D)
                    current_action_sequence = current_action_sequence[0]

                # If policy returns only a single action: (D,)
                if current_action_sequence.ndim == 1:
                    # → (1, D)
                    current_action_sequence = current_action_sequence[None, :]

            # Basic sanity: at this point we expect (T, D)
            if not isinstance(current_action_sequence, np.ndarray) or current_action_sequence.ndim != 2:
                raise ValueError(
                    f"Policy must return array of shape (T, D) or (1, T, D) or (D,), "
                    f"got type={type(current_action_sequence)}, shape="
                    f"{getattr(current_action_sequence, 'shape', None)}"
                )

            # Optionally enforce minimum horizon, like your vectorized version
            if current_action_sequence.shape[0] < H:
                raise ValueError(
                    f"Policy returned horizon={current_action_sequence.shape[0]}, expected >= {H}"
                )

            current_action_index = 0

        # ---------------------------------------------------------
        # 2. Pick the action for this step
        # ---------------------------------------------------------
        actions = current_action_sequence[current_action_index]
        diffusion_actions = current_diffusion_actions[current_action_index]
        current_action_index += 1

        # ---------------------------------------------------------
        # 3. Step environment
        # ---------------------------------------------------------
        step_variables = env.step(actions)

        if len(step_variables) == 4:
            next_observations, rewards, dones, infos = step_variables
        else:
            next_observations, rewards, dones, truncated, infos = step_variables
            dones = np.logical_or(dones, truncated)

        if save_video and episode_count < max_episodes_for_video:
            images = env.render()

        step_index += 1
        log_progress = False

        # ---------------------------------------------------------
        # 4. Store transition
        # ---------------------------------------------------------
        obs = observations
        next_obs = next_observations

        transition = dict(
            observation=obs,
            next_observation=next_obs,
            action=actions,
            diffusion_action=diffusion_actions,
            reward=rewards,
            done=dones,
            info={},  # to match vectorized version
        )

        if current_vlm_output is not None:
            transition['vlm_output'] = current_vlm_output
        if current_actions_for_vlm_output is not None:
            transition['actions_for_vlm_output'] = current_actions_for_vlm_output

        if save_video and episode_count < max_episodes_for_video:
            transition["image"] = images.copy()

        add_to(trajectories[-1], transition)

        # ---------------------------------------------------------
        # 5. Episode end handling
        # ---------------------------------------------------------
        if dones:
            if episode_count < num_episodes:
                if len(curr_episode_q_vs_mc_returns_vals) > 0:
                    q_vs_mc_returns_vals.append(curr_episode_q_vs_mc_returns_vals)
                    # breakpoint()
                    current_vlm_output = None
                    curr_episode_q_vs_mc_returns_vals = []
            
            episode_count += 1
            step_index = 0
            log_progress = True

            # IMPORTANT: reset horizon state on episode boundary
            current_action_index = 0
            current_action_sequence = None

            if episode_count < num_episodes:
                trajectories.append(defaultdict(list))
                
                observations = env.reset()
                if isinstance(observations, tuple) and len(observations) == 2:
                    observations, _ = observations
            else:
                observations = next_observations
        else:
            observations = next_observations

        if log_progress:
            logging.info(
                f"Completed {episode_count} out of {num_episodes} eval episodes..."
            )

    # breakpoint()
    if len(q_vs_mc_returns_vals) > 0:
        return trajectories, q_vs_mc_returns_vals
    else:
        return trajectories


# --- utility ---
def add_to(container, data):
    for k, v in data.items():
        container[k].append(v)


def save_rollout_gif(frames, save_dir, step_i, rollout_j, fps=10):
    """
    Save rollout frames as a GIF in structure: save_dir/eval_gifs/step_{i}/rollout_{j}.gif

    Args:
        frames (list[np.ndarray]): list of RGB frames (H, W, 3)
        save_dir (str): base directory
        step_i (int): current training step
        rollout_j (int): rollout index within step_i
        fps (int): frames per second for the gif
    """
    # Construct full directory path
    step_dir = os.path.join(save_dir, f"eval_gifs/step_{step_i}")
    os.makedirs(step_dir, exist_ok=True)

    # Ensure frames are uint8
    frames_uint8 = [np.uint8(f) for f in frames]

    # Save the gif
    duration = 1 / fps
    gif_path = os.path.join(step_dir, f"rollout_{rollout_j}.gif")
    iio.imwrite(gif_path, frames_uint8, plugin="pillow", duration=duration)
    print(f"Saved: {gif_path}")