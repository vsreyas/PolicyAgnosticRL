from collections import defaultdict
from typing import Optional

import numpy as np
import copy
from jaxrl_m.common.evaluation import add_to


class TrajSampler(object):
    """
    sampling trajectories for training or eval, with the option of adding the
    trajectories into the replay buffer.
    Apply reward and action modifications on the trajectories returned.
    """

    def __init__(
        self, env, clip_action, reward_scale, reward_bias, max_traj_length=10, action_horizon=1,
    ):
        self.clip_action = clip_action
        self.reward_scale = reward_scale
        self.reward_bias = reward_bias
        self.max_traj_length = max_traj_length
        self._env = env
        self.action_horizon = action_horizon

    def sample(
        self,
        fns_dict, # Dictionary containing access to different functions for the agent for flexibility 
        num_episodes,
        replay_buffer=None,
        goal_relabel_fn: Optional[callable] = None,
        calc_mc_return_fn: Optional[callable] = None,
        store_max_trajectory_reward: bool = False,
        terminate_on_success: bool = False,
    ):
        """
        args:
            replay_buffer: if not None, insert the trajectories into the replay buffer
            goal_relabel_fn: if not None, relabel the goal of each transition and store in
                                transition["goals"].
                                This function should be f(trajectory) -> traj
                                the policy_fn should be f(observation, goal) -> action
            calc_mc_return_fn: if not None, calculate the MC return and add to each transition.
                                This function should be f(rewards, dones) -> mc_returns
            store_max_trajectory_reward: if True, store the max reward of the trajectory on the
                                replay buffer.
        """
        # Extract functions from fns_dict #
        policy_fn = fns_dict["policy_fn"]

        trajectories = []
        q_vs_mc_returns_vals = []
        H = self.action_horizon
        half_H = max(1, H // 2)

        assert H % half_H == 0, "H must be divisible by half_H to ensure vlm_output is properly built up"

        for _ in range(num_episodes):
            trajectory = defaultdict(list)
            # print("Starting new trajectory")
            reset_variables = self._env.reset()
            if isinstance(reset_variables, np.ndarray) or isinstance(
                reset_variables, dict
            ):
                observation = reset_variables
                info = {}
            else:
                assert len(reset_variables) == 2
                observation, info = reset_variables
            done = False
            step = 0
            current_action_index = 0
            current_action_sequence = None
            padding_dict = None

            current_vlm_output = None
            curr_episode_q_vs_mc_returns_vals = []
            valid_timesteps_for_action_chunk = []

            while not done and step < self.max_traj_length:
                # breakpoint()
                # from PIL import Image; Image.fromarray(observation['image'].astype('uint8')).save('image.png')
                # print(f"Step {step}")
                # print(f"Done: {done}")
                # print(f"Max traj length: {self.max_traj_length}")
                
                observation_storing = copy.deepcopy(observation)
                # print("deepcopy done")

                # current_vlm_output = vlm_output_fn(observation)
                # out_dict = {
                #     "vlm_output": current_vlm_output,
                # }
                # out_dict = policy_fn(observation)
                # current_vlm_output = out_dict["vlm_output"]
                
                if padding_dict is not None:
                    out_dict = {}
                    for k in padding_dict.keys():
                        out_dict[k] = np.zeros(padding_dict[k])
                # print("padding dict done")
                
                if current_action_sequence is None or current_action_index >= half_H:
                    if goal_relabel_fn is not None:
                        current_action_sequence = policy_fn(observation, info.get("goal"))
                    else:
                        # try:
                        out_dict = policy_fn(observation)

                        if padding_dict is None:
                            padding_dict = {}
                            for k in out_dict.keys():
                                padding_dict[k] = out_dict[k].shape
                        
                        current_action_sequence = out_dict["actions"]
                        current_vlm_output = out_dict["vlm_output"]

                        valid_timesteps_for_action_chunk.append(step)
                    
                    # print("policy fn completed")

                    # Normalize shapes:
                    if isinstance(current_action_sequence, np.ndarray):
                        # case: (1, H, D)
                        if current_action_sequence.ndim == 3:
                            current_action_sequence = current_action_sequence[0]

                        # case: (D,)
                        if current_action_sequence.ndim == 1:
                            current_action_sequence = current_action_sequence[None, :]

                    # Must now be (T, D)
                    if (
                        not isinstance(current_action_sequence, np.ndarray)
                        or current_action_sequence.ndim != 2
                    ):
                        raise ValueError(
                            f"Policy must return array of shape (T, D) or (1, T, D) or (D,), "
                            f"got type={type(current_action_sequence)}, shape="
                            f"{getattr(current_action_sequence, 'shape', None)}"
                        )

                    if current_action_sequence.shape[0] < H:
                        raise ValueError(
                            f"Policy returned horizon={current_action_sequence.shape[0]}, expected >= {H}"
                        )

                    current_action_index = 0

                    if current_vlm_output is not None:
                        curr_episode_q_vs_mc_returns_vals.append((current_vlm_output, current_action_sequence))


                action = current_action_sequence[current_action_index]
                current_action_index += 1

                step_variables = self._env.step(action)
                if len(step_variables) == 5:
                    next_observation, r, terminated, truncated, info = step_variables
                    done = terminated or truncated
                else:
                    assert len(step_variables) == 4
                    next_observation, r, done, info = step_variables
                
                # print("Completed env step")
                # print("Done: ", done)
                
                if done:
                    terminals = len(trajectory['rewards']) < self._env.max_steps
                    truncates = not terminals

                    out_dict_final_observation = policy_fn(next_observation)

                    # Compute value of final state in case it is part of returned data #
                    if 'state_values' in out_dict_final_observation:
                        assert "value_fn" in fns_dict
                        value_fn = fns_dict["value_fn"]
                        final_state_value = value_fn(out_dict_final_observation["vlm_output"])
                        trajectory["state_values"].append(final_state_value)

                    if terminals:
                        # Pick the last valid timestep for vlm_output and update it #
                        last_valid_timestep_for_action_chunk = len(trajectory['rewards']) - H + 1
                        out_dict_last_valid_timestep = policy_fn(trajectory['observations'][last_valid_timestep_for_action_chunk])
                        out_dict_last_valid_timestep.pop('actions')
                        for k in out_dict_last_valid_timestep.keys():
                            trajectory[k][last_valid_timestep_for_action_chunk] = out_dict_last_valid_timestep[k]
                        
                        if 'state_values' in out_dict_final_observation:
                            state_values_last_valid_timestep = value_fn(out_dict_last_valid_timestep["vlm_output"])
                            trajectory["state_values"][last_valid_timestep_for_action_chunk] = state_values_last_valid_timestep
                        
                        valid_timesteps_for_action_chunk.append(last_valid_timestep_for_action_chunk)
                else:
                    terminals = False
                    truncates = False
                
                transition = dict(
                    observations=observation_storing,
                    next_observations=next_observation,
                    actions=np.clip(action, -self.clip_action, self.clip_action),
                    rewards=r * self.reward_scale + self.reward_bias,
                    terminals=terminals,
                    truncates=truncates,
                    masks=1.0 - terminals, # Bootstrap to next state in truncates case
                )

                if 'actions' in out_dict:
                    out_dict.pop('actions')
                
                # Add out_dict keys to transition #
                for key in out_dict.keys():
                    transition[key] = out_dict[key]

                add_to(trajectory, transition)
                # print("add to trajectory done")

                # terminate on success
                if (
                    terminate_on_success
                    and (r * self.reward_scale + self.reward_bias) == 0
                ):
                    current_action_sequence = None
                    current_action_index = 0
                    break

                observation = next_observation
                step += 1
                # print("step incremented")
                # print("---")

            # get MC return
            if calc_mc_return_fn is not None:
                mc_returns = calc_mc_return_fn(
                    trajectory["rewards"], trajectory["masks"]
                )
                trajectory["mc_returns"] = mc_returns

            # goal relabeling
            if goal_relabel_fn is not None:
                trajectory = goal_relabel_fn(trajectory)

            # insert into replay buffer
            if replay_buffer is not None:
                for i in range(len(trajectory["rewards"])):
                    transition = dict(
                        observations=trajectory["observations"][i],
                        actions=trajectory["actions"][i],
                        rewards=trajectory["rewards"][i],
                        next_observations=trajectory["next_observations"][i],
                        masks=trajectory["masks"][i],
                    )
                    if goal_relabel_fn is not None:
                        transition["goals"] = trajectory["goals"][i]
                    if calc_mc_return_fn is not None:
                        transition["mc_returns"] = trajectory["mc_returns"][i]
                    if store_max_trajectory_reward:
                        transition["max_trajectory_rewards"] = np.max(
                            trajectory["rewards"]
                        )

                    replay_buffer.insert(transition)

            if 'vlm_output' in out_dict_final_observation:
                trajectory['vlm_output'].append(out_dict_final_observation['vlm_output'])
                trajectory['diffusion_actions'].append(out_dict_final_observation['diffusion_actions'])
            
            valid_timesteps_for_action_chunk = sorted(valid_timesteps_for_action_chunk)
            trajectory['valid_timesteps_for_action_chunk'] = valid_timesteps_for_action_chunk

            trajectories.append(trajectory)
            # breakpoint()
            
            if len(curr_episode_q_vs_mc_returns_vals) > 0:
                q_vs_mc_returns_vals.append(curr_episode_q_vs_mc_returns_vals)
        
        if len(q_vs_mc_returns_vals) > 0:
            return trajectories, q_vs_mc_returns_vals
        
        return trajectories


def calc_return_to_go(rewards, masks, gamma, push_failed_to_min=True, min_reward=0.0):
    """default calc return_to_go function"""
    if rewards[-1] == min_reward and push_failed_to_min:
        # failed trajectory
        assert np.all(np.array(rewards) <= 0)
        reward_to_go = [min_reward / (1 - gamma)] * len(rewards)
    else:
        # success trajectory
        reward_to_go = [0] * len(rewards)
        prev_return = 0
        for i in range(len(rewards)):
            reward_to_go[-i - 1] = rewards[-i - 1] + gamma * prev_return * masks[-i - 1]
            prev_return = reward_to_go[-i - 1]

    return np.array(reward_to_go, dtype=np.float32)
