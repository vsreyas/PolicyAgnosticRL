"""Script for offline to online RL."""

# Try increasing the number of open files limit
import resource
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(65535, hard), hard))


import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Union

import cv2
import flax
import gym
import jax
# jax.config.update("jax_log_compiles", True)
# jax.config.update("jax_explain_cache_misses", True)
# jax.config.update("jax_traceback_filtering", "off")  # more context in logs

import jax.numpy as jnp
import numpy as np
import seaborn as sns
import tensorflow as tf
import io
from PIL import Image
import pickle
import gc

import wandb
from absl import app, flags, logging
from matplotlib import pyplot as plt
from ml_collections import config_flags
from tqdm import tqdm
import functools

from jaxrl_m.agents import agents
from jaxrl_m.agents.continuous.action_optimization import (
    LocalOptimizationState,
    action_optimization_sample_actions,
    add_base_policy_actions_to_batch,
    local_optimization_steps as take_local_optimization_steps,
)
from jaxrl_m.agents.continuous.auto_regressive_transformer import (
    AutoRegressiveTransformerAgent,
)
from jaxrl_m.agents.continuous.base_policy import BasePolicy, BasePolicyTypes
from jaxrl_m.agents.continuous.ddpm_bc import DDPMBCAgent
from jaxrl_m.agents.continuous.openvla import OpenVLAAgent
from jaxrl_m.agents.continuous.pi_0 import PiPolicy
from jaxrl_m.common.common import JaxRLTrainState
from jaxrl_m.common.evaluation import evaluate_with_trajectories_vectorized, supply_rng, evaluate_with_trajectories_libero, save_rollout_gif
from jaxrl_m.common.traj import TrajSampler, calc_return_to_go
from jaxrl_m.common.typing import Batch, Data
from jaxrl_m.common.wandb import WandBLogger
from jaxrl_m.data.bridge_dataset import (
    BridgeDataset,
    get_task_to_initial_eep,
    glob_to_path_list,
)
from jaxrl_m.data.image_replay_buffer_pi import (
    ImageReplayBufferPi,
    save_trajectory_as_tfrecord,
)
from jaxrl_m.data.replay_buffer import ReplayBuffer
# D4RL is not needed for Libero EXPO; keep optional to avoid Mujoco dependency errors.
# from jaxrl_m.envs.d4rl import TruncationWrapper, get_d4rl_dataset_with_mc_calculation
from jaxrl_m.utils.timer_utils import Timer
from jaxrl_m.utils.train_utils import concatenate_batches, load_recorded_video
from jaxrl_m.vision import encoders
from jaxrl_m.utils.train_utils import preprocess_action, repack_action
from jaxrl_m.agents.continuous.pi_vlm_cached import (
    create_agent,
    PiResidualTD3Cache,
    PiResidualPPOCache,
)
from jaxrl_m.utils.expo_utils import calc_mc_return_fn
from jaxrl_m.envs.libero import StepTimeout, time_limit, STEP_TIME_LIMIT

try:
    from jax_smi import initialise_tracking  # type: ignore

    initialise_tracking()
except ImportError:
    pass

from openpi.training.config import get_config


print("\n\n\n IMPORTS DONE \n\n\n")

FLAGS = flags.FLAGS

flags.DEFINE_string("environment_name", "", "Environment name.")
flags.DEFINE_string("wandb_project_name", "PI-0.5-finetuning", "WandB project name.") #"PA-RL""debug"
flags.DEFINE_string("wandb_experiment_name", "", "WandB experiment name.")
flags.DEFINE_string("wandb_group", "", "WandB group.")
flags.DEFINE_string("agent_name", "pi_residual_td3", "Agent name (pi_residual_td3 or pi_residual_ppo).")
config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)
config_flags.DEFINE_config_file(
    "bridgedata_config",
    None,
    "File path to the bridgedata configuration.",
    lock_config=False,
)
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_float("reward_scale", 1.0, "Reward scale.")
flags.DEFINE_float("reward_bias", 0.0, "Reward bias.")
flags.DEFINE_float("clip_action", 200.0, "Clip action.")
flags.DEFINE_integer("num_parallel_envs", 1, "Number of parallel environments.")
flags.DEFINE_bool("debug", False, "Debug config")
flags.DEFINE_string("resume_path", None, "Resume training from checkpoint.")
flags.DEFINE_integer("max_episode_steps", 1000, "Maximum episode steps.")

# Q-diffusion pre-processing flags
flags.DEFINE_integer(
    "num_online_trajectories_per_epoch",
    1,
    "Number of trajectories collected from interaction per online epoch.",
)
flags.DEFINE_integer(
    "num_warmup_trajectories",
    1,
    "Number of trajectories collected from interaction before training.",
)
flags.DEFINE_integer("critic_utd", 1, "update-to-data ratio of the critic")
flags.DEFINE_float(
    "base_policy_utd",
    1,
    "Update-to-data ratio of the base policy for distillation.",
)
flags.DEFINE_string(
    "base_policy_offline_cache_path",
    None,
    "Path to pre-computed base policy actions to use for pre-training.",
)
flags.DEFINE_string(
    "task_name",
    "put both moka pots on the stove", 
    "Name of fixed task"
)
flags.DEFINE_bool(
    "plot_q_values_over_trajectory_figure",
    False,
    "Plot Q-values over trajectory time step.",
)
flags.DEFINE_bool(
    "use_lang",
    True,
    "Use language conditioning."
)
flags.DEFINE_integer(
    "num_edit_samples",
    4,
    "Number of edit samples to use for editing the actions.",
)
flags.DEFINE_integer(
    "num_actions_to_sample",
    4,
    "Number of actions to sample for the policy.",
)
flags.DEFINE_integer(
    "online_trajectory_collection_frequency",
    10, # Every 10 update steps, collect 1 trajectory from environment #
    "Frequency of online trajectory collection.",
)
flags.DEFINE_integer(
    "warmup_steps",
    100, # Warmup critic for 100 update steps #
    "Number of steps to warmup critic.",
)
flags.DEFINE_bool(
    "final_step_sparse_reward",
    False,
    "Use sparse reward for the final step.",
)
flags.DEFINE_bool(
    "use_wrist_view",
    True,
    "Use Wrist view camera."
)
flags.DEFINE_string(
    "params_path",
    None,
    "Path to the parameters to load.",
)
flags.DEFINE_bool(
    "filter_successful_trajectories",
    False,
    "Filter successful trajectories.",
)
flags.DEFINE_float(
    "scale_success_alpha",
    -1,
    "Alpha for the reward scaling factor.",
)
flags.DEFINE_float(
    "exploration_epsilon",
    0.05,
    "Exploration epsilon.",
)
flags.DEFINE_integer(
    "num_trajectories_to_collect",
    10,
    "Number of trajectories to collect.",
)
flags.DEFINE_integer(
    "num_train_steps",
    1000000,
    "Number of training steps.",
)
flags.DEFINE_float(
    "intermediate_reward_mul_factor",
    1.0,
    "Multiplier for the intermediate reward.",
)
flags.DEFINE_bool(
    "balance_offline_training_data",
    False,
    "Balance offline training data by downsampling the more frequent trajectories among success/failed ones.",
)
flags.DEFINE_bool(
    "on_policy",
    False,
    "Flag for whether we want to run on policy updates"
)
flags.DEFINE_float(
    "critic_success_wt",
    1.0,
    "Weight to multiply successful trajectories in a batch to account for imbalanced data during critic training",
)
flags.DEFINE_float(
    "bc_loss_coef",
    1000.0,
    "Weight to multiply successful trajectories in a batch with bc loss",
)
flags.DEFINE_bool(
    "load_action_samples",
    False,
    "Flag for whether we want to load action samples for each state"
)
flags.DEFINE_string(
    "critic_warmup_type",
    "sarsa",
    "Type of critic warmup: 'sarsa' for SARSA loss or 'calql' for Cal-QL loss with MC return lower bound."
)
flags.DEFINE_float(
    "calql_lower_bound",
    -1000.0,
    "Constant lower bound for Q-values in Cal-QL warmup. Used instead of MC returns."
)
flags.DEFINE_float(
    "cql_alpha",
    5.0,
    "CQL regularization strength for Cal-QL warmup."
)
flags.DEFINE_float(
    "cql_temp",
    1.0,
    "Temperature for logsumexp in CQL/Cal-QL loss."
)
flags.DEFINE_integer(
    "calql_random_actions",
    0,
    "Number of random actions in [-1, 1] to sample for Cal-QL OOD pessimism loss."
)
flags.DEFINE_float(
    "grpo_beta",
    1.0,
    "Temperature for advantage weighting in GRPO actor loss."
)
flags.DEFINE_float(
    "grpo_weight_threshold",
    0.0,
    "Threshold for zeroing out small weights in GRPO loss (after exponentiation)."
)
flags.DEFINE_integer(
    "num_diffusion_samples",
    4,
    "Number of action samples from base policy for GRPO training."
)
flags.DEFINE_integer(
    "edit_actor_utd_ratio",
    1,
    "Update-to-data ratio for the edit actor."
)
flags.DEFINE_bool(
    "ws_critic",
    False,
    "Whether to warmstart the critic during warmup phase."
)
flags.DEFINE_bool(
    "ws_edit_actor",
    False,
    "Whether to warmstart the edit actor during warmup phase."
)
flags.DEFINE_string(
    "pi_config_name",
    None,
    "Name of the PI config to use (required)."
)
flags.DEFINE_integer(
    "num_balanced_trajectories",
    -1,
    "Max number of successful and failed trajectories to keep after balancing. -1 means no limit."
)

### Try subprocenv ###
import multiprocessing as mp
from multiprocessing.connection import wait

def env_worker(conn, make_env_fn):
    # Import robosuite / libero INSIDE the subprocess
    env = make_env_fn()
    try:
        while True:
            msg = conn.recv()
            cmd = msg[0]
            if cmd == "reset":
                conn.send(env.reset())
            elif cmd == "get_attr":
                conn.send(getattr(env, "max_steps"))
            elif cmd == "step":
                action = msg[1]
                conn.send(env.step(action))
            elif cmd == "close":
                try:
                    env.close()
                except Exception:
                    pass
                conn.send(None)
                break
            else:
                raise RuntimeError(f"Unknown cmd: {cmd}")
    finally:
        try:
            env.close()
        except Exception:
            pass

class SubprocEnv:
    def __init__(self, make_env_fn):
        self.make_env_fn = make_env_fn
        self._ctx = mp.get_context("spawn")
        self._start()

    def _start(self):
        self.parent_conn, child_conn = self._ctx.Pipe()
        self.proc = self._ctx.Process(target=env_worker, args=(child_conn, self.make_env_fn), daemon=True)
        self.proc.start()

    def _restart(self):
        self.kill()
        self._start()

    def kill(self):
        try:
            if self.proc.is_alive():
                self.proc.terminate()
                self.proc.join(timeout=5)
        finally:
            try:
                self.parent_conn.close()
            except Exception:
                pass

    def call(self, cmd, *args, timeout_s=300):
        self.parent_conn.send((cmd, *args))
        ready = wait([self.parent_conn], timeout=timeout_s)
        if not ready:
            # hung in native code -> hard reset
            self._restart()
            raise TimeoutError(f"{cmd} timed out after {timeout_s}s")
        return self.parent_conn.recv()

    def reset(self, timeout_s=300):
        return self.call("reset", timeout_s=timeout_s)

    def step(self, action, timeout_s=300):
        return self.call("step", action, timeout_s=timeout_s)

    def close(self):
        try:
            self.call("close", timeout_s=5)
        except Exception:
            pass
        self.kill()
    
    def get_attr(self, name: str):
        return self.call("get_attr", name)

    @property
    def max_steps(self):
        # assumes underlying env exposes `max_steps`
        return self.get_attr("max_steps")

def make_env(task_name):
    from jaxrl_m.envs.libero import get_libero_env, get_libero_config
    cfg = get_libero_config()
    return get_libero_env(cfg=cfg, task_name=task_name, is_pi=True)
########################################################################################################################

# 2: 07 2 13
BASE_POLICY_TYPE_TO_CLASS = {
    BasePolicyTypes.OpenVLA: OpenVLAAgent,
    BasePolicyTypes.DDPM: DDPMBCAgent,
    BasePolicyTypes.AutoRegressiveTransformer: AutoRegressiveTransformerAgent,
    BasePolicyTypes.Pi0: PiPolicy
}

devices = jax.local_devices()

def get_policy_fn(
    agent,
    rng: jax.random.PRNGKey,
    timer: Timer | None = None,
    deterministic_actions: bool = False,
    num_diffusion_samples: int = 1,
) -> Callable[[Data], np.ndarray]:
    def policy_fn(observations: Data, *args, **kwargs) -> np.ndarray:
        if not isinstance(observations, dict):
            observations = {"state": observations}
        
        out_dict = jax.device_get(
            agent.sample_actions(
                observations, *args, **kwargs, timer=timer, output_action_chunk=True, 
                use_deterministic_actions=deterministic_actions,
                num_diffusion_samples=num_diffusion_samples,
            )
        )

        return out_dict

    policy_fn = supply_rng(policy_fn, rng=rng)

    return policy_fn

def get_base_policy_fn(
    agent,
    rng: jax.random.PRNGKey,
    num_diffusion_samples: int = 1,
) -> Callable[[Data], np.ndarray]:
    def policy_fn(observations: Data, *args, **kwargs) -> np.ndarray:
        if not isinstance(observations, dict):
            observations = {"state": observations}
        
        out_dict = jax.device_get(
            agent.sample_base_actions(
                observations, *args, **kwargs,
                num_diffusion_samples=num_diffusion_samples,
            )
        )

        return out_dict

    policy_fn = supply_rng(policy_fn, rng=rng)

    return policy_fn

def get_value_fn(
    agent: PiResidualPPOCache,
    rng: jax.random.PRNGKey,
    timer: Timer | None = None,
) -> Callable[[Data], np.ndarray]:
    def _value_fn(vlm_output_with_state: np.ndarray, *args, **kwargs) -> np.ndarray:
        return jax.device_get(agent.get_state_values(vlm_output_with_state, *args, **kwargs))

    value_fn = supply_rng(_value_fn, rng=rng)

    return value_fn

def get_vlm_output_fn(
    agent,
    rng: jax.random.PRNGKey,
    timer: Timer | None = None,
    debug_mode: bool = False,
) -> Callable[[Data], np.ndarray]:
    def policy_fn(observations: Data, *args, **kwargs) -> np.ndarray:
        if not isinstance(observations, dict):
            observations = {"state": observations}
        if "state" in observations:
            obs_ndim = observations["state"].ndim
        else:
            assert "proprio" in observations
            obs_ndim = observations["proprio"].ndim
        
        vlm_output = jax.device_get(
            agent.get_vlm_output(
                observations, *args, **kwargs, timer=timer, output_action_chunk=True, debug_mode=debug_mode
            )
        )

        return vlm_output

    policy_fn = supply_rng(policy_fn, rng=rng)

    return policy_fn


def balance_offline_training_data(offline_dset_paths):
    # Inspect tf records, check if trajectory is successful or not and subsmple more frequent ones.
    # Returns a list of paths to the balanced offline training data.
    success_trajectories = []
    failed_trajectories = []

    for path in offline_dset_paths:
        dataset = tf.data.TFRecordDataset(path)
        for raw_record in dataset:
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            found_keys = list(example.features.feature.keys())
            features_dict = {
                k: tf.io.FixedLenFeature([], tf.string)
                for k in found_keys
            }
            parsed_features = tf.io.parse_single_example(raw_record, features_dict)
            raw_bytes = parsed_features["terminals"]
            dtype = tf.float32
            tensor = tf.io.parse_tensor(raw_bytes, out_type=dtype)

            if tensor.numpy().sum() > 0:
                success_trajectories.append(path)
            else:
                failed_trajectories.append(path)
    
    # Downsample the more frequent trajectories among success/failed ones.
    if len(success_trajectories) > len(failed_trajectories):
        success_trajectories = np.random.choice(success_trajectories, size=len(failed_trajectories), replace=False)
        success_trajectories = list(success_trajectories)
    elif len(success_trajectories) < len(failed_trajectories):
        failed_trajectories = np.random.choice(failed_trajectories, size=len(success_trajectories), replace=False)
        failed_trajectories = list(failed_trajectories)
    
    # Further limit to num_balanced_trajectories if set
    if FLAGS.num_balanced_trajectories > 0:
        if len(success_trajectories) > FLAGS.num_balanced_trajectories:
            success_trajectories = np.random.choice(success_trajectories, size=FLAGS.num_balanced_trajectories, replace=False)
            success_trajectories = list(success_trajectories)
        if len(failed_trajectories) > FLAGS.num_balanced_trajectories:
            failed_trajectories = np.random.choice(failed_trajectories, size=FLAGS.num_balanced_trajectories, replace=False)
            failed_trajectories = list(failed_trajectories)
    
    print("Balanced offline training data: Success trajectories: ", len(success_trajectories), "Failed trajectories: ", len(failed_trajectories))
    print("\n\n\n\n\n")

    return success_trajectories + failed_trajectories

def train_agent(_):
    # Validate required flags
    assert FLAGS.pi_config_name is not None, "--pi_config_name must be provided"
    
    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    os.environ["WANDB__SERVICE_WAIT"] = "300"
    os.environ["WANDB_INIT_TIMEOUT"] = "120"
    wandb.require("core")
    devices = jax.local_devices()
    num_devices = len(devices)
    assert FLAGS.config.batch_size % num_devices == 0

    # Get PI config #
    pi_config = get_config(FLAGS.pi_config_name)
    pi_config.fsdp_devices = 1 # Try out with model parallel
    pi_config.exp_name = FLAGS.wandb_experiment_name
    pi_config.overwrite = True

    # LOG: WANDB setup #
    if FLAGS.wandb_project_name is not None:
        wandb_config = WandBLogger.get_default_config()
        wandb_config.update(
            {
                "project": FLAGS.wandb_project_name,
                "exp_descriptor": FLAGS.wandb_experiment_name,
                "tag": None,
                "group": FLAGS.wandb_group,
            }
        )
        wandb_logger = WandBLogger(
            wandb_config=wandb_config,
            variant=FLAGS.config.to_dict(),
            debug=FLAGS.debug,
            allow_val_change=True,
        )
        save_dir = tf.io.gfile.join(
            (
                os.path.abspath(FLAGS.config.save_dir)
                if "gs://" not in FLAGS.config.save_dir
                else FLAGS.config.save_dir
            ),
            # wandb_logger.config.project,
            # wandb_logger.config.exp_descriptor,
            f"seed_{FLAGS.seed}",
        )
    else:
        wandb_logger = None
        save_dir = tf.io.gfile.join(
            os.path.abspath(FLAGS.config.save_dir),
        )
        FLAGS.config.wandb_enabled = False
    
    ####### Dataset and evironment setup ###########
    if FLAGS.environment_name=="libero":
        assert FLAGS.config.image_observations
        from jaxrl_m.envs.libero import (
            get_libero_tfrecord_dataset,
        )

        if len(FLAGS.config.libero_tfrecord_regexp) > 0:
            paths = None
            if FLAGS.balance_offline_training_data:
                offline_dset_paths = glob_to_path_list(FLAGS.config.libero_tfrecord_regexp)
                paths = balance_offline_training_data(offline_dset_paths)

            dataset = get_libero_tfrecord_dataset(
                tfrecord_regexp=FLAGS.config.libero_tfrecord_regexp, use_wrist_view=FLAGS.use_wrist_view, 
                use_language=FLAGS.use_lang, config=pi_config, is_pi=True, **FLAGS.config.dataset_kwargs,
                task_name=FLAGS.task_name, 
                final_step_sparse_reward=FLAGS.final_step_sparse_reward,
                filter_successful_trajectories=FLAGS.filter_successful_trajectories,
                use_reverse_data_paths=False,
                alpha=FLAGS.scale_success_alpha,
                scale_success_reward=FLAGS.scale_success_alpha > 0,
                intermediate_reward_mul_factor=FLAGS.intermediate_reward_mul_factor,
                drop_images_from_output=True, # Do not need it as we are caching things at trajectory generation time #
                paths=paths, # If `None` then constructs paths based on `tfrecord_regexp`
                offline_flag=1.0,
                **FLAGS.config.offline_image_replay_buffer_kwargs,
            )
        else:
            dataset = None

        train_env = SubprocEnv(make_env_fn=functools.partial(make_env, task_name=FLAGS.task_name))
        eval_env = SubprocEnv(make_env_fn=functools.partial(make_env, task_name=FLAGS.task_name))
    else:
       raise NotImplementedError
    if FLAGS.config.image_observations:
        image_replay_buffer = None  # Will be created when switching to online training.
        state_replay_buffer = None

    rng = jax.random.PRNGKey(FLAGS.seed)
    np.random.seed(FLAGS.seed)  # Seed NumPy for reproducible dataset balancing

    if dataset is not None:
        offline_train_iterator = dataset.iterator(
            batch_size=FLAGS.config.batch_size
        )
    else:
        offline_train_iterator = None
    
    # Online dataset/buffer #
    # Online iterators will be set when switching to online training.
    online_train_iterator = None
    #########################################################
    
    ### Create trajectory sampler ###
    data_collection_trajectory_sampler = TrajSampler(
        train_env,
        clip_action=FLAGS.clip_action,
        reward_scale=FLAGS.reward_scale,
        reward_bias=FLAGS.reward_bias,
        max_traj_length=FLAGS.config.get("max_episode_steps", 1000),
        action_horizon=pi_config.model.action_horizon,
    )

    ### Create EXPO agent #
    rng, construct_rng = jax.random.split(rng)
    
    # Make sure online buffer does not exist already otherwise this will overwrite it #
    assert not tf.io.gfile.exists(
        tf.io.gfile.join(save_dir, "image_replay_buffer", "episode_0.tfrecord")
    ), f"Image replay buffer already exists! ({tf.io.gfile.join(save_dir, 'image_replay_buffer', 'episode_0.tfrecord')})"
    
    # Build agent kwargs from config and flags
    # Remove keys that are already passed explicitly to avoid conflicts
    agent_kwargs = dict(FLAGS.config.agent_kwargs)
    for key in ['config', 'seed', 'batch_size', 'rng', 'params_path']:
        agent_kwargs.pop(key, None)
    
    agent = create_agent(
        agent_name=FLAGS.agent_name,
        config=pi_config,
        seed=FLAGS.seed,
        batch_size=FLAGS.config.batch_size,
        rng=construct_rng,
        params_path=FLAGS.params_path,
        **agent_kwargs,
    )

    timer = Timer()

    ### Evaluation Setup ###
    env_data_collection_policy_fn = None  # Will get set later
    rng, eval_policy_fn_key = jax.random.split(rng)
    eval_policy_fn = get_policy_fn(
        agent=agent,
        rng=eval_policy_fn_key,
        timer=timer,
        deterministic_actions=True,
    )
    #########################################################

    # TODO: Remove hardcode and init with flags appropriately #
    num_trajectories_to_collect = FLAGS.num_trajectories_to_collect
    online_env_steps = 0
    online_trajectories_added = 0

    ### Online training ###
    for i in range(FLAGS.num_train_steps):
        online_env_steps_this_epoch = 0
        # logging.info("Switching to online training...")

        ### Collect Trajectories ###
        if i % FLAGS.online_trajectory_collection_frequency == 0:
            print("Collecting trajectories...")
            data_collection_rng_key, rng = jax.random.split(rng)
            
            fns_dict = {}
            policy_rng_key, value_rng_key = jax.random.split(data_collection_rng_key)
            if i >= FLAGS.warmup_steps:
                fns_dict["policy_fn"] = get_policy_fn(
                    agent=agent,
                    rng=policy_rng_key,
                    timer=timer,
                    deterministic_actions=False,
                    num_diffusion_samples=FLAGS.num_diffusion_samples,
                )
            else:
               fns_dict["policy_fn"] = get_base_policy_fn(
                    agent=agent,
                    rng=policy_rng_key,
                    num_diffusion_samples=FLAGS.num_diffusion_samples,
                )
            fns_dict["value_fn"] = get_value_fn(
                agent=agent,
                rng=value_rng_key,
                timer=timer,
            )

            trajectories = []
            q_vs_mc_returns_vals = []
            for traj_index in range(num_trajectories_to_collect):
                timer.tick("trajectory_sampling_time")

                sampled_trajectories_successfully = False
                while not sampled_trajectories_successfully:
                    try:
                        with time_limit(STEP_TIME_LIMIT):
                            trajs, _q_vs_mc_returns_vals = data_collection_trajectory_sampler.sample(
                                fns_dict=fns_dict,
                                num_episodes=1,
                                replay_buffer=state_replay_buffer,
                                calc_mc_return_fn=functools.partial(calc_mc_return_fn, discount=FLAGS.config.agent_kwargs.discount, reward_bias=FLAGS.reward_bias),
                                store_max_trajectory_reward=True,
                                terminate_on_success=False,
                            )
                            sampled_trajectories_successfully = True
                            break
                    # except:
                    except (StepTimeout, TimeoutError, EOFError, BrokenPipeError) as e:
                        print(f"Trajectory sampling failed/timed out: {type(e).__name__}: {e}")
                        train_env._restart()
                        data_collection_trajectory_sampler = TrajSampler(
                            train_env,
                            clip_action=FLAGS.clip_action,
                            reward_scale=FLAGS.reward_scale,
                            reward_bias=FLAGS.reward_bias,
                            max_traj_length=FLAGS.config.get("max_episode_steps", 1000),
                            action_horizon=pi_config.model.action_horizon,
                        )


                traj = trajs[0]
                _q_vs_mc_returns_vals = _q_vs_mc_returns_vals[0]
                timer.tock("trajectory_sampling_time")
                print(timer.get_total_times(reset=False))
                # LOG: `traj` statistics #
                trajectories.append(traj)
                q_vs_mc_returns_vals.append(_q_vs_mc_returns_vals)

                if FLAGS.config.image_observations:
                    # Save trajectory as tfrecord
                    save_trajectory_as_tfrecord(
                        trajectory=traj,
                        path=tf.io.gfile.join(
                            save_dir,
                            "image_replay_buffer",
                            f"episode_{online_trajectories_added}.tfrecord",
                        ),
                    )
                online_trajectories_added += 1
                online_env_steps_this_epoch += len(traj["rewards"])
                
            
            # Get trajectory statistics
            # LOG: Log some statistics for the collected trajectories #
            mean_trajectory_return = np.mean(
                [np.sum(t["rewards"]) for t in trajectories]
            )
            mean_trajectory_length = np.mean([len(t["rewards"]) for t in trajectories])
            mean_max_reward = np.mean([np.max(t["rewards"]) for t in trajectories])
            if wandb_logger is not None:
                wandb_logger.log(
                    {
                        "train_env": {
                            "mean_trajectory_return": mean_trajectory_return,
                            "mean_trajectory_length": mean_trajectory_length,
                            "mean_max_reward": mean_max_reward,
                        },
                        "online_env_steps": online_env_steps,
                        "online_trajectories_added": online_trajectories_added,
                    },
                    step=i,
                )
            
            del trajectories, q_vs_mc_returns_vals
            import gc; gc.collect()
            

            # Finished collecting trajectories
            # LOG: Construct buffers using the trajectories #
            # LOG: Looks like two iterators are constructed, one for online trajectories and `dataset` from previous definition, for offline dataset #
            online_env_steps += online_env_steps_this_epoch
            # Recreate the image replay buffer iterator to include the new trajectories
            timer.tick("recreate_image_replay_buffer_iterator")
            data_paths = glob_to_path_list(
                tf.io.gfile.join(save_dir, "image_replay_buffer", "*.tfrecord")
            )

            if FLAGS.on_policy:
                # Keep only latest collected trajectories and discard all the other ones
                def extract_episode_id(path):
                    """Extract episode ID from path like 'episode_<id>.tfrecord'"""
                    basename = os.path.basename(path)
                    match = re.match(r'episode_(\d+)\.tfrecord', basename)
                    if match:
                        return int(match.group(1))
                    return -1  # For files that don't match the pattern

                # Sort by episode ID in descending order (highest ID first)
                data_paths = sorted(data_paths, key=extract_episode_id, reverse=True)

                # Keep only the last num_trajectories_to_collect episodes (highest IDs)
                if len(data_paths) > FLAGS.num_trajectories_to_collect:
                    paths_to_keep = data_paths[:FLAGS.num_trajectories_to_collect]
                    paths_to_delete = data_paths[FLAGS.num_trajectories_to_collect:]
                    
                    # Delete old tfrecord files from disk
                    for path in paths_to_delete:
                        try:
                            tf.io.gfile.remove(path)
                        except Exception as e:
                            logging.warning(f"Failed to delete {path}: {e}")
                    
                    logging.info(f"On-policy mode: Deleted {len(paths_to_delete)} old trajectories, keeping {len(paths_to_keep)} (episode IDs: {[extract_episode_id(p) for p in paths_to_keep]})")
                    data_paths = paths_to_keep
            
            #########################
            
            # Do some cleanups to avoid hangs #
            online_train_iterator = None
            image_replay_buffer = None
            import gc; gc.collect()

            #########################################################
            buffer_rng, rng = jax.random.split(rng)
            # buffer_seed = int(jax.random.randint(buffer_rng, (), 0, 2**31))
            image_replay_buffer = ImageReplayBufferPi(
                data_paths=data_paths,
                seed=FLAGS.seed,
                train=True,
                task_name=FLAGS.task_name,
                use_wrist_view=FLAGS.use_wrist_view, 
                use_language=FLAGS.use_lang, config=pi_config,
                final_step_sparse_reward=False, # Use rewards from environment and DO NOT override with sparse 0/1 rewards at final step #
                filter_successful_trajectories=False, # Whether to use success buffer #
                use_reverse_data_paths=False,
                alpha=FLAGS.scale_success_alpha,
                scale_success_reward=FLAGS.scale_success_alpha > 0,
                intermediate_reward_mul_factor=FLAGS.intermediate_reward_mul_factor,
                drop_images_from_output=True,
                offline_flag=0.0,
                **FLAGS.config.image_replay_buffer_kwargs,
            )
            timer.tock("recreate_image_replay_buffer_iterator")
            #########################################################

            # Update online iterator #
            online_train_iterator = image_replay_buffer.iterator(
                batch_size=FLAGS.config.batch_size
            )

        rng, rng_update = jax.random.split(rng)
        
        is_warmup_flag = i < FLAGS.warmup_steps
        # Combine warmup flag with individual ws_critic and ws_edit_actor flags
        critic_warmup = is_warmup_flag and FLAGS.ws_critic
        edit_actor_warmup = is_warmup_flag and FLAGS.ws_edit_actor
        
        timer.tick("online_iter_total")
        print("Warmup")
        timer.tick("sample_batch_time")
        offline_batch = next(offline_train_iterator)
        online_batch = next(online_train_iterator)

        update_critic = True
        update_edit_actor = True

        if is_warmup_flag:
            batch = offline_batch

            if not FLAGS.ws_critic:
                update_critic = False
                assert FLAGS.ws_edit_actor, "Edit actor must be warmed up if critic is not warmed up"
            if not FLAGS.ws_edit_actor:
                update_edit_actor = False
                assert FLAGS.ws_critic, "Critic must be warmed up if edit actor is not warmed up"
        else:
            batch = concatenate_batches([offline_batch, online_batch])
        timer.tock("sample_batch_time")
        agent, info = agent.update(batch,
            utd_ratio=FLAGS.config.utd_ratio,
            timer=timer,
            seed=rng_update,
            update_critic=update_critic,
            update_edit_actor=update_edit_actor,
            critic_warmup=critic_warmup,
            critic_warmup_type=FLAGS.critic_warmup_type,
            calql_lower_bound=FLAGS.calql_lower_bound,
            edit_actor_warmup=edit_actor_warmup,
            bc_loss_coef=FLAGS.bc_loss_coef,
            cql_alpha=FLAGS.cql_alpha,
            cql_temp=FLAGS.cql_temp,
            calql_random_actions=FLAGS.calql_random_actions,
            grpo_beta=FLAGS.grpo_beta,
            grpo_weight_threshold=FLAGS.grpo_weight_threshold,
            edit_actor_utd_ratio=FLAGS.edit_actor_utd_ratio,
        )
        
        # Log batch statistics #
        batch_stats = {
            "batch_stats/batch_size": len(batch),
            "batch_stats/masks_mean": np.mean(batch["masks"]),
            "batch_stats/mc_returns_mean": np.mean(batch["mc_returns"]),
            "batch_stats/mc_returns_min": np.min(batch["mc_returns"]),
            "batch_stats/mc_returns_max": np.max(batch["mc_returns"]),
            "batch_stats/rewards_mean": np.mean(batch["rewards"]),
            "batch_stats/rewards_min": np.min(batch["rewards"]),
            "batch_stats/rewards_max": np.max(batch["rewards"]),
            "batch_stats/terminals_mean": np.mean(batch["terminals"]),
            "batch_stats/truncations_mean": np.mean(batch["truncates"]),
            "batch_stats/success_mean": np.mean(batch["success"]),
            "batch_stats/success_min": np.min(batch["success"]),
            "batch_stats/success_max": np.max(batch["success"]),
            "batch_stats/is_offline_data_mean": np.mean(batch["is_offline_data"]),
            "batch_stats/is_offline_data_max": np.max(batch["is_offline_data"]),
            "batch_stats/is_offline_data_min": np.min(batch["is_offline_data"]),
        }
        if wandb_logger is not None:
            wandb_logger.log(batch_stats, step=i)
        

        # Log training metrics #
        if wandb_logger is not None:
            wandb_logger.log(info, step=i)
        
        
        timer.tock("online_iter_total")
        print(timer.get_total_times(reset=True))

        ### Evaluation ###
        if (
            (i + 1) % FLAGS.config.eval_interval == 0
        ) and eval_env is not None:
            """eval"""
            logging.info("Evaluating...")
            timer.tick("evaluation/total")

            if FLAGS.config.save_video:
                try:
                    eval_env.start_recording(
                        FLAGS.config.num_episodes_per_video,
                        FLAGS.config.num_episodes_per_row,
                    )
                except Exception as e:
                    pass
            if FLAGS.config.num_eval_episodes > 0:
                print("Evaluating...")
                q_vs_mc_returns_vals = None
                if "libero" not in FLAGS.environment_name: 
                    trajectories = evaluate_with_trajectories_vectorized(
                        eval_policy_fn,
                        eval_env,
                        FLAGS.config.num_eval_episodes,
                    )
                else:
                    if FLAGS.num_parallel_envs != 1:
                        trajectories = evaluate_with_trajectories_vectorized(
                        eval_policy_fn,
                        eval_env,
                        FLAGS.config.num_eval_episodes,
                    )
                    else:
                        eval_policy_fn = get_policy_fn(
                            agent=agent,
                            rng=eval_policy_fn_key,
                            timer=timer,
                            deterministic_actions=True,
                        )

                        evaluated_trajectories_successfully = False
                        while not evaluated_trajectories_successfully:
                            try:
                                with time_limit(STEP_TIME_LIMIT):
                                    trajectories, q_vs_mc_returns_vals = evaluate_with_trajectories_libero(
                                        eval_policy_fn,
                                        eval_env,
                                        FLAGS.config.num_eval_episodes,
                                    )
                                    evaluated_trajectories_successfully = True
                                    break
                            except (StepTimeout, TimeoutError, EOFError, BrokenPipeError) as e:
                                print(f"Evaluation failed/timed out: {type(e).__name__}: {e}")
                                eval_env._restart()

                if (FLAGS.environment_name == "calvin" or FLAGS.environment_name =='libero') and FLAGS.config.save_video:
                    trajectories_to_save = trajectories[
                        : FLAGS.config.num_episodes_per_video
                    ]
                    frames = []
                    ind_traj = []
                    for j, traj in enumerate(trajectories_to_save):
                        trajectory_return = 0
                        for transition, reward in zip(
                            traj["observation"], traj["reward"]
                        ):
                            assert transition["image"].shape[-1] == 3
                            if len(transition["image"].shape) == 4:
                                transition["image"] = transition["image"][0]
                            image = transition["image"]  # .transpose(2, 0, 1)
                            # Add text for reward and return so far
                            trajectory_return += reward
                            # image = np.flipud(image)
                            image = np.ascontiguousarray(image) 
                            frame = cv2.putText(
                                image,
                                f"reward: {reward}. return: {trajectory_return}",
                                (10, 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.3,
                                (0, 0, 0),
                                1,
                            )
                            ind_traj.append(frame)
                            frame = frame.transpose(2, 0, 1)
                            frames.append(frame)
                        
                        save_rollout_gif(ind_traj, save_dir, step_i=i, rollout_j=j)
                        ind_traj = []
                    
                    del ind_traj, frames
                    import gc; gc.collect()

                eval_metrics = {
                    "eval/average_return": np.mean(
                        [np.sum(t["reward"]) for t in trajectories]
                    ),
                    "eval/average_episode_length": np.mean(
                        [len(t["reward"]) for t in trajectories]
                    ),
                    **(
                        {
                            "eval/average_normalized_return": np.mean(
                                [
                                    eval_env.get_normalized_score(np.sum(t["reward"]))
                                    for t in trajectories
                                ]
                            ),
                            "eval/min_normalized_return": np.min(
                                [
                                    eval_env.get_normalized_score(np.sum(t["reward"]))
                                    for t in trajectories
                                ]
                            ),
                            "eval/max_normalized_return": np.max(
                                [
                                    eval_env.get_normalized_score(np.sum(t["reward"]))
                                    for t in trajectories
                                ]
                            ),
                        }
                        if hasattr(eval_env, "get_normalized_score")
                        else {}
                    ),
                    "eval/average_max_reward": np.mean(
                        [np.max(t["reward"]) for t in trajectories]
                    ),
                }

                if wandb_logger is not None:
                    wandb_logger.log(eval_metrics, step=i)
                
                del trajectories
                import gc; gc.collect()
            timer.tock("evaluation/total")

        if i % FLAGS.config.save_interval == 0:
            if FLAGS.config.save_dir:
                os.makedirs(FLAGS.config.save_dir, exist_ok=True)
                final_checkpoint_path = os.path.join(FLAGS.config.save_dir, f"checkpoint_{i}.pkl")
                logging.info(f"Saving final checkpoint to {final_checkpoint_path}")
                agent.save_checkpoint(final_checkpoint_path, step=i)
                logging.info(f"Saved checkpoint to {final_checkpoint_path}")

if __name__ == "__main__":
    app.run(train_agent)