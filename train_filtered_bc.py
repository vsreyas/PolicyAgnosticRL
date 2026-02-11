"""Filtered behavior cloning training script.

Loads an offline dataset (with optional filtering for successful trajectories
and class-balanced sampling) and trains the base pi0 actor via supervised BC.
No warmstart, no online data collection, no critic/edit-actor updates.

Based on train_pi_residual.py template.
"""

# Try increasing the number of open files limit
import resource
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(65535, hard), hard))

import os
import gc

import jax
import numpy as np
import tensorflow as tf

import wandb
from absl import app, flags, logging
from ml_collections import config_flags

from jaxrl_m.common.wandb import WandBLogger
from jaxrl_m.data.bridge_dataset import glob_to_path_list
from jaxrl_m.data.img_replay_buffer_pi_bc import ImageReplayBufferPi
from jaxrl_m.utils.timer_utils import Timer
from jaxrl_m.agents.continuous.pi_vlm_cached import create_agent

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
flags.DEFINE_string("agent_name", "pi_residual_td3_grpo", "Agent name.")
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
flags.DEFINE_string(
    "online_critic_update_type",
    None,
    "Critic update type during online (non-warmup) training: 'sarsa', 'calql', or None for default TD3 critic."
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
flags.DEFINE_bool(
    "do_ascent",
    False,
    "Use gradient ascent on base actions via sample_base_actions_gradq.",
)
flags.DEFINE_float(
    "eta_ascent",
    0.01,
    "Step size for gradient ascent.",
)
flags.DEFINE_integer(
    "num_ascent_steps",
    50,
    "Number of gradient ascent steps.",
)
flags.DEFINE_bool(
    "critic_balance_classes",
    False,
    "Balance successful and unsuccessful trajectories in critic loss using success ratio weighting.",
)


def balance_offline_training_data(offline_dset_paths):
    """Inspect tf records, check if trajectory is successful or not and subsample
    the more frequent class so both classes have equal representation."""
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
        success_trajectories = list(np.random.choice(
            success_trajectories, size=len(failed_trajectories), replace=False))
    elif len(success_trajectories) < len(failed_trajectories):
        failed_trajectories = list(np.random.choice(
            failed_trajectories, size=len(success_trajectories), replace=False))

    # Further limit to num_balanced_trajectories if set
    if FLAGS.num_balanced_trajectories > 0:
        if len(success_trajectories) > FLAGS.num_balanced_trajectories:
            success_trajectories = list(np.random.choice(
                success_trajectories, size=FLAGS.num_balanced_trajectories, replace=False))
        if len(failed_trajectories) > FLAGS.num_balanced_trajectories:
            failed_trajectories = list(np.random.choice(
                failed_trajectories, size=FLAGS.num_balanced_trajectories, replace=False))

    print(f"Balanced offline training data: Success: {len(success_trajectories)}, "
          f"Failed: {len(failed_trajectories)}")
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

    # Get PI config
    pi_config = get_config(FLAGS.pi_config_name)
    pi_config.fsdp_devices = 1
    pi_config.exp_name = FLAGS.wandb_experiment_name
    pi_config.overwrite = True

    # WandB setup
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
            f"seed_{FLAGS.seed}",
        )
    else:
        wandb_logger = None
        save_dir = tf.io.gfile.join(
            os.path.abspath(FLAGS.config.save_dir),
        )
        FLAGS.config.wandb_enabled = False

    ####### Dataset setup (using img_replay_buffer_pi_bc.py) ###########
    assert len(FLAGS.config.libero_tfrecord_regexp) > 0, \
        "libero_tfrecord_regexp must be provided in config"

    paths = glob_to_path_list(FLAGS.config.libero_tfrecord_regexp)
    if FLAGS.balance_offline_training_data:
        paths = balance_offline_training_data(paths)

    dataset = ImageReplayBufferPi(
        data_paths=paths,
        seed=FLAGS.seed,
        train=True,
        task_name=FLAGS.task_name,
        use_wrist_view=FLAGS.use_wrist_view,
        use_language=FLAGS.use_lang,
        config=pi_config,
        filter_successful_trajectories=FLAGS.filter_successful_trajectories,
    )
    train_iterator = dataset.iterator(batch_size=FLAGS.config.batch_size)

    ####### Agent setup ###########
    rng = jax.random.PRNGKey(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    rng, construct_rng = jax.random.split(rng)

    # Build agent kwargs from config and flags
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

    ####### Training loop ###########
    for i in range(FLAGS.num_train_steps):
        timer.tick("total_step_time")

        timer.tick("sample_batch_time")
        batch = next(train_iterator)
        timer.tock("sample_batch_time")

        timer.tick("update_actor_time")
        agent, info = agent.update_actor(batch, timer=timer)
        timer.tock("update_actor_time")

        timer.tock("total_step_time")

        # Log training metrics
        if wandb_logger is not None:
            wandb_logger.log(info, step=i)

        if i % 100 == 0:
            print(f"Step {i}: {info}")
            print(timer.get_total_times(reset=True))

        # Save checkpoint
        if i % FLAGS.config.save_interval == 0 and FLAGS.config.save_dir:
            logging.info(f"Saving pi0 checkpoint at step {i}")
            agent.actor.save_checkpoint(FLAGS.config.save_dir, step=i)
            logging.info(f"Saved pi0 checkpoint at step {i}")


if __name__ == "__main__":
    app.run(train_agent)
