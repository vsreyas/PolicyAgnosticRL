# Default configs.
import os
from copy import deepcopy

import tensorflow as tf
from jaxrl_m.agents.continuous.cql import (
    get_default_config as get_continuous_cql_config,
)
from jaxrl_m.agents.continuous.diffusion_q_learning import (
    get_default_config as get_diffusion_q_learning_config,
)

SAVE_DIR_PREFIX = os.environ.get("SAVE_DIR_PREFIX", "libero_10_pi05_put_the_two_mocha_pots_on_the_stove")
DEFAULT_PARL_CONFIG = dict(
    num_base_policy_actions=16,
    num_actions_to_keep=10,
    num_steps=10,
    step_size=3e-4,
    optimize_critic_ensemble_min=False,
    use_target_critic=False,
)

# Used to pre-train a Diffusion Policy base policy.
BASE_DDPM_CONFIG = dict(
    agent="ddpm_bc",
    batch_size=2,
    save_dir=tf.io.gfile.join(SAVE_DIR_PREFIX, "results"),
    eval_interval=2,
    save_interval=100,
    log_interval=10,
    deterministic_eval=True,
    num_eval_episodes=10,
    num_episodes_per_video=5,
    num_episodes_per_row=5,
    save_video=False,
    image_observations=False,
    goal_conditioned=False,
    agent_kwargs=dict(
        batch_size=256,
        score_network_kwargs=dict(
            time_dim=128,
            num_blocks=3,
            dropout_rate=0.1,
            hidden_dim=256,
            use_layer_norm=True,
        ),
        use_proprio=False,
        beta_schedule="cosine",
        diffusion_steps=5,
        action_samples=64,
        repeat_last_step=0,
        learning_rate=3e-4,
        warmup_steps=1000,
        actor_decay_steps=int(3e6),
        image_observations=False,
        discount=0.99,
        drq_padding=0,
    ),
)


ddpm_base_policy_agent_kwargs_for_parl = BASE_DDPM_CONFIG["agent_kwargs"].copy()
ddpm_base_policy_agent_kwargs_for_parl.update(
    learning_rate=5e-5,
    actor_decay_steps=None,
)
BASE_PARL_CALQL_CONFIG = dict(
    agent="parl_calql",
    batch_size=256,
    save_dir=tf.io.gfile.join(SAVE_DIR_PREFIX, "results"),
    eval_interval=50,
    save_interval=500,
    log_interval=10,
    deterministic_eval=True,
    num_eval_episodes=50,
    num_episodes_per_video=5,
    num_episodes_per_row=5,
    save_video=False,
    parl_config=DEFAULT_PARL_CONFIG,
    data_collection_particle_choosing_strategy="max_q_value",
    evaluation_particle_choosing_strategy="max_q_value",
    image_observations=False,
    goal_conditioned=False,
    improve_base_policy_actions_with_global_search=True,
    base_policy_path="",
    mixing_ratio=0.5,
    distill_argmax=False,
    agent_kwargs=get_continuous_cql_config(
        updates=dict(
            discount=0.99,
            batch_size=256,
            distributional_critic=True,
            distributional_critic_kwargs=dict(
                q_min=-100.0,
                q_max=0.0,
                num_bins=128,
            ),
            critic_network_type="mlp",
            critic_kwargs=dict(
                kernel_init_type="orthogonal",
                kernel_init_params=dict(
                    scale=1e-2,
                ),
            ),
            critic_network_kwargs=dict(
                hidden_dims=(256, 256),
                activate_final=True,
                kernel_scale_final=1e-2,
                use_feature_normalization=False,
                use_layer_norm=True,
            ),
            critic_optimizer_kwargs={
                "learning_rate": 3e-4,
                "warmup_steps": 0,
                "weight_decay": 0.0,
            },
            cql_importance_sample=False,
            cql_n_actions=10,
            use_calql=True,
            use_calql_on_random_actions=False,
            autotune_entropy=False,
            cql_autotune_alpha=False,
            critic_ensemble_size=10,
            critic_subsample_size=2,
            policy_optimizes_ensemble_mean=False,
            drq_padding=0,
            cql_alpha=0.005,
            only_use_next_actions_for_cql=False,
        ),
    ),
    base_policy_agent_kwargs=ddpm_base_policy_agent_kwargs_for_parl,
)

BASE_PARL_IQL_CONFIG = deepcopy(BASE_PARL_CALQL_CONFIG)
BASE_PARL_IQL_CONFIG = BASE_PARL_IQL_CONFIG.update(
    agent="parl_iql",
    agent_kwargs=dict(
        discount=0.99,
        expectile=0.7,
        batch_size=256,
        distributional_critic=True,
        distributional_critic_kwargs=dict(
            q_min=-100.0,
            q_max=0.0,
            num_bins=128,
        ),
        critic_network_type="mlp",
        critic_kwargs=dict(
            kernel_init_type="orthogonal",
            kernel_init_params=dict(
                scale=1e-2,
            ),
        ),
        value_fns_network_kwargs=dict(
            hidden_dims=(256, 256),
            activate_final=True,
            kernel_scale_final=1e-2,
            use_feature_normalization=False,
            use_layer_norm=True,
        ),
        value_critic_optimizer_kwargs={
            "learning_rate": 3e-4,
            "warmup_steps": 0,
            "weight_decay": 0.0,
        },
        critic_ensemble_size=10,
        critic_subsample_size=2,
        drq_padding=0,
    ),
)

BASE_AUTO_REGRESSIVE_TRANSFORMER_CONFIG = dict(
    agent="auto_regressive_transformer",
    batch_size=256,
    save_dir=tf.io.gfile.join(SAVE_DIR_PREFIX, "results"),
    eval_interval=500,
    save_interval=1000,
    log_interval=10,
    deterministic_eval=True,
    num_eval_episodes=50,
    num_episodes_per_video=5,
    num_episodes_per_row=5,
    save_video=False,
    image_observations=False,
    goal_conditioned=False,
    agent_kwargs=dict(
        learning_rate=1e-4,
        warmup_steps=2000,
        image_observations=False,
        discount=0.99,
    ),
)

BASE_TRANSFORMER_PARL_CONFIG = deepcopy(BASE_PARL_CALQL_CONFIG)
BASE_TRANSFORMER_PARL_CONFIG.update(
    base_policy_agent_kwargs=BASE_AUTO_REGRESSIVE_TRANSFORMER_CONFIG["agent_kwargs"],
)


BASE_DIFFUSION_Q_LEARNING_CONFIG = dict(
    agent="diffusion_q_learning",
    batch_size=256,
    save_dir=tf.io.gfile.join(SAVE_DIR_PREFIX, "results"),
    eval_interval=50,
    save_interval=500,
    log_interval=10,
    deterministic_eval=True,
    num_eval_episodes=50,
    num_episodes_per_video=5,
    num_episodes_per_row=5,
    save_video=False,
    image_observations=False,
    goal_conditioned=False,
    mixing_ratio=0.5,
    agent_kwargs=get_diffusion_q_learning_config(
        updates=dict(
            discount=0.99,
            batch_size=256,
            distributional_critic=True,
            distributional_critic_kwargs=dict(
                q_min=-100.0,
                q_max=0.0,
                num_bins=128,
            ),
            critic_network_type="mlp",
            critic_kwargs=dict(
                kernel_init_type="orthogonal",
                kernel_init_params=dict(
                    scale=1e-2,
                ),
            ),
            critic_network_kwargs=dict(
                hidden_dims=(256, 256),
                activate_final=True,
                kernel_scale_final=1e-2,
                use_feature_normalization=False,
                use_layer_norm=True,
            ),
            actor_optimizer_kwargs=dict(
                learning_rate=3e-4,
                warmup_steps=0,
                clip_grad_norm=10.0,
                cosine_decay_steps=-1.0,
            ),
            critic_optimizer_kwargs={
                "learning_rate": 3e-4,
                "warmup_steps": 0,
                "weight_decay": 0.0,
                "clip_grad_norm": 10.0,
                "cosine_decay_steps": -1.0,
            },
            temperature_optimizer_kwargs={
                "learning_rate": 1e-4,
            },
            autotune_entropy=False,
            critic_ensemble_size=10,
            critic_subsample_size=2,
            policy_optimizes_ensemble_mean=True,
            rl_weight=1.0,
            cql_max_target_backup=False,
            cql_n_actions=10,
            drq_padding=0,
        )
    ),
    base_policy_agent_kwargs=ddpm_base_policy_agent_kwargs_for_parl,
)

BASE_GAUSSIAN_CALQL_CONFIG = dict(
    agent="calql",
    batch_size=256,
    save_dir=tf.io.gfile.join(SAVE_DIR_PREFIX, "results"),
    eval_interval=100,
    save_interval=500,
    log_interval=10,
    deterministic_eval=True,
    num_eval_episodes=50,
    num_episodes_per_video=5,
    num_episodes_per_row=5,
    save_video=False,
    agent_kwargs=get_continuous_cql_config(
        updates=dict(
            discount=0.99,
            batch_size=256,
            distributional_critic=True,
            distributional_critic_kwargs=dict(
                q_min=-100.0,
                q_max=0.0,
                num_bins=128,
            ),
            critic_kwargs=dict(
                kernel_init_type="orthogonal",
                kernel_init_params=dict(
                    scale=1e-2,
                ),
            ),
            policy_kwargs=dict(
                tanh_squash_distribution=True,
                std_parameterization="exp",
                kernel_init_type="orthogonal",
                kernel_init_params=dict(
                    scale=1e-2,
                ),
            ),
            critic_network_kwargs=dict(
                hidden_dims=(256, 256),
                activate_final=True,
                kernel_init_type="orthogonal",
                activations="relu",
                kernel_scale_final=1e-2,
            ),
            policy_network_kwargs=dict(
                hidden_dims=(256, 256),
                activate_final=True,
                kernel_init_type="orthogonal",
                activations="relu",
                kernel_scale_final=1e-2,
            ),
            actor_optimizer_kwargs={
                "learning_rate": 1e-4,
                "warmup_steps": 0,
            },
            critic_optimizer_kwargs={
                "learning_rate": 3e-4,
                "warmup_steps": 0,
            },
            cql_n_actions=10,
        ),
    ),
)

BASE_PI_CONFIG = dict(
    agent="pi-0",
    batch_size=16,
    save_dir=tf.io.gfile.join(SAVE_DIR_PREFIX, "results"),
    eval_interval=10,
    save_interval=5,
    log_interval=1,
    deterministic_eval=True,
    num_eval_episodes=4,
    num_episodes_per_video=2,
    num_episodes_per_row=1,
    save_video=True,
    image_observations=False,
    goal_conditioned=False,
    agent_kwargs=dict(
        batch_size=16,
        score_network_kwargs=dict(
            time_dim=128,
            num_blocks=3,
            dropout_rate=0.1,
            hidden_dim=256,
            use_layer_norm=True,
        ),
        use_proprio=False,
        beta_schedule="cosine",
        diffusion_steps=5,
        action_samples=64,
        repeat_last_step=0,
        learning_rate=3e-4,
        warmup_steps=1000,
        actor_decay_steps=int(3e6),
        image_observations=False,
        discount=0.99,
        drq_padding=0,
    ),
)

BASE_EXPO_CONFIG = dict(
    batch_size=16,
    save_video=True,
    image_observations=True,
    libero_tfrecord_regexp="/data/hf_cache/datasets/LIBERO/libero_10_tf/*.tfrecord",
    dataset_kwargs=dict(
        cache=False,
        tfrecords_include_next_observations=False,
    ),
    num_epochs=100,
    num_train_steps_per_epoch=1000,
    num_eval_episodes=4,
    num_episodes_per_video=2,
    eval_interval=10,
    log_interval=10,
    save_interval=50,
    discount=0.99,
    actor_lr=3e-4,
    critic_lr=3e-4,
    temp_lr=3e-4,
    tau=0.005,
    utd_ratio=4,
    q_clip_low=-50.0,
    q_clip_high=5.0,
    max_episode_steps=1000,
    save_dir=os.path.join(SAVE_DIR_PREFIX, "results_expo"),
    # PaliGemma config name from openpi
    paligemma_config_name="pi05_libero_custom_low_mem",
    agent_kwargs=dict(
        batch_size=16,
        score_network_kwargs=dict(
            time_dim=128,
            num_blocks=3,
            dropout_rate=0.1,
            hidden_dim=256,
            use_layer_norm=True,
        ),
        use_proprio=False,
        beta_schedule="cosine",
        diffusion_steps=5,
        action_samples=64,
        repeat_last_step=0,
        learning_rate=3e-4,
        warmup_steps=1000,
        actor_decay_steps=int(3e6),
        image_observations=False,
        discount=0.99,
        drq_padding=0,
    ),
)

BASE_RESIDUAL_TD3_CONFIG = dict(
    batch_size=16,
    save_video=True,
    image_observations=True,
    libero_tfrecord_regexp="/data/hf_cache/datasets/LIBERO/libero_10_tf/*.tfrecord",
    dataset_kwargs=dict(
        cache=False,
        tfrecords_include_next_observations=False,
    ),
    num_epochs=100,
    num_train_steps_per_epoch=1000,
    num_eval_episodes=4,
    num_episodes_per_video=2,
    eval_interval=10,
    log_interval=10,
    save_interval=1000,
    discount=0.99,
    actor_lr=3e-4,
    critic_lr=3e-4,
    temp_lr=3e-4,
    tau=0.005,
    utd_ratio=4,
    q_clip_low=-50.0,
    q_clip_high=5.0,
    max_episode_steps=1000,
    save_dir=os.path.join(SAVE_DIR_PREFIX, "results_pi_residual_td3"),
    # PaliGemma config name from openpi
    paligemma_config_name="pi05_libero_custom_low_mem",
    agent_kwargs=dict(
        # Agent architecture params
        action_dim=7,
        state_dim=8,
        hidden_dims=(512, 512, 512, 512),
        
        # Learning rates
        actor_lr=3e-4,
        critic_lr=3e-4,
        temp_lr=3e-4,
        
        # Training params
        discount=0.99,
        tau=0.005,
        decay_steps=int(3e6),
        
        # Critic params
        num_qs=10,
        num_min_qs=2,
        critic_dropout_rate=None,
        critic_weight_decay=None,
        critic_layer_norm=True,
        use_critic_resnet=False,
        
        # Actor params
        actor_tau=0.003,
        actor_dropout_rate=None,
        actor_num_blocks=3,
        actor_layer_norm=True,
        
        # Action sampling params
        N=4,
        n_edit_samples=4,
        edit_action_scale=3.0,
        exploration_epsilon=0.05,
        
        # Entropy params
        target_entropy=None,
        entropy_scale=1.0,
        init_temperature=1.0,
        entropy_mul_scale_factor=3.0,
        adjust_target_entropy=True,
        
        # Other params
        backup_entropy=True,
        use_pnorm=False,
        clip_sampler=True,
        time_dim=128,
        T=10,
        M=0,
        batch_split=1,
        ddpm_temperature=1.0,
        beta_schedule='vp',
        batch_size_dict_key='actions',
    ),
)

BASE_RESIDUAL_PPO_CONFIG = dict(
    batch_size=16,
    save_video=True,
    image_observations=True,
    libero_tfrecord_regexp="/data/hf_cache/datasets/LIBERO/libero_10_tf/*.tfrecord",
    dataset_kwargs=dict(
        cache=False,
        tfrecords_include_next_observations=False,
    ),
    num_epochs=100,
    num_train_steps_per_epoch=1000,
    num_eval_episodes=4,
    num_episodes_per_video=2,
    eval_interval=10,
    log_interval=10,
    save_interval=50,
    discount=0.99,
    actor_lr=3e-4,
    critic_lr=3e-4,
    temp_lr=3e-4,
    tau=0.005,
    utd_ratio=4,
    q_clip_low=-50.0,
    q_clip_high=5.0,
    max_episode_steps=1000,
    save_dir=os.path.join(SAVE_DIR_PREFIX, "results_pi_residual_ppo"),
    # PaliGemma config name from openpi
    paligemma_config_name="pi05_libero_custom_low_mem",
    agent_kwargs=dict(
        # Agent architecture params
        action_dim=7,
        state_dim=8,
        hidden_dims=(512, 512, 512, 512),
        
        # Learning rates
        actor_lr=3e-4,
        critic_lr=3e-4,
        temp_lr=3e-4,
        
        # Training params
        discount=0.99,
        tau=0.005,
        decay_steps=int(3e6),
        
        # Critic params (PPO uses value function, num_qs=1 by default)
        num_qs=1,
        num_min_qs=None,
        critic_dropout_rate=None,
        critic_weight_decay=None,
        critic_layer_norm=True,
        use_critic_resnet=False,
        critic_success_wt=1.0,
        
        # Actor params
        actor_tau=0.003,
        actor_dropout_rate=None,
        actor_num_blocks=3,
        actor_layer_norm=True,
        
        # Action sampling params
        N=4,
        n_edit_samples=4,
        edit_action_scale=3.0,
        exploration_epsilon=0.05,
        
        # PPO-specific params
        epsilon_low=0.05,
        epsilon_high=0.05,
        ent_coef=0.00001,
        
        # Entropy params
        target_entropy=None,
        entropy_scale=1.0,
        init_temperature=1.0,
        entropy_mul_scale_factor=3.0,
        adjust_target_entropy=True,
        
        # Other params
        backup_entropy=True,
        use_pnorm=False,
        clip_sampler=True,
        time_dim=128,
        T=10,
        M=0,
        batch_split=1,
        ddpm_temperature=1.0,
        beta_schedule='vp',
        batch_size_dict_key='actions',
    ),
)

BASE_RESIDUAL_TD3_GRPO_CONFIG = dict(
    batch_size=16,
    save_video=True,
    image_observations=True,
    libero_tfrecord_regexp="/data/hf_cache/datasets/LIBERO/libero_10_tf/*.tfrecord",
    dataset_kwargs=dict(
        cache=False,
        tfrecords_include_next_observations=False,
    ),
    num_epochs=100,
    num_train_steps_per_epoch=1000,
    num_eval_episodes=4,
    num_episodes_per_video=2,
    eval_interval=10,
    log_interval=10,
    save_interval=1000,
    discount=0.99,
    actor_lr=3e-4,
    critic_lr=3e-4,
    temp_lr=3e-4,
    tau=0.005,
    utd_ratio=4,
    q_clip_low=-50.0,
    q_clip_high=5.0,
    max_episode_steps=1000,
    save_dir=os.path.join(SAVE_DIR_PREFIX, "results_pi_residual_td3_grpo"),
    # PaliGemma config name from openpi
    paligemma_config_name="pi05_libero_custom_low_mem",
    agent_kwargs=dict(
        # Agent architecture params
        action_dim=7,
        state_dim=8,
        hidden_dims=(512, 512, 512, 512),
        
        # Learning rates
        actor_lr=3e-4,
        critic_lr=3e-4,
        temp_lr=3e-4,
        
        # Training params
        discount=0.99,
        tau=0.005,
        decay_steps=int(3e6),
        
        # Critic params
        num_qs=10,
        num_min_qs=2,
        critic_dropout_rate=0.0,
        critic_weight_decay=0.0,
        critic_layer_norm=True,
        use_critic_resnet=False,
        
        # Actor params
        actor_tau=0.003,
        actor_dropout_rate=0.0,
        actor_num_blocks=3,
        actor_layer_norm=True,
        
        # Action sampling params
        N=4,
        n_edit_samples=4,
        edit_action_scale=3.0,
        exploration_epsilon=0.05,
        
        # GRPO-specific params
        grpo_beta=1.0,
        
        # Entropy params
        target_entropy=-7.0,
        entropy_scale=1.0,
        init_temperature=1.0,
        entropy_mul_scale_factor=3.0,
        adjust_target_entropy=True,
        
        # Other params
        backup_entropy=True,
        use_pnorm=False,
        clip_sampler=True,
        time_dim=128,
        T=10,
        M=0,
        batch_split=1,
        ddpm_temperature=1.0,
        beta_schedule='vp',
        batch_size_dict_key='actions',
    ),
)

pi0_base_policy_agent_kwargs_for_parl = BASE_PI_CONFIG["agent_kwargs"].copy()
pi0_base_policy_agent_kwargs_for_parl.update(
    learning_rate=5e-5,
    actor_decay_steps=None,
)
BASE_PARL_CALQL_CONFIG_Pi0 = dict(
    agent="parl_calql",
    batch_size=16,
    save_dir=tf.io.gfile.join(SAVE_DIR_PREFIX, "results"),
    eval_interval=10,
    save_interval=5,
    log_interval=1,
    deterministic_eval=True,
    num_eval_episodes=4,
    num_episodes_per_video=2,
    num_episodes_per_row=1,
    save_video=True,
    parl_config=DEFAULT_PARL_CONFIG,
    data_collection_particle_choosing_strategy="max_q_value",
    evaluation_particle_choosing_strategy="max_q_value",
    image_observations=False,
    goal_conditioned=False,
    improve_base_policy_actions_with_global_search=True,
    base_policy_path="",
    mixing_ratio=0.5,
    distill_argmax=False,
    agent_kwargs=get_continuous_cql_config(
        updates=dict(
            discount=0.99,
            batch_size=16,
            distributional_critic=True,
            distributional_critic_kwargs=dict(
                q_min=-100.0,
                q_max=0.0,
                num_bins=128,
            ),
            critic_network_type="mlp",
            critic_kwargs=dict(
                kernel_init_type="orthogonal",
                kernel_init_params=dict(
                    scale=1e-2,
                ),
            ),
            critic_network_kwargs=dict(
                hidden_dims=(256, 256),
                activate_final=True,
                kernel_scale_final=1e-2,
                use_feature_normalization=False,
                use_layer_norm=True,
            ),
            critic_optimizer_kwargs={
                "learning_rate": 3e-4,
                "warmup_steps": 0,
                "weight_decay": 0.0,
            },
            cql_importance_sample=False,
            cql_n_actions=10,
            use_calql=True,
            use_calql_on_random_actions=False,
            autotune_entropy=False,
            cql_autotune_alpha=False,
            critic_ensemble_size=10,
            critic_subsample_size=2,
            policy_optimizes_ensemble_mean=False,
            drq_padding=0,
            cql_alpha=0.005,
            only_use_next_actions_for_cql=False,
            use_wrist_view=True,
            use_proprio=True
        ),
    ),
    base_policy_agent_kwargs=pi0_base_policy_agent_kwargs_for_parl,
)
