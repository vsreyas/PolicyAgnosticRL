


export SAVE_DIR_PREFIX="/data/user_data/sreyasv/q_chunking_pi_no_-1_10_critic_ws_debug/libero_10_pi05_put_the_two_mocha_pots_on_the_stove/debug_plots/results/debug-runs/pi0_5_finetuning_parl_qchunking_-1_10_critic_ws_debug/seed_0/debug_plots" 


XLA_PYTHON_CLIENT_PREALLOCATE=false env -u PYOPENGL_PLATFORM python ./q_vs_mc_returns.py --environment_name=libero \
                  -wandb_experiment_name=pi0_5_finetuning_parl_qchunking_SARSA_Qwarmstart_-1_10 \
                  --config=./configs/libero_config.py:parl_calql_pi0 \
                  --seed=0\
                  --config.agent_kwargs.critic_network_kwargs.hidden_dims=256,256,256,256 \
                  --config.base_policy_path=pi0 --config.agent_kwargs.cql_alpha=0.000 \
                  --num_offline_epochs=500 --num_online_epochs=50 \
                  --config.agent_kwargs.critic_ensemble_size=2 --config.data_collection_particle_choosing_strategy="max_q_value" \
                  --config.evaluation_particle_choosing_strategy="max_q_value"