export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR:-/tmp}/pycache"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/xdg_cache"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mplconfig"
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/numba_cache"

export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

export CUDA_VISIBLE_DEVICES=0
conda activate parl

XLA_PYTHON_CLIENT_PREALLOCATE=false env -u PYOPENGL_PLATFORM python ./train_expo_pi_cache.py --config=configs/libero_config.py:expo --task_name="put both moka pots on the stove" --num_online_epochs=10000 --seed=0 --task_name="put both moka pots on the stove" --config.eval_interval=1 --config.batch_size=32 --config.agent_kwargs.batch_size=32 --environment_name=libero --wandb_experiment_name=expo_trial --num_edit_samples=4 --num_actions_to_sample=4 --num_offline_epochs=0 --final_step_sparse_reward=False --critic_warmup_steps 0 --online_trajectory_collection_frequency 50 --config.utd_ratio=2 --config.num_eval_episodes=10 --config.num_episodes_per_video=2 --config.eval_interval=50 --reward_scale=1.0 --reward_bias=-0.1 --config.libero_tfrecord_regexp="/home/skowshik/vla/codebase/PolicyAgnosticRL/BACKUP_libero_10_pi05_put_the_two_mocha_pots_on_the_stove/results_expo/seed_0/image_replay_buffer/*.tfrecord" --critic_params_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/results_expo_debug-td_mc_v3/seed_0/checkpoint_200000.pkl" --reward_scale=1.0 --reward_bias=-0.1 --filter_successful_trajectories=False | tee logs/debug.log
