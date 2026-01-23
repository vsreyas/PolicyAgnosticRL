#!/bin/bash
#SBATCH --job-name=pi_05_q_wms             # Job name
#SBATCH --nodes=1                              # Number of nodes
#SBATCH --gres=gpu:1                           # GPUs per node
#SBATCH --cpus-per-task=48                      # CPU cores per task
#SBATCH --mem=512G                              # Memory per node
#SBATCH --time=48:00:00                        # Walltime (hh:mm:ss)
#SBATCH --partition=general                    # Partition/queue name
#SBATCH --output=/data/user_data/sreyasv/VLA_ft_logs/logs/q_warmstart_debug_%x_%j.out   # Stdout log
#SBATCH --error=/data/user_data/sreyasv/VLA_ft_logs/logs/q_warmstart_debug_%x_%j.err    # Stderr log

# -------------------------------
# Environment setup
# -------------------------------
source /home/sreyasv/miniconda3/etc/profile.d/conda.sh
conda activate parl_pi     # use the correct env (adjust if it's 'seer' or something else)

mkdir -p /data/user_data/sreyasv/VLA_ft_logs/logs/

# -------------------------------
# Run command
# -------------------------------
echo "Starting training on $(hostname)"
echo "Running at $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

# Disable XLA GPU memory preallocation (helps avoid OOMs on TPU/GPU)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export SAVE_DIR_PREFIX="/data/user_data/sreyasv/q_chunking_pi_no_-1_10_critic_ws_debug/libero_10_pi05_put_the_two_mocha_pots_on_the_stove/debug_plots" 

# Run the training script
XLA_PYTHON_CLIENT_PREALLOCATE=false env -u PYOPENGL_PLATFORM python ./critic_warmstart_debug.py --environment_name=libero \
                  --wandb_experiment_name=pi0_5_finetuning_parl_qchunking_-1_10_critic_ws_debug \
                  --config=./configs/libero_config.py:parl_calql_pi0 \
                  --seed=0\
                  --config.agent_kwargs.critic_network_kwargs.hidden_dims=256,256,256,256 \
                  --config.base_policy_path=pi0 --config.agent_kwargs.cql_alpha=0.000 \
                  --num_offline_epochs=500 --num_online_epochs=50 \
                  --config.agent_kwargs.critic_ensemble_size=2 --config.data_collection_particle_choosing_strategy="max_q_value" \
                  --config.evaluation_particle_choosing_strategy="max_q_value"

