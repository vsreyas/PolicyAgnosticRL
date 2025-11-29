#!/bin/bash
#SBATCH --job-name=pi_05_finetuning             # Job name
#SBATCH --nodes=1                              # Number of nodes
#SBATCH --gres=gpu:4                           # GPUs per node
#SBATCH --cpus-per-task=48                      # CPU cores per task
#SBATCH --mem=512G                              # Memory per node
#SBATCH --time=48:00:00                        # Walltime (hh:mm:ss)
#SBATCH --partition=general                    # Partition/queue name
#SBATCH --output=/data/user_data/sreyasv/VLA_ft_logs/logs/libero_10_pi05_put_the_two_mocha_pots_on_the_stove_%x_%j.out   # Stdout log
#SBATCH --error=/data/user_data/sreyasv/VLA_ft_logs/logs/libero_10_debug_%x_%j.err    # Stderr log

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

# Run the training script
XLA_PYTHON_CLIENT_PREALLOCATE=false env -u PYOPENGL_PLATFORM python ./train_pi.py --environment_name=libero \
                  --wandb_experiment_name=pi0_5_finetuning_parl \
                  --config=./configs/libero_config.py:parl_calql_pi0 \
                  --seed=0 --reward_bias=-1 \
                  --config.agent_kwargs.critic_network_kwargs.hidden_dims=256,256,256,256 \
                  --config.base_policy_path=pi0 --config.agent_kwargs.cql_alpha=0.005 \
                  --num_offline_epochs=20 --num_online_epochs=100 --config.agent_kwargs.distributional_critic_kwargs.q_min=-100 \
                  --config.agent_kwargs.critic_ensemble_size=1 --config.data_collection_particle_choosing_strategy="max_q_value" \
                  --config.evaluation_particle_choosing_strategy="max_q_value"

