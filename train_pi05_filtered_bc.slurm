#!/bin/bash
#SBATCH --job-name=pi_05_finetuning_filtered_bc             # Job name
#SBATCH --nodes=1                              # Number of nodes
#SBATCH --gres=gpu:4                           # GPUs per node
#SBATCH --cpus-per-task=48                      # CPU cores per task
#SBATCH --mem=512G                              # Memory per node
#SBATCH --time=31-00:00:00 
#SBATCH --partition=preempt                    # Partition/queue name
#SBATCH --output=/data/user_data/sreyasv/VLA_ft_logs/logs/filtered_bc_5_libero_10_pi05_put_the_two_mocha_pots_on_the_stove_%x_%j.out   # Stdout log
#SBATCH --error=/data/user_data/sreyasv/VLA_ft_logs/logs/filtered_bc_5_libero_10_pi05_put_the_two_mocha_pots_on_the_stove_%x_%j.err    # Stderr log

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
export SAVE_DIR_PREFIX="/data/user_data/sreyasv/filtered_bc_pi05/libero_10_pi05_put_the_two_mocha_pots_on_the_stove/" 

# Run the training script
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 env -u PYOPENGL_PLATFORM python ./train_fbc_pi.py --environment_name=libero --wandb_experiment_name=filtered_bc_libero_10_5_ep_ft_put_the_two_mocha_pots_on_the_stove --config=./configs/libero_config.py:pi0 


