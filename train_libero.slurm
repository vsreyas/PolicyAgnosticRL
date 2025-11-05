#!/bin/bash
#SBATCH --job-name=ddpm_bc_libero              # Job name
#SBATCH --nodes=1                              # Number of nodes
#SBATCH --gres=gpu:4                           # GPUs per node
#SBATCH --cpus-per-task=32                      # CPU cores per task
#SBATCH --mem=256G                              # Memory per node
#SBATCH --time=48:00:00                        # Walltime (hh:mm:ss)
#SBATCH --partition=general                    # Partition/queue name
#SBATCH --output=/data/user_data/sreyasv/VLA_ft_logs/logs/%x_%j.out   # Stdout log
#SBATCH --error=/data/user_data/sreyasv/VLA_ft_logs/logs/%x_%j.err    # Stderr log

# -------------------------------
# Environment setup
# -------------------------------
source /home/sreyasv/miniconda3/etc/profile.d/conda.sh
conda activate parl     # use the correct env (adjust if it's 'seer' or something else)

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
python ./train.py \
  --environment_name=libero \
  --wandb_experiment_name=ddpm_bc_wrist_view_lang_resnet_34 \
  --config=./configs/libero_config.py:ddpm \
  --num_offline_epochs=1000 \
  --num_online_epochs=0 \
  --seed=0

echo "Job completed at $(date)"
