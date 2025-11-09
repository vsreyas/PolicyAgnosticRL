#!/bin/bash
#SBATCH --job-name=ddpm_calvin_8_blocks_512_hd              # Job name
#SBATCH --nodes=1                              # Number of nodes
#SBATCH --gres=gpu:4                           # GPUs per node
#SBATCH --cpus-per-task=64                      # CPU cores per task
#SBATCH --mem=256G                              # Memory per node
#SBATCH --time=48:00:00                        # Walltime (hh:mm:ss)
#SBATCH --partition=general                    # Partition/queue name
#SBATCH --output=/data/user_data/sreyasv/diffusion_calvin/logs/8_blocks_512_hd_%x_%j.out   # Stdout log
#SBATCH --error=/data/user_data/sreyasv/diffusion_calvin/logs/8_blocks_512_hd_%x_%j.err    # Stderr log

# -------------------------------
# Environment setup
# -------------------------------
source /home/sreyasv/miniconda3/etc/profile.d/conda.sh
conda activate parl     # use the correct env (adjust if it's 'seer' or something else)

mkdir -p /data/user_data/sreyasv/diffusion_calvin/logs/

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
  --environment_name=calvin \
  --wandb_experiment_name=ddpm_bc_with_lang_calvin_8_blocks_512_hd \
  --config=./configs/calvin_config.py:ddpm \
  --num_offline_epochs=3000 \
  --num_online_epochs=0 \
  --seed=0

echo "Job completed at $(date)"
