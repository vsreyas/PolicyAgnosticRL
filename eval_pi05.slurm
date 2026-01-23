#!/bin/bash
#SBATCH --job-name=pi_05_finetuning_filtered_bc             # Job name
#SBATCH --nodes=1                              # Number of nodes
#SBATCH --gres=gpu:4                           # GPUs per node
#SBATCH --cpus-per-task=48                      # CPU cores per task
#SBATCH --mem=512G                              # Memory per node
#SBATCH --time=31-00:00:00 
#SBATCH --partition=preempt                    # Partition/queue name
#SBATCH --output=/data/user_data/sreyasv/eval_logs/logs/eval_pi05_%x_%j.out   # Stdout log
#SBATCH --error=/data/user_data/sreyasv/eval_logs/logs/eval_pi05_%x_%j.err    # Stderr log

# -------------------------------
# Environment setup
# -------------------------------
source /home/sreyasv/miniconda3/etc/profile.d/conda.sh
conda activate parl_pi     # use the correct env (adjust if it's 'seer' or something else)

mkdir -p /data/user_data/sreyasv/eval_logs/logs/

# -------------------------------
# Run command
# -------------------------------
echo "Starting training on $(hostname)"
echo "Running at $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

# Disable XLA GPU memory preallocation (helps avoid OOMs on TPU/GPU)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export SAVE_DIR_PREFIX="/data/user_data/sreyasv/eval_ouptuts/pi-05/pi05_libero_lora_checkpoint_mokapots_2k/" 

# Run the training script
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 env -u PYOPENGL_PLATFORM python ./eval_pi.py --environment_name=libero \
                --wandb_experiment_name=eval_pi05_libero_lora_checkpoint_mokapots_2k \
                --config=./configs/libero_config.py:pi0 \
                --num_offline_epochs=1 --num_online_epochs=1 \
                --num_train_steps_per_offline_epoch=500 \
                --PI_config=pi05_libero_eval \
                --config.num_eval_episodes=50 \
                --config.num_episodes_per_video=50\
                --logtostderr 2>&1 | tee ${SAVE_DIR_PREFIX}/training_output_${SLURM_JOB_ID}.log

