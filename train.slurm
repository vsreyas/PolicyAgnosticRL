#!/bin/bash
#SBATCH --job-name=openpi-loraft-debug
#SBATCH --partition=general          # partition name
#SBATCH --qos=normal            # qos
#SBATCH --gres=gpu:A6000:8
#SBATCH --cpus-per-task=8          # number of CPUs (increased)
#SBATCH --time=48:00:00          # time (1 day)
#SBATCH --mem=128G                   # 128GB total RAM (increased from 48GB)
#SBATCH --output=/data/user_data/skowshik/openpi-loraft-debug/outcome_%j.out      # Standard output file (%j is job ID)
#SBATCH --error=/data/user_data/skowshik/openpi-loraft-debug/outcome_%j.out       # Standard error file (%j is job ID)


# XLA_PYTHON_CLIENT_PREALLOCATE=false env -u PYOPENGL_PLATFORM python ./train_pi.py --environment_name=libero --wandb_experiment_name=pi0_trial --config=./configs/libero_config.py:pi0 --num_offline_epochs=30 --num_online_epochs=0 --seed=0 --config.batch_size=256 --config.agent_kwargs.batch_size=256

XLA_PYTHON_CLIENT_PREALLOCATE=false env -u PYOPENGL_PLATFORM python ./train_pi.py --environment_name=libero --wandb_experiment_name=pi0_trial --config=./configs/libero_config.py:pi0 --num_offline_epochs=30 --num_online_epochs=0 --seed=0 --task_name="put both moka pots on the stove" --config.eval_interval=1 --config.batch_size=256 --config.agent_kwargs.batch_size=256

