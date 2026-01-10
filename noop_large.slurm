#!/bin/bash
#SBATCH --job-name=rlvr-v1
#SBATCH --partition=general          # partition name
#SBATCH --gres=gpu:1
#SBATCH --constraint=VRAM_80GB
#SBATCH --cpus-per-task=20
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --output=/data/user_data/skowshik/rlvr-debug/outcome_%j.out      # Standard output file (%j is job ID)
#SBATCH --error=/data/user_data/skowshik/rlvr-debug/outcome_%j.out       # Standard error file (%j is job ID)


conda run -n openpi python noop.py

