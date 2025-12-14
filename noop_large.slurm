#!/bin/bash
#SBATCH --job-name=openpi-loraft-vision_fullft
#SBATCH --partition=general          # partition name
#SBATCH --constraint=VRAM_80GB
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00          # time (1 day)
#SBATCH --output=/data/user_data/skowshik/rlvr-debug/outcome_%j.out      # Standard output file (%j is job ID)
#SBATCH --error=/data/user_data/skowshik/rlvr-debug/outcome_%j.out       # Standard error file (%j is job ID)

conda run -n openpi python noop.py