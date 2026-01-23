#!/bin/bash

# Disable XLA GPU memory preallocation (helps avoid OOMs on TPU/GPU)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export SAVE_DIR_PREFIX="/data/user_data/sreyasv/eval_ouptuts/pi-05/pretrained_pi05_libero/" 

# Run the training script
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 env -u PYOPENGL_PLATFORM python ./eval_pi.py --environment_name=libero \
                --wandb_experiment_name=eval_pretrained_pi05_libero \
                --config=./configs/libero_config.py:pi0 \
                --num_offline_epochs=1 --num_online_epochs=1 \
                --num_train_steps_per_offline_epoch=500 \
                --PI_config=pi05_libero_eval \
                --config.num_eval_episodes=50 \
                --config.num_episodes_per_video=50\
