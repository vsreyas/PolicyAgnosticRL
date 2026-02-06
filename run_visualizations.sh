#!/bin/bash

# List of episode IDs to process
EPISODE_IDS=(0 15 2 8 13 19)
# EPISODE_IDS=(1)

# Base paths
BASE_DIR="/home/skowshik/vla/codebase/PolicyAgnosticRL/critic_EVAL_calql_4_2_ch8k"
# If `BASE_DIR` does not exist, create it
if [ ! -d "${BASE_DIR}" ]; then
    mkdir -p "${BASE_DIR}"
fi

TFRECORD_BASE="/data/user_data/skowshik/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_base_rollouts_EVAL/image_replay_buffer/"
# TD3_CHECKPOINT="/home/skowshik/vla/codebase/PolicyAgnosticRL/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2/checkpoint_9000.pkl"
# Append BASE_DIR to the output base name
OUT_BASE_NAME="${BASE_DIR}/ch8k"
# PI_CONFIG_NAME="pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k"
PI_CONFIG_NAME="pi05_libero_gradacc2_2k"
TD3_GRPO_CHECKPOINT="/home/skowshik/vla/codebase/PolicyAgnosticRL/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_calql_10_2_clean_v2_balance/checkpoint_8000.pkl"

# Loop through each episode
for EPISODE_ID in "${EPISODE_IDS[@]}"; do
    echo "==========================================="
    echo "Processing Episode ${EPISODE_ID}"
    echo "==========================================="
    
    TFRECORD_PATH="${TFRECORD_BASE}/episode_${EPISODE_ID}.tfrecord"
    
    # Command 1: TD3 Basic UMAP Visualization
    # echo "Running TD3 basic UMAP visualization for episode ${EPISODE_ID}..."
    # python visualize_traj_predictions.py \
    #   --config=configs/libero_config.py:pi_residual_td3 \
    #   --agent_name=pi_residual_td3 \
    #   --pi_config_name=$PI_CONFIG_NAME \
    #   --tfrecord_path="${TFRECORD_PATH}" \
    #   --checkpoint_path="${TD3_CHECKPOINT}" \
    #   --output_dir="${OUT_BASE_NAME}${EPISODE_ID}_rs42" \
    #   --vis_type=2 \
    #   --config.image_replay_buffer_kwargs.load_action_samples=True
    
    # # Command 2: TD3 with Gradient Q-line (scale -1 to 1)
    # echo "Running TD3 with gradient Q-line (scale -1 to 1) for episode ${EPISODE_ID}..."
    # python visualize_traj_predictions.py \
    #   --config=configs/libero_config.py:pi_residual_td3 \
    #   --agent_name=pi_residual_td3 \
    #   --pi_config_name=$PI_CONFIG_NAME \
    #   --tfrecord_path="${TFRECORD_PATH}" \
    #   --checkpoint_path="${TD3_CHECKPOINT}" \
    #   --output_dir="${OUT_BASE_NAME}${EPISODE_ID}_gradq_lim1_rs42" \
    #   --vis_type=2 \
    #   --config.image_replay_buffer_kwargs.load_action_samples=True \
    #   --plot_grad_q_line=True \
    #   --grad_line_scale_low=-1.0 \
    #   --grad_line_scale_high=1.0
    
    # # Command 3: TD3 with Gradient Q-line (scale -5 to 5)
    # echo "Running TD3 with gradient Q-line (scale -5 to 5) for episode ${EPISODE_ID}..."
    # python visualize_traj_predictions.py \
    #   --config=configs/libero_config.py:pi_residual_td3 \
    #   --agent_name=pi_residual_td3 \
    #   --pi_config_name=$PI_CONFIG_NAME \
    #   --tfrecord_path="${TFRECORD_PATH}" \
    #   --checkpoint_path="${TD3_CHECKPOINT}" \
    #   --output_dir="${OUT_BASE_NAME}${EPISODE_ID}_gradq_lim5_rs42" \
    #   --vis_type=2 \
    #   --config.image_replay_buffer_kwargs.load_action_samples=True \
    #   --plot_grad_q_line=True \
    #   --grad_line_scale_low=-5.0 \
    #   --grad_line_scale_high=5.0
    
    # # Command 4: TD3 with Gradient Q-line (scale -10 to 10)
    # echo "Running TD3 with gradient Q-line (scale -10 to 10) for episode ${EPISODE_ID}..."
    # python visualize_traj_predictions.py \
    #   --config=configs/libero_config.py:pi_residual_td3 \
    #   --agent_name=pi_residual_td3 \
    #   --pi_config_name=$PI_CONFIG_NAME \
    #   --tfrecord_path="${TFRECORD_PATH}" \
    #   --checkpoint_path="${TD3_CHECKPOINT}" \
    #   --output_dir="${OUT_BASE_NAME}${EPISODE_ID}_gradq_lim10_rs42" \
    #   --vis_type=2 \
    #   --config.image_replay_buffer_kwargs.load_action_samples=True \
    #   --plot_grad_q_line=True \
    #   --grad_line_scale_low=-10.0 \
    #   --grad_line_scale_high=10.0
    
    # # Command 5: TD3 with generated action samples from ep5 base model
    # echo "Running TD3 with generated action samples from ep5 base model for episode ${EPISODE_ID}..."
    # python visualize_traj_predictions.py \
    #   --config=configs/libero_config.py:pi_residual_td3 \
    #   --agent_name=pi_residual_td3 \
    #   --pi_config_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k \
    #   --tfrecord_path="${TFRECORD_PATH}" \
    #   --checkpoint_path="${TD3_CHECKPOINT}" \
    #   --output_dir="${OUT_BASE_NAME}${EPISODE_ID}_w_ep5_base_rs42" \
    #   --vis_type=2 \
    #   --config.image_replay_buffer_kwargs.load_action_samples=True \
    #   --generate_action_samples=True \
    #   --num_action_samples=16
    
    # echo "Completed Episode ${EPISODE_ID}"
    # echo ""

    # Command 6: TD3 GRPO with residual visualization and critic params
    echo "Running TD3 GRPO with residual visualization for episode ${EPISODE_ID}..."
    python visualize_traj_predictions.py \
      --config=configs/libero_config.py:pi_residual_td3_grpo \
      --agent_name=pi_residual_td3_grpo \
      --pi_config_name=$PI_CONFIG_NAME \
      --tfrecord_path="${TFRECORD_PATH}" \
      --checkpoint_path="${TD3_GRPO_CHECKPOINT}" \
      --output_dir="${OUT_BASE_NAME}${EPISODE_ID}" \
      --vis_type=2 \
      --config.image_replay_buffer_kwargs.load_action_samples=True \
      --num_qs=4 \

    echo "Completed Episode ${EPISODE_ID}"
    echo ""
done

echo "==========================================="
echo "All episodes processed successfully!"
echo "==========================================="
