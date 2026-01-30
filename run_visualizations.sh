#!/bin/bash

# List of episode IDs to process
EPISODE_IDS=(4 13 23 39 53)

# Base paths
TFRECORD_BASE="/data/user_data/skowshik/gradacc_2k_base_policy_rollouts/image_replay_buffer"
TD3_CHECKPOINT="/home/skowshik/vla/codebase/PolicyAgnosticRL/td3_gradacc2_2k_base_ws_balanced/checkpoint_1000.pkl"

# Loop through each episode
for EPISODE_ID in "${EPISODE_IDS[@]}"; do
    echo "==========================================="
    echo "Processing Episode ${EPISODE_ID}"
    echo "==========================================="
    
    TFRECORD_PATH="${TFRECORD_BASE}/episode_${EPISODE_ID}.tfrecord"
    
    # Command 1: TD3 Basic UMAP Visualization
    echo "Running TD3 basic UMAP visualization for episode ${EPISODE_ID}..."
    python visualize_traj_predictions.py \
      --config=configs/libero_config.py:pi_residual_td3 \
      --agent_name=pi_residual_td3 \
      --pi_config_name=pi05_libero_gradacc2_2k \
      --tfrecord_path="${TFRECORD_PATH}" \
      --checkpoint_path="${TD3_CHECKPOINT}" \
      --output_dir="./td3_gradacc2_2k_base_ws_balanced_trajvis_1k_ep${EPISODE_ID}_rs42" \
      --vis_type=2 \
      --config.image_replay_buffer_kwargs.load_action_samples=True
    
    # Command 2: TD3 with Gradient Q-line (scale -1 to 1)
    echo "Running TD3 with gradient Q-line (scale -1 to 1) for episode ${EPISODE_ID}..."
    python visualize_traj_predictions.py \
      --config=configs/libero_config.py:pi_residual_td3 \
      --agent_name=pi_residual_td3 \
      --pi_config_name=pi05_libero_gradacc2_2k \
      --tfrecord_path="${TFRECORD_PATH}" \
      --checkpoint_path="${TD3_CHECKPOINT}" \
      --output_dir="./td3_gradacc2_2k_base_ws_balanced_trajvis_1k_ep${EPISODE_ID}_gradq_lim1_rs42" \
      --vis_type=2 \
      --config.image_replay_buffer_kwargs.load_action_samples=True \
      --plot_grad_q_line=True \
      --grad_line_scale_low=-1.0 \
      --grad_line_scale_high=1.0
    
    # Command 3: TD3 with Gradient Q-line (scale -5 to 5)
    echo "Running TD3 with gradient Q-line (scale -5 to 5) for episode ${EPISODE_ID}..."
    python visualize_traj_predictions.py \
      --config=configs/libero_config.py:pi_residual_td3 \
      --agent_name=pi_residual_td3 \
      --pi_config_name=pi05_libero_gradacc2_2k \
      --tfrecord_path="${TFRECORD_PATH}" \
      --checkpoint_path="${TD3_CHECKPOINT}" \
      --output_dir="./td3_gradacc2_2k_base_ws_balanced_trajvis_1k_ep${EPISODE_ID}_gradq_lim5_rs42" \
      --vis_type=2 \
      --config.image_replay_buffer_kwargs.load_action_samples=True \
      --plot_grad_q_line=True \
      --grad_line_scale_low=-5.0 \
      --grad_line_scale_high=5.0
    
    # Command 4: TD3 with Gradient Q-line (scale -10 to 10)
    echo "Running TD3 with gradient Q-line (scale -10 to 10) for episode ${EPISODE_ID}..."
    python visualize_traj_predictions.py \
      --config=configs/libero_config.py:pi_residual_td3 \
      --agent_name=pi_residual_td3 \
      --pi_config_name=pi05_libero_gradacc2_2k \
      --tfrecord_path="${TFRECORD_PATH}" \
      --checkpoint_path="${TD3_CHECKPOINT}" \
      --output_dir="./td3_gradacc2_2k_base_ws_balanced_trajvis_1k_ep${EPISODE_ID}_gradq_lim10_rs42" \
      --vis_type=2 \
      --config.image_replay_buffer_kwargs.load_action_samples=True \
      --plot_grad_q_line=True \
      --grad_line_scale_low=-10.0 \
      --grad_line_scale_high=10.0
    
    # Command 5: TD3 with generated action samples from ep5 base model
    echo "Running TD3 with generated action samples from ep5 base model for episode ${EPISODE_ID}..."
    python visualize_traj_predictions.py \
      --config=configs/libero_config.py:pi_residual_td3 \
      --agent_name=pi_residual_td3 \
      --pi_config_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k \
      --tfrecord_path="${TFRECORD_PATH}" \
      --checkpoint_path="${TD3_CHECKPOINT}" \
      --output_dir="./td3_gradacc2_2k_base_ws_balanced_trajvis_1k_ep${EPISODE_ID}_w_ep5_base_rs42" \
      --vis_type=2 \
      --config.image_replay_buffer_kwargs.load_action_samples=True \
      --generate_action_samples=True \
      --num_action_samples=16
    
    echo "Completed Episode ${EPISODE_ID}"
    echo ""
done

echo "==========================================="
echo "All episodes processed successfully!"
echo "==========================================="
