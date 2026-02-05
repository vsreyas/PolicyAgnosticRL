#!/bin/bash

# List of episode IDs to process
EPISODE_IDS=(1 10 83 93 53 39)

# Base paths
BASE_DIR="./grad_q_vis_pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_base_rollouts_subbase_4k_each"
if [ ! -d "${BASE_DIR}" ]; then
    mkdir -p "${BASE_DIR}"
fi

TFRECORD_BASE="/data/user_data/skowshik/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_base_rollouts/image_replay_buffer"
CHECKPOINT="/home/skowshik/vla/codebase/PolicyAgnosticRL/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2/checkpoint_4000.pkl"
PI_CONFIG_NAME="pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k"

# Loop through each episode
for EPISODE_ID in "${EPISODE_IDS[@]}"; do
    echo "==========================================="
    echo "Processing Episode ${EPISODE_ID}"
    echo "==========================================="

    TFRECORD_PATH="${TFRECORD_BASE}/episode_${EPISODE_ID}.tfrecord"

    python visualize_grad_q_plots.py \
      --config=configs/libero_config.py:pi_residual_td3_grpo \
      --tfrecord_path="${TFRECORD_PATH}" \
      --checkpoint_path="${CHECKPOINT}" \
      --output_dir="${BASE_DIR}/ep${EPISODE_ID}" \
      --pi_config_name="${PI_CONFIG_NAME}" \
      --agent_name=pi_residual_td3 \
      --lim=10.0 \
      --num_line_points=50 \
      --eta=0.01 \
      --num_grad_steps=50 \
      --plot_type=3 \
      --timestep=-1 \
      --clip_actions \
      --sub_base_q_network_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/pi05_libero_gradacc2_2k_q_star_sarsa_10_2/checkpoint_4000.pkl"

    echo "Completed Episode ${EPISODE_ID}"
    echo ""
done

echo "==========================================="
echo "All episodes processed successfully!"
echo "==========================================="
