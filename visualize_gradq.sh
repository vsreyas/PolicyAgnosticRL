#!/bin/bash

# List of episode IDs to process
EPISODE_IDS=(0 15 2 8 13 19)

# Base paths
BASE_DIR="/home/skowshik/vla/codebase/PolicyAgnosticRL/gradq_visualization_EVAL_calql4_2_ch8k_qstar_rng_fix_10.5k"
if [ ! -d "${BASE_DIR}" ]; then
    mkdir -p "${BASE_DIR}"
fi

TFRECORD_BASE="/data/user_data/skowshik/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_base_rollouts_EVAL/image_replay_buffer/"
CHECKPOINT="/home/skowshik/vla/codebase/PolicyAgnosticRL/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_calql_10_2_clean_v2_balance/checkpoint_8000.pkl"
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
      --num_qs=4 \
      --num_min_qs=2 \
      --plot_type=3 \
      --timestep=-1 \
      --clip_actions \
      --sub_base_q_network_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/pi05_libero_gradacc2_2k_q_star_sarsa_10_2_rng_fix/checkpoint_10500.pkl"

    echo "Completed Episode ${EPISODE_ID}"
    echo ""
done

echo "==========================================="
echo "All episodes processed successfully!"
echo "==========================================="
