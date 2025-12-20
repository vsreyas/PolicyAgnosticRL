#!/bin/bash
# Example usage script for dump_vlm_actions.py

# Example 1: Dump VLM actions from replay buffer
echo "Example 1: Dumping VLM actions from replay buffer"
python scripts/dump_vlm_actions.py \
    --config=configs/your_config.py \
    --task_name="put both moka pots on the stove" \
    --output_path="./outputs/vlm_actions_replay.pkl" \
    --replay_buffer_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/libero_10_pi05_put_the_two_mocha_pots_on_the_stove/episode_0.tfrecord" \
    --num_actions_to_sample=4 \
    --num_edit_samples=4 \
    --seed=0

# Example 2: Dump VLM actions from offline dataset (uses config's libero_tfrecord_regexp)
echo "Example 2: Dumping VLM actions from offline dataset"
python scripts/dump_vlm_actions.py \
    --config=configs/your_config.py \
    --task_name="put both moka pots on the stove" \
    --output_path="./outputs/vlm_actions_offline.pkl" \
    --num_actions_to_sample=8 \
    --seed=42

# Example 3: Process limited number of batches for testing
echo "Example 3: Processing only 10 batches for quick testing"
python scripts/dump_vlm_actions.py \
    --config=configs/your_config.py \
    --task_name="put both moka pots on the stove" \
    --replay_buffer_path="/path/to/replay_buffer/*.tfrecord" \
    --output_path="./outputs/vlm_actions_test.pkl" \
    --max_batches=10 \
    --seed=0

# Example 4: Load and inspect the dumped data
echo "Example 4: Loading and inspecting dumped data"
python scripts/load_vlm_actions.py ./outputs/vlm_actions_replay.pkl




