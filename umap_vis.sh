# CRITIC_PARAMS_PATH="/home/skowshik/vla/codebase/PolicyAgnosticRL/results_expo_debug-td-clean_v12-scale5_actnorm/seed_0/checkpoint_5000.pkl"
CRITIC_PARAMS_PATH="/home/skowshik/vla/codebase/PolicyAgnosticRL/critic_cache_ws_ep5-scale10_200_actnorm-num_q2-swish-v1/seed_0/checkpoint_10000.pkl"
REPLAY_BUFFER_PATH="/data/user_data/skowshik/libero_10_pi05_put_the_two_mocha_pots_on_the_stove/results_expo/image_replay_buffer/episode_1.tfrecord"
OUTPUT_PATH="/home/skowshik/vla/codebase/PolicyAgnosticRL/umap_vis_episode_1_ep5_critic_ws_10k/"

mkdir -p $OUTPUT_PATH
python dump_tf_record.py --tfrecord_path $REPLAY_BUFFER_PATH --output_gif_path $OUTPUT_PATH/img_base.mp4 --camera_view "base"
python dump_tf_record.py --tfrecord_path $REPLAY_BUFFER_PATH --output_gif_path $OUTPUT_PATH/img_wrist.mp4 --camera_view "wrist"

CUDA_VISIBLE_DEVICES=0 python umap_action_vis.py \
    --config=configs/libero_config.py:expo \
    --task_name="put both moka pots on the stove" \
    --output_path=$OUTPUT_PATH \
    --replay_buffer_path=$REPLAY_BUFFER_PATH \
    --seed=0 \
    --config.agent_kwargs.batch_size=512 \
    --config.batch_size=512 \
    --critic_params_path=$CRITIC_PARAMS_PATH \
    --mp4_path_base_view=$OUTPUT_PATH/img_base.mp4 \
    --mp4_path_wrist_view=$OUTPUT_PATH/img_wrist.mp4


CRITIC_PARAMS_PATH="/home/skowshik/vla/codebase/PolicyAgnosticRL/critic_cache_ws_ep5-scale10_200_actnorm-num_q2-swish-v1/seed_0/checkpoint_10000.pkl"
REPLAY_BUFFER_PATH="/data/user_data/skowshik/libero_10_pi05_put_the_two_mocha_pots_on_the_stove/results_expo/image_replay_buffer/episode_13.tfrecord"
OUTPUT_PATH="/home/skowshik/vla/codebase/PolicyAgnosticRL/umap_vis_episode_13_ep5_critic_ws_10k/"

mkdir -p $OUTPUT_PATH
python dump_tf_record.py --tfrecord_path $REPLAY_BUFFER_PATH --output_gif_path $OUTPUT_PATH/img_base.mp4 --camera_view "base"
python dump_tf_record.py --tfrecord_path $REPLAY_BUFFER_PATH --output_gif_path $OUTPUT_PATH/img_wrist.mp4 --camera_view "wrist"

CUDA_VISIBLE_DEVICES=0 python umap_action_vis.py \
    --config=configs/libero_config.py:expo \
    --task_name="put both moka pots on the stove" \
    --output_path=$OUTPUT_PATH \
    --replay_buffer_path=$REPLAY_BUFFER_PATH \
    --seed=0 \
    --config.agent_kwargs.batch_size=512 \
    --config.batch_size=512 \
    --critic_params_path=$CRITIC_PARAMS_PATH \
    --mp4_path_base_view=$OUTPUT_PATH/img_base.mp4 \
    --mp4_path_wrist_view=$OUTPUT_PATH/img_wrist.mp4

