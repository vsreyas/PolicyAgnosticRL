# Policy Agnostic RL

Jax codebase for [Policy Agnostic RL: Offline RL and Online RL Fine-Tuning of Any Policy Class and Backbone](https://arxiv.org/abs/2412.06685).

## Environment

### Clean Setup on Babel
```
conda create -n parl python=3.11
conda activate parl
pip install uv
uv pip install -e .
uv pip install -r requirements_og.txt

# Install libero from parl branch
git clone https://github.com/vsreyas/LIBERO.git
git checkout parl
uv pip install -r requirements.txt
uv pip install -e .

# Install openpi from parl branch
git clone https://github.com/vsreyas/openpi.git
git checkout parl
# Remove "rerun-sdk==0.26.2"
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv pip install "augmax>=0.3.4"
uv pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

```
pip install "opencv-python-headless==4.10.0.84"

```

```
conda create -n parl python=3.11
conda activate parl
pip install uv
uv pip install -e .
# Keep only part above `#install libero and calvin`
uv pip install -r requirements_clean.txt

git clone https://github.com/vsreyas/LIBERO.git
git checkout parl
uv pip install -r requirements.txt
uv pip install -e .

git clone https://github.com/vsreyas/openpi.git
git checkout parl
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv pip install "augmax>=0.3.4"


```

If you run into GL/glew.h: No such file or directory, run this:
```
conda install -c conda-forge glew
conda install -c conda-forge mesalib
conda install -c menpo glfw3
```

### Clean setup on Maxlab PC
```
conda create -n parl python=3.11
conda activate parl
pip install uv
uv pip install -e .

# Keep only part above `#install libero and calvin`
uv pip install -r requirements_clean.txt

git clone https://github.com/vsreyas/LIBERO.git
git checkout parl
uv pip install -r requirements.txt
uv pip install -e .

git clone https://github.com/vsreyas/openpi.git
git checkout parl
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv pip install "augmax>=0.3.4"

# Comment part above `#install libero and calvin` and uncomment below
uv pip install -r requirements.txt

conda activate parl
pip uninstall -y torch torchvision torchaudio
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu130
pip install wandb --upgrade

```


For TPU
```
pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```
See the [Jax Github page](https://github.com/google/jax) for more details on installing Jax.

## Example training scripts

First pre-train a Diffusion Policy with BC:
```
XLA_PYTHON_CLIENT_PREALLOCATE=false env -u PYOPENGL_PLATFORM python ./train.py --environment_name=antmaze-{size:large,medium}-{dataset:diverse,play}-v2 --wandb_experiment_name=ddpm_bc_antmaze-{size}-{dataset}-v2 --config=./configs/state_config.py:ddpm --num_offline_epochs=3000 --num_online_epochs=0 --seed={seed:0-4}
```

Then run the PA-RL script, which does offline RL pre-training followed by online fine-tuning:
```
XLA_PYTHON_CLIENT_PREALLOCATE=false env -u PYOPENGL_PLATFORM python ./train.py --environment_name=antmaze-{size:large,medium}-{dataset:diverse,play}-v2 --wandb_experiment_name=parl_antmaze-{size}-{dataset}-v2 --config=./configs/state_config.py:parl_calql --seed={seed:0-4} --reward_bias=-1 --config.agent_kwargs.critic_network_kwargs.hidden_dims=256,256,256,256 --config.base_policy_path=ddpm:./results/PA-RL/ddpm_bc_antmaze-{size}-{dataset}-v2/seed_{seed}/agent_checkpoints/checkpoint_3000/ --config.agent_kwargs.cql_alpha=0.005 --num_offline_epochs=1000 --num_online_epochs=1000 --config.agent_kwargs.distributional_critic_kwargs.q_min=-100 --config.agent_kwargs.critic_ensemble_size=10 --config.data_collection_particle_choosing_strategy="max_q_value" --config.evaluation_particle_choosing_strategy="max_q_value"
```
To train faster on small GPUs, consider reducing the critic ensemble size to 2.

## Real Robot Setup

Our real robot experiments are based on the [Bridge Data Robot](https://github.com/MaxSobolMark/bridge_data_robot) repo. Follow instructions there for environment setup and demo collection if applicable.

If your task involves demo collection, you need to process the raw demonstrations into tfrecords, and create a config in `configs.bridgedata_config.py` for your task. Starting from the raw files the Bridge Data Robot repo produces, first run `scripts/bridgedata_raw_to_numpy.py`, and then `scripts/bridgedata_numpy_to_tfrecord.py`. Then, specify the directory to your tfrecords in `configs.bridgedata_config.py`, and pass the flag `--bridgedata_config=./configs/bridgedata_config.py:your_task_name`.

If you would like to do agent training in a separate computer than the one used for controlling the robot, you can use the `--train_on_separate_computer_mode` option to create separate processes for environment steps and agent training.

## OpenVLA Fine-Tuning
To specify training OpenVLA, you need to change the base policy type in the `config.base_policy_path` flag to `--config.base_policy_path=openvla`.
Additionally, caching workers need to be running in parallel to the main training process. Here is an example script for the caching workers:
```
XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/openvla_caching_worker.py --checkpoints_dir=./results/PA-RL/ft_openvla/seed_0/base_policy_checkpoints_from_agent_trainer --instruction="put eggplant in pot" --repeat=8 --worker_id={worker_id:0-11} --num_workers=12
```
The computer physically connected to the robot might not be powerful enough to run OpenVLA at a reasonable speed. You can run policy inference on a separate computer by adjusting the `ip` and `port` parameters in the `real_config.py` file (i.e., run the robot server on one machine, and then specify that machine's IP on the config).

## Acknowledgements
This codebase is based on the [BridgeData V2](https://github.com/rail-berkeley/bridge_data_v2) repo.
The auto-regressive transformer policy class was implemented by Bhavya Agrawalla, Khush Agrawal, and Fahim Tajwar.

In case of any questions, feel free to contact me at maxsobolmark at cmu dot edu


## Training Libero 
XLA_PYTHON_CLIENT_PREALLOCATE=false env -u PYOPENGL_PLATFORM python ./train.py --environment_name=libero --wandb_experiment_name=ddpm_bc_trial --config=./configs/libero_config.py:ddpm --num_offline_epochs=30 --num_online_epochs=0 --seed=0

XLA_PYTHON_CLIENT_PREALLOCATE=false env -u PYOPENGL_PLATFORM python ./train.py --environment_name=antmaze-medium-diverse-v2 --wandb_experiment_name=ddpm_bc_antmaze-medium-diverse-v2 --config=./configs/state_config.py:ddpm --num_offline_epochs=3000 --num_online_epochs=0 --seed=0

XLA_PYTHON_CLIENT_PREALLOCATE=false env -u PYOPENGL_PLATFORM python ./train.py --environment_name=calvin --wandb_experiment_name=ddpm_bc_trial --config=./configs/calvin_config.py:ddpm --num_offline_epochs=30 --num_online_epochs=0 --seed=0

Note setup libero and calvin separately by installing their files
 we only need calvin_env and not the other components


XLA_PYTHON_CLIENT_PREALLOCATE=false env -u PYOPENGL_PLATFORM python ./train_pi.py --environment_name=libero --wandb_experiment_name=pi0_trial --config=./configs/libero_config.py:pi0 --num_offline_epochs=30 --num_online_epochs=0 --seed=0 --task_name="put both moka pots on the stove" --config.eval_interval=1

XLA_PYTHON_CLIENT_PREALLOCATE=false env -u PYOPENGL_PLATFORM python ./train_pi.py --environment_name=libero --wandb_experiment_name=pi0_trial_parl --config=./configs/libero_config.py:parl_calql_pi0 --seed=0 --reward_bias=-1 --config.agent_kwargs.critic_network_kwargs.hidden_dims=256,256,256,256 --config.base_policy_path=pi0 --config.agent_kwargs.cql_alpha=0.005 --num_offline_epochs=1000 --num_online_epochs=1000 --config.agent_kwargs.distributional_critic_kwargs.q_min=-100 --config.agent_kwargs.critic_ensemble_size=2 --config.data_collection_particle_choosing_strategy="max_q_value" --config.evaluation_particle_choosing_strategy="max_q_value"

XLA_PYTHON_CLIENT_PREALLOCATE=false env -u PYOPENGL_PLATFORM python ./train_pi.py --environment_name=libero --config=./configs/libero_config.py:pi0 --seed=0 --reward_bias=-1 --config.agent_kwargs.critic_network_kwargs.hidden_dims=256,256,256,256 --config.base_policy_path=pi0 --config.agent_kwargs.cql_alpha=0.005 --num_offline_epochs=1 --num_online_epochs=1 --config.agent_kwargs.distributional_critic_kwargs.q_min=-100 --config.agent_kwargs.critic_ensemble_size=2 --config.data_collection_particle_choosing_strategy="max_q_value" --config.evaluation_particle_choosing_strategy="max_q_value" --config.batch_size=8 --config.mixing_ratio=0.5

Pi0 FT related command
```
XLA_PYTHON_CLIENT_PREALLOCATE=false env -u PYOPENGL_PLATFORM python -m pdb ./train_pi.py --environment_name=libero --wandb_experiment_name=pi0_trial --config=./configs/libero_config.py:pi0 --num_offline_epochs=30 --num_online_epochs=0 --seed=0 --task_name="put both moka pots on the stove" --config.eval_interval=1

```

EXPO related command
```
XLA_PYTHON_CLIENT_PREALLOCATE=false env -u PYOPENGL_PLATFORM python -m pdb ./train_expo_pi.py --config=configs/libero_config.py:expo --task_name="put both moka pots on the stove" --num_online_epochs=30 --seed=0 --task_name="put both moka pots on the stove" --config.eval_interval=1 --config.batch_size=16 --config.agent_kwargs.batch_size=16 --environment_name=libero --wandb_experiment_name=expo_trial --num_edit_samples=4 --num_actions_to_sample=8 --num_offline_epochs=0 --final_step_sparse_reward=True --critic_warmup_steps 50 --online_trajectory_collection_frequency 50 --config.utd_ratio=8 --config.num_eval_episodes=1 --config.num_episodes_per_video=1


# Debug mode
XLA_PYTHON_CLIENT_PREALLOCATE=false env -u PYOPENGL_PLATFORM python -m pdb ./train_expo_pi.py --config=configs/libero_config.py:expo --task_name="put both moka pots on the stove" --num_online_epochs=10000 --seed=0 --task_name="put both moka pots on the stove" --config.eval_interval=1 --config.batch_size=32 --config.agent_kwargs.batch_size=32 --environment_name=libero --wandb_experiment_name=expo_trial --num_edit_samples=4 --num_actions_to_sample=4 --num_offline_epochs=0 --final_step_sparse_reward=False --critic_warmup_steps 0 --online_trajectory_collection_frequency 50 --config.utd_ratio=2 --config.num_eval_episodes=10 --config.num_episodes_per_video=2 --config.eval_interval=50 --reward_scale=1.0 --reward_bias=-0.1 --config.libero_tfrecord_regexp="/home/skowshik/vla/codebase/PolicyAgnosticRL/BACKUP_libero_10_pi05_put_the_two_mocha_pots_on_the_stove/results_expo/seed_0/image_replay_buffer/*.tfrecord" --critic_params_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/results_expo_debug-td_mc_v3/seed_0/checkpoint_200000.pkl" --reward_scale=1.0 --reward_bias=-0.1 --filter_successful_trajectories=False | tee logs/debug.log

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false env -u PYOPENGL_PLATFORM python -m pdb ./train_expo_pi_cache.py --config=configs/libero_config.py:expo --task_name="put both moka pots on the stove" --num_online_epochs=10000 --seed=0 --task_name="put both moka pots on the stove" --config.batch_size=64 --config.agent_kwargs.batch_size=64 --environment_name=libero --wandb_experiment_name=expo_trial --num_edit_samples=4 --num_actions_to_sample=4 --num_offline_epochs=0 --final_step_sparse_reward=False --critic_warmup_steps 0 --online_trajectory_collection_frequency 300 --config.utd_ratio=4 --config.num_eval_episodes=10 --config.num_episodes_per_video=5 --config.eval_interval=1000 --reward_scale=1.0 --reward_bias=-0.1 --config.libero_tfrecord_regexp="/home/skowshik/vla/codebase/PolicyAgnosticRL/CLEAN_traj30_v1/*.tfrecord" --reward_scale=1.0 --reward_bias=-0.1 --filter_successful_trajectories=False --critic_params_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/results_expo_debug-td-clean_v3/seed_0/checkpoint_100000.pkl" --config.save_dir="./results_expo_debug-expo_clean_v3" | tee logs/debug.log

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./train_expo_pi_cache.py --config=configs/libero_config.py:expo --task_name="put both moka pots on the stove" --num_online_epochs=10000 --seed=0 --task_name="put both moka pots on the stove" --config.batch_size=64 --config.agent_kwargs.batch_size=64 --environment_name=libero --wandb_experiment_name=expo_trial --num_edit_samples=4 --num_actions_to_sample=4 --num_offline_epochs=0 --final_step_sparse_reward=False --critic_warmup_steps 0 --online_trajectory_collection_frequency 300 --config.utd_ratio=4 --config.num_eval_episodes=10 --config.num_episodes_per_video=5 --config.eval_interval=1000 --reward_scale=1.0 --reward_bias=-0.1 --config.libero_tfrecord_regexp="/home/skowshik/vla/codebase/PolicyAgnosticRL/clean_skip_v2_server/*.tfrecord" --reward_scale=1.0 --reward_bias=-0.1 --filter_successful_trajectories=False --critic_params_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/results_expo_debug-td-clean_v3/seed_0/checkpoint_100000.pkl" --config.save_dir="./results_expo_debug-expo_clean_v3" | tee logs/debug.log

WANDB_ENTITY=shreyas-kowshik CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python ./train_expo_pi_cache.py --config=configs/libero_config.py:expo --task_name="put both moka pots on the stove" --num_online_epochs=1000000 --seed=0 --task_name="put both moka pots on the stove" --config.batch_size=256 --config.agent_kwargs.batch_size=256 --environment_name=libero --wandb_experiment_name=expo_clean_v8 --num_edit_samples=4 --num_actions_to_sample=4 --num_offline_epochs=0 --final_step_sparse_reward=False --critic_warmup_steps 0 --online_trajectory_collection_frequency 100 --config.utd_ratio=16 --config.num_eval_episodes=10 --config.num_episodes_per_video=5 --config.eval_interval=1000 --reward_scale=1.0 --reward_bias=-0.1 --config.libero_tfrecord_regexp="/home/skowshik/vla/codebase/PolicyAgnosticRL/clean_skip_v2_server/results_expo/image_replay_buffer/*.tfrecord" --reward_scale=1.0 --reward_bias=-0.1 --filter_successful_trajectories=False --critic_params_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/results_expo_debug-td-clean_v8/seed_0/checkpoint_4500.pkl" --config.save_dir="./results_expo_debug-expo_clean_v8" --num_parallel_envs 1 | tee logs/debug.log

### Remote machine Max Lab run
WANDB_ENTITY=shreyas-kowshik CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python ./train_expo_pi_cache.py --config=configs/libero_config.py:expo --task_name="put both moka pots on the stove" --num_online_epochs=1000000 --seed=0 --task_name="put both moka pots on the stove" --config.batch_size=256 --config.agent_kwargs.batch_size=256 --environment_name=libero --wandb_experiment_name=expo_trial --num_edit_samples=4 --num_actions_to_sample=4 --num_offline_epochs=0 --final_step_sparse_reward=False --critic_warmup_steps 0 --online_trajectory_collection_frequency 100 --config.utd_ratio=16 --config.num_eval_episodes=10 --config.num_episodes_per_video=5 --config.eval_interval=1000 --reward_scale=1.0 --reward_bias=-0.1 --config.libero_tfrecord_regexp="/home/skowshik/vla/codebase/PolicyAgnosticRL/clean_skip_v2_server/results_expo/image_replay_buffer/*.tfrecord" --reward_scale=1.0 --reward_bias=-0.1 --filter_successful_trajectories=False --critic_params_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/critic_ws/checkpoint_4500.pkl" --config.save_dir="./results_expo_debug-expo_clean_v8" --num_parallel_envs 1 | tee logs/debug.log

### Dump tf_record to visualize training dumps
python dump_tf_record.py --tfrecord_path "/home/skowshik/vla/codebase/PolicyAgnosticRL/libero_10_pi05_put_the_two_mocha_pots_on_the_stove/results_expo/image_replay_buffer/episode_0.tfrecord" --output_gif_path "/home/skowshik/vla/codebase/PolicyAgnosticRL/img.gif" 

```

# Sample base trajectories
```
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./sample_base_traj.py --config=configs/libero_config.py:expo --task_name="put both moka pots on the stove" --num_online_epochs=10000 --seed=0 --task_name="put both moka pots on the stove" --config.eval_interval=1 --config.batch_size=32 --config.agent_kwargs.batch_size=32 --environment_name=libero --wandb_experiment_name=expo_trial --num_edit_samples=4 --num_actions_to_sample=4 --num_offline_epochs=0 --final_step_sparse_reward=False --critic_warmup_steps 50 --online_trajectory_collection_frequency 50 --config.utd_ratio=2 --config.num_eval_episodes=10 --config.num_episodes_per_video=2 --config.eval_interval=50 --reward_scale=1.0 --reward_bias=-0.1 --reward_scale=1.0 --reward_bias=-0.1 --filter_successful_trajectories=False | tee logs/debug.log

conda install -c conda-forge mesalib
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=osmesa LIBGL_ALWAYS_SOFTWARE=1 MESA_GL_VERSION_OVERRIDE=3.3 PYOPENGL_PLATFORM=osmesa python -m pdb ./sample_base_traj.py --config=configs/libero_config.py:expo --task_name="put both moka pots on the stove" --num_online_epochs=10000 --seed=0 --task_name="put both moka pots on the stove" --config.eval_interval=1 --config.batch_size=32 --config.agent_kwargs.batch_size=32 --environment_name=libero --wandb_experiment_name=expo_trial --num_edit_samples=4 --num_actions_to_sample=4 --num_offline_epochs=0 --final_step_sparse_reward=False --critic_warmup_steps 50 --online_trajectory_collection_frequency 50 --config.utd_ratio=2 --config.num_eval_episodes=10 --config.num_episodes_per_video=2 --config.eval_interval=50 --reward_scale=1.0 --reward_bias=-0.1 --reward_scale=1.0 --reward_bias=-0.1 --filter_successful_trajectories=False | tee logs/debug.log

```


# Caching
```
CUDA_VISIBLE_DEVICES=0 python -m pdb dump_vlm_actions.py \
    --config=configs/libero_config.py:expo \
    --task_name="put both moka pots on the stove" \
    --output_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/clean_skip_v3_server.pkl" \
    --replay_buffer_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/clean_skip_v3_server/results_expo/image_replay_buffer/*.tfrecord" \
    --num_actions_to_sample=4 \
    --num_edit_samples=4 \
    --seed=0 \
    --config.agent_kwargs.batch_size=32 \
    --config.batch_size=32

```

# Warmstart training on cached data
```
WANDB_ENTITY=shreyas-kowshik CUDA_VISIBLE_DEVICES=0 python -m pdb train_expo_pi_critic_ws.py \
    --config=configs/libero_config.py:expo \
    --vlm_cache_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/pkl_files/clean_new_merged_skip_v1_v3_server.pkl" \
    --num_train_steps=100000 \
    --wandb_project_name="critic-warmstart" \
    --wandb_experiment_name="critic_cache_ws_debug-td-clean_v8" \
    --seed=0 \
    --config.agent_kwargs.batch_size=256 \
    --config.batch_size=256 \
    --config.eval_interval=500000 \
    --config.num_eval_episodes=10 \
    --config.num_episodes_per_video=5 \
    --reward_scale=1.0 \
    --reward_bias=-0.1 \
    --num_actions_to_sample=4 \
    --config.save_dir="./results_expo_debug-td-clean_v8" \
    --config.q_clip_low=-10000.0 \
    --config.q_clip_high=10000.0


```

