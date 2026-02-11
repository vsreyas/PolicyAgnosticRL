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

# Clean Training Setup for Residual Actor Learning on top of Pi0.5
## Sample Base Trajectories for warmup and future use
```
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./sample_base_traj.py --config=configs/libero_config.py:pi_residual_td3 --agent_name=pi_residual_td3 --seed=0 --environment_name=libero --reward_scale=1.0 --reward_bias=-0.1 --filter_successful_trajectories=False --config.save_dir="/data/user_data/skowshik/pi0_libero_base_actions" --num_trajectories_to_collect=40 --pi_config_name="pi0_libero_all_freeze" --num_diffusion_samples=1
```

```
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./sample_base_traj.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --seed=0 --environment_name=libero --reward_scale=1.0 --reward_bias=-0.1 --filter_successful_trajectories=False --config.save_dir="/data/hf_cache/models/pi05_libero_gradacc2_2k_base_trajectories_rng_fix" --num_trajectories_to_collect=300 --pi_config_name="pi05_libero_gradacc2_2k" --num_diffusion_samples=1
```

### Sample trajectories with gradq ascent
```
### Baseline
```
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./sample_base_traj.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --seed=0 --environment_name=libero --reward_scale=1.0 --reward_bias=-0.1 --filter_successful_trajectories=False --config.save_dir="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_Graq_ascent_BASELINE" --num_trajectories_to_collect=50 --pi_config_name="pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k" --num_diffusion_samples=1 --params_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_calql_10_2_clean_v2_balance/checkpoint_8000.pkl" --do_ascent=False
```

```
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./sample_base_traj.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --seed=0 --environment_name=libero --reward_scale=1.0 --reward_bias=-0.1 --filter_successful_trajectories=False --config.save_dir="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_Graq_ascent_calql_critic_4_2_8k_eta0.01_10steps" --num_trajectories_to_collect=50 --pi_config_name="pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k" --num_diffusion_samples=1 --params_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_calql_10_2_clean_v2_balance/checkpoint_8000.pkl" --do_ascent=True --eta_ascent=0.01 --num_ascent_steps=10
```

```
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./sample_base_traj.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --seed=0 --environment_name=libero --reward_scale=1.0 --reward_bias=-0.1 --filter_successful_trajectories=False --config.save_dir="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_Graq_ascent_calql_critic_4_2_8k_eta0.01_20steps" --num_trajectories_to_collect=50 --pi_config_name="pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k" --num_diffusion_samples=1 --params_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_calql_10_2_clean_v2_balance/checkpoint_8000.pkl" --do_ascent=True --eta_ascent=0.01 --num_ascent_steps=20
```

```
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./sample_base_traj.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --seed=0 --environment_name=libero --reward_scale=1.0 --reward_bias=-0.1 --filter_successful_trajectories=False --config.save_dir="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_Graq_ascent_calql_critic_4_2_8k_eta0.01_50steps" --num_trajectories_to_collect=50 --pi_config_name="pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k" --num_diffusion_samples=1 --params_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_calql_10_2_clean_v2_balance/checkpoint_8000.pkl" --do_ascent=True --eta_ascent=0.01 --num_ascent_steps=50
```

```
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./sample_base_traj.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --seed=0 --environment_name=libero --reward_scale=1.0 --reward_bias=-0.1 --filter_successful_trajectories=False --config.save_dir="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_Graq_ascent_calql_critic_4_2_8k_eta0.01_5steps" --num_trajectories_to_collect=50 --pi_config_name="pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k" --num_diffusion_samples=1 --params_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_calql_10_2_clean_v2_balance/checkpoint_8000.pkl" --do_ascent=True --eta_ascent=0.01 --num_ascent_steps=5
```

```
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./sample_base_traj.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --seed=0 --environment_name=libero --reward_scale=1.0 --reward_bias=-0.1 --filter_successful_trajectories=False --config.save_dir="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_Graq_ascent_calql_critic_4_2_8k_eta0.001_30steps" --num_trajectories_to_collect=50 --pi_config_name="pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k" --num_diffusion_samples=1 --params_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_calql_10_2_clean_v2_balance/checkpoint_8000.pkl" --do_ascent=True --eta_ascent=0.001 --num_ascent_steps=30
```

```
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./sample_base_traj.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --seed=0 --environment_name=libero --reward_scale=1.0 --reward_bias=-0.1 --filter_successful_trajectories=False --config.save_dir="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_Graq_ascent_calql_critic_4_2_8k_eta0.001_10steps" --num_trajectories_to_collect=50 --pi_config_name="pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k" --num_diffusion_samples=1 --params_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_calql_10_2_clean_v2_balance/checkpoint_8000.pkl" --do_ascent=True --eta_ascent=0.001 --num_ascent_steps=10
```

```
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./sample_base_traj.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --seed=0 --environment_name=libero --reward_scale=1.0 --reward_bias=-0.1 --filter_successful_trajectories=False --config.save_dir="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_Graq_ascent_calql_critic_4_2_8k_eta0.0001_50steps" --num_trajectories_to_collect=50 --pi_config_name="pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k" --num_diffusion_samples=1 --params_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_calql_10_2_clean_v2_balance/checkpoint_8000.pkl" --do_ascent=True --eta_ascent=0.0001 --num_ascent_steps=50
```

```
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./sample_base_traj.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --seed=0 --environment_name=libero --reward_scale=1.0 --reward_bias=-0.1 --filter_successful_trajectories=False --config.save_dir="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_Graq_ascent_calql_critic_4_2_8k_eta0.001_40steps" --num_trajectories_to_collect=50 --pi_config_name="pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k" --num_diffusion_samples=1 --params_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_calql_10_2_clean_v2_balance/checkpoint_8000.pkl" --do_ascent=True --eta_ascent=0.001 --num_ascent_steps=40
```


### SARSA Critic with best params above
```
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./sample_base_traj.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --seed=0 --environment_name=libero --reward_scale=1.0 --reward_bias=-0.1 --filter_successful_trajectories=False --config.save_dir="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_Graq_ascent_sarsa_critic_10_2_8k_eta0.001_30steps" --num_trajectories_to_collect=50 --pi_config_name="pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k" --num_diffusion_samples=1 --params_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance/checkpoint_8000.pkl" --do_ascent=True --eta_ascent=0.001 --num_ascent_steps=30
```

```
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./sample_base_traj.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --seed=0 --environment_name=libero --reward_scale=1.0 --reward_bias=-0.1 --filter_successful_trajectories=False --config.save_dir="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_Graq_ascent_sarsa_critic_10_2_8k_eta0.001_30steps-adam_noclip" --num_trajectories_to_collect=50 --pi_config_name="pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k" --num_diffusion_samples=1 --params_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance/checkpoint_8000.pkl" --do_ascent=True --eta_ascent=0.001 --num_ascent_steps=30 --action_optimizer=adam --adam_beta1=0.9 --adam_beta2=0.999
```

### Sarsa Q SGD ascent with mean()
```
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./sample_base_traj.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --seed=0 --environment_name=libero --reward_scale=1.0 --reward_bias=-0.1 --filter_successful_trajectories=False --config.save_dir="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_Graq_ascent_sarsa_critic_10_2_8k_eta0.001_30steps_gradq_mean" --num_trajectories_to_collect=50 --pi_config_name="pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k" --num_diffusion_samples=1 --params_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance/checkpoint_8000.pkl" --do_ascent=True --eta_ascent=0.001 --num_ascent_steps=30
```

##### BoN version with gradq ascent
```
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./sample_base_traj.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --seed=0 --environment_name=libero --reward_scale=1.0 --reward_bias=-0.1 --filter_successful_trajectories=False --config.save_dir="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_Graq_ascent_sarsa_critic_10_2_8k_eta0.001_30steps_gradq_mean_bon16" --num_trajectories_to_collect=50 --pi_config_name="pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k" --num_diffusion_samples=1 --params_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance/checkpoint_8000.pkl" --do_ascent=True --eta_ascent=0.001 --num_ascent_steps=30 --action_optimizer="gradient_ascent" --use_bon=True --bon_actions=16
```

### BoN Baseline
```
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./sample_base_traj.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --seed=0 --environment_name=libero --reward_scale=1.0 --reward_bias=-0.1 --filter_successful_trajectories=False --config.save_dir="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_Graq_ascent_sarsa_critic_10_2_8k_bon16" --num_trajectories_to_collect=50 --pi_config_name="pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k" --num_diffusion_samples=1 --params_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance/checkpoint_8000.pkl" --do_ascent=True --eta_ascent=0.001 --num_ascent_steps=30 --action_optimizer="gradient_ascent" --use_bon_no_gradq=True --bon_actions=16
```


### Sarsa Q RMSProp ascent with mean()
```
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./sample_base_traj.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --seed=0 --environment_name=libero --reward_scale=1.0 --reward_bias=-0.1 --filter_successful_trajectories=False --config.save_dir="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_Graq_ascent_sarsa_critic_10_2_8k_eta0.01_10steps_rmsprop_mean" --num_trajectories_to_collect=50 --pi_config_name="pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k" --num_diffusion_samples=1 --params_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance/checkpoint_8000.pkl" --do_ascent=True --eta_ascent=0.01 --num_ascent_steps=10 --action_optimizer=rmsprop --rmsprop_beta=0.99
```

```
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./sample_base_traj.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --seed=0 --environment_name=libero --reward_scale=1.0 --reward_bias=-0.1 --filter_successful_trajectories=False --config.save_dir="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_Graq_ascent_sarsa_critic_10_2_8k_eta0.001_20steps_adam_mean_0.98" --num_trajectories_to_collect=50 --pi_config_name="pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k" --num_diffusion_samples=1 --params_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance/checkpoint_8000.pkl" --do_ascent=True --eta_ascent=0.001 --num_ascent_steps=20 --action_optimizer=adam --adam_beta1=0.9 --adam_beta2=0.98

python dump_tf_record.py --tfrecord_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_Graq_ascent_sarsa_critic_10_2_8k_eta0.001_30steps-adam_noclip/image_replay_buffer/episode_0.tfrecord" --output_gif_path "./ep0_gradq_debug_noclip_0.001.mp4"
```

### After training critic using gradq ascent policy
```
```
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./sample_base_traj.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --seed=0 --environment_name=libero --reward_scale=1.0 --reward_bias=-0.1 --filter_successful_trajectories=False --config.save_dir="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_Graq_ascent_sarsa_critic_online_train_v3_1.2k_eta0.001_30steps_gradq_mean" --num_trajectories_to_collect=50 --pi_config_name="pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k" --num_diffusion_samples=1 --params_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-online_rl_v3/checkpoint_1200.pkl" --do_ascent=True --eta_ascent=0.00001 --num_ascent_steps=30

python find_traj_stats.py --tf_record_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_Graq_ascent_sarsa_critic_10_2_8k_eta0.001_30steps_noclip/image_replay_buffer/*.tfrecord"
```

```
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./sample_base_traj.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --seed=0 --environment_name=libero --reward_scale=1.0 --reward_bias=-0.1 --filter_successful_trajectories=False --config.save_dir="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_Graq_ascent_sarsa_critic_online_train_v4_600_eta0.001_30steps_gradq_mean" --num_trajectories_to_collect=50 --pi_config_name="pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k" --num_diffusion_samples=1 --params_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-online_rl_v4/checkpoint_600.pkl" --do_ascent=True --eta_ascent=0.001 --num_ascent_steps=30

python find_traj_stats.py --tf_record_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_Graq_ascent_sarsa_critic_online_train_v2_30k_eta0.001_30steps_gradq_mean/image_replay_buffer/*.tfrecord"
```

```
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./sample_base_traj.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --seed=0 --environment_name=libero --reward_scale=1.0 --reward_bias=-0.1 --filter_successful_trajectories=False --config.save_dir="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_Graq_ascent_sarsa_critic_online_train_v4_200_eta0.0001_30steps_gradq_mean" --num_trajectories_to_collect=50 --pi_config_name="pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k" --num_diffusion_samples=1 --params_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-online_rl_v4/checkpoint_200.pkl" --do_ascent=True --eta_ascent=0.0001 --num_ascent_steps=30
```

```
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./sample_base_traj.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --seed=0 --environment_name=libero --reward_scale=1.0 --reward_bias=-0.1 --filter_successful_trajectories=False --config.save_dir="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_Graq_ascent_sarsa_critic_online_train_v4_200_eta0.0001_10steps_gradq_mean" --num_trajectories_to_collect=50 --pi_config_name="pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k" --num_diffusion_samples=1 --params_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-online_rl_v4/checkpoint_200.pkl" --do_ascent=True --eta_ascent=0.0001 --num_ascent_steps=10
```

### BoN=16 baseline
```
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./sample_base_traj.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --seed=0 --environment_name=libero --reward_scale=1.0 --reward_bias=-0.1 --filter_successful_trajectories=False --config.save_dir="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_Graq_ascent_sarsa_critic_online_train_v4_200_bon16" --num_trajectories_to_collect=50 --pi_config_name="pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k" --num_diffusion_samples=1 --params_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-online_rl_v4/checkpoint_200.pkl" --do_ascent=True --eta_ascent=0.0001 --num_ascent_steps=10 --action_optimizer="gradient_ascent" --use_bon_no_gradq=True --bon_actions=16
```


### After continuing critic training analysis
```
/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-online_rl_v6/checkpoint_1000.pkl

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./sample_base_traj.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --seed=0 --environment_name=libero --reward_scale=1.0 --reward_bias=-0.1 --filter_successful_trajectories=False --config.save_dir="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-online_rl_v7-ch9k-INFERENCE-0.001-80-grip_debug" --num_trajectories_to_collect=10 --pi_config_name="pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k" --num_diffusion_samples=1 --params_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-online_rl_v7/checkpoint_9000.pkl" --do_ascent=True --eta_ascent=0.001 --num_ascent_steps=80

python find_traj_stats.py --tf_record_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-online_rl_v6-ch6k-INFERENCE-0.001-30/image_replay_buffer/*.tfrecord"

rm -r /data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-online_rl_v7-ch9k-INFERENCE-0.001-80-grip_debug/

python dump_tf_record.py --tfrecord_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-online_rl_v7-ch9k-INFERENCE-0.001-80-grip_debug/image_replay_buffer/episode_0.tfrecord" --output_gif_path "./debug_v7_80_grip_debug0.mp4"
```

### Minimal experiment gradq online training samples
```
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./sample_base_traj.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --seed=0 --environment_name=libero --reward_scale=1.0 --reward_bias=-0.1 --filter_successful_trajectories=False --config.save_dir="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-minimal_intuition-1k_online_steps-ch1k-INFERENCE-0.001-20" --num_trajectories_to_collect=50 --pi_config_name="pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k" --num_diffusion_samples=1 --params_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-minimal_intuition-1k_online_steps/checkpoint_1000.pkl" --do_ascent=True --eta_ascent=0.001 --num_ascent_steps=20

python dump_tf_record.py --tfrecord_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-minimal_intuition-1k_online_steps-ch1k-INFERENCE-0.001-50/image_replay_buffer/episode_0.tfrecord" --output_gif_path "./debug_1k_online_gradq50_ep0.mp4"

python dump_tf_record.py --tfrecord_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-minimal_intuition-1k_online_steps-ch1k-INFERENCE-0.001-50/image_replay_buffer/episode_5.tfrecord" --output_gif_path "./debug_1k_online_gradq50_ep5.mp4"

python dump_tf_record.py --tfrecord_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-minimal_intuition-1k_online_steps-ch1k-INFERENCE-0.001-50/image_replay_buffer/episode_15.tfrecord" --output_gif_path "./debug_1k_online_gradq50_ep15.mp4"

python dump_tf_record.py --tfrecord_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-minimal_intuition-1k_online_steps-ch1k-INFERENCE-0.001-50/image_replay_buffer/episode_25.tfrecord" --output_gif_path "./debug_1k_online_gradq50_ep25.mp4"

python dump_tf_record.py --tfrecord_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-minimal_intuition-1k_online_steps-ch1k-INFERENCE-0.001-50/image_replay_buffer/episode_6.tfrecord" --output_gif_path "./debug_1k_online_gradq50_ep6.mp4"

python dump_tf_record.py --tfrecord_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-minimal_intuition-1k_online_steps-ch1k-INFERENCE-0.001-50/image_replay_buffer/episode_9.tfrecord" --output_gif_path "./debug_1k_online_gradq50_ep9.mp4"

python dump_tf_record.py --tfrecord_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-minimal_intuition-1k_online_steps-ch1k-INFERENCE-0.001-50_zero_gripper/image_replay_buffer/episode_0.tfrecord" --output_gif_path "./debug_1k_online_gradq50_zero_grip_ep0.mp4"

python dump_tf_record.py --tfrecord_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-minimal_intuition-1k_online_steps-ch1k-INFERENCE-0.0-30_adam/image_replay_buffer/episode_0.tfrecord" --output_gif_path "./debug_1k_online_gradq0.0_30_adam_ep0.mp4"

python find_traj_stats.py --tf_record_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-minimal_intuition-1k_online_steps-ch1k-INFERENCE-0.001-30_adam/image_replay_buffer/*.tfrecord"
```

```
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./sample_base_traj.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --seed=0 --environment_name=libero --reward_scale=1.0 --reward_bias=-0.1 --filter_successful_trajectories=False --config.save_dir="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-minimal_intuition-1k_online_steps-ch1k-INFERENCE-0.001-35" --num_trajectories_to_collect=50 --pi_config_name="pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k" --num_diffusion_samples=1 --params_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-minimal_intuition-1k_online_steps/checkpoint_1000.pkl" --do_ascent=True --eta_ascent=0.001 --num_ascent_steps=35

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./sample_base_traj.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --seed=0 --environment_name=libero --reward_scale=1.0 --reward_bias=-0.1 --filter_successful_trajectories=False --config.save_dir="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-minimal_intuition-1k_online_steps-ch1k-INFERENCE-0.001-40_zero_gripper" --num_trajectories_to_collect=50 --pi_config_name="pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k" --num_diffusion_samples=1 --params_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-minimal_intuition-1k_online_steps/checkpoint_1000.pkl" --do_ascent=True --eta_ascent=0.001 --num_ascent_steps=40 --zero_grad_gripper=True

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./sample_base_traj.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --seed=0 --environment_name=libero --reward_scale=1.0 --reward_bias=-0.1 --filter_successful_trajectories=False --config.save_dir="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-minimal_intuition-1k_online_steps-ch1k-INFERENCE-0.001-50_zero_gripper" --num_trajectories_to_collect=50 --pi_config_name="pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k" --num_diffusion_samples=1 --params_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-minimal_intuition-1k_online_steps/checkpoint_1000.pkl" --do_ascent=True --eta_ascent=0.001 --num_ascent_steps=50 --zero_grad_gripper=True

### Just try adam
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./sample_base_traj.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --seed=0 --environment_name=libero --reward_scale=1.0 --reward_bias=-0.1 --filter_successful_trajectories=False --config.save_dir="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-minimal_intuition-1k_online_steps-ch1k-INFERENCE-0.0=00001-30_adam" --num_trajectories_to_collect=50 --pi_config_name="pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k" --num_diffusion_samples=1 --params_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-minimal_intuition-1k_online_steps/checkpoint_1000.pkl" --do_ascent=True --eta_ascent=0.00001 --num_ascent_steps=5 --action_optimizer=adam --adam_beta1=0.9 --adam_beta2=0.999
```

### Case of critic balance
### HERE ###
```
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./sample_base_traj.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --seed=0 --environment_name=libero --reward_scale=1.0 --reward_bias=-0.1 --filter_successful_trajectories=False --config.save_dir="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-minimal_intuition-1k_online_steps-critic_balance-0.001_50-bon16-bon_v2" --num_trajectories_to_collect=50 --pi_config_name="pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k" --num_diffusion_samples=1 --params_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-minimal_intuition-1k_online_steps-critic_balance/checkpoint_1000.pkl" --do_ascent=True --eta_ascent=0.001 --num_ascent_steps=50 --action_optimizer="gradient_ascent" --use_bon_gradq_v2=True --bon_actions=16

python dump_tf_record.py --tfrecord_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-minimal_intuition-1k_online_steps-ch1k-INFERENCE-0.0=00001-30_adam/image_replay_buffer/episode_0.tfrecord" --output_gif_path "./debug_1k_online_adam_0.00001_30_ep0.mp4"

python find_traj_stats.py --tf_record_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-minimal_intuition-1k_online_steps-critic_balance-0.001_30-bon16-bon_v2/image_replay_buffer/*.tfrecord"

python find_traj_stats.py --tf_record_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-minimal_intuition-1k_online_steps-critic_balance-0.001_50-bon16-bon_v2/image_replay_buffer/*.tfrecord"
```
```


### Debug run to just see residual norm ranges
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./sample_base_traj.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --seed=0 --environment_name=libero --reward_scale=1.0 --reward_bias=-0.1 --filter_successful_trajectories=False --config.save_dir="/data/hf_cache/models/DEBUG" --num_trajectories_to_collect=50 --pi_config_name="pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k" --num_diffusion_samples=1 --params_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance/checkpoint_8000.pkl" --do_ascent=True --eta_ascent=0.001 --num_ascent_steps=30 --config.agent_kwargs.num_qs=10 --config.agent_kwargs.num_min_qs=2
```

## Find trajectory statistics
```
python find_traj_stats.py --tf_record_path="/data/user_data/skowshik/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_base_rollouts_EVAL/image_replay_buffer/*.tfrecord"

python find_traj_stats.py --tf_record_path="/data/user_data/skowshik/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_base_rollouts_EVAL/image_replay_buffer/*.tfrecord"
```

```
```
python find_traj_stats.py --tf_record_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_Graq_ascent_BASELINE/image_replay_buffer/*.tfrecord"

python find_traj_stats.py --tf_record_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_Graq_ascent_calql_critic_4_2_8k_eta0.01_5steps/image_replay_buffer/*.tfrecord"

python find_traj_stats.py --tf_record_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_Graq_ascent_calql_critic_4_2_8k_eta0.01_10steps/image_replay_buffer/*.tfrecord"

python find_traj_stats.py --tf_record_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_Graq_ascent_calql_critic_4_2_8k_eta0.01_20steps/image_replay_buffer/*.tfrecord"

python find_traj_stats.py --tf_record_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_Graq_ascent_calql_critic_4_2_8k_eta0.01_50steps/image_replay_buffer/*.tfrecord"

python find_traj_stats.py --tf_record_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_Graq_ascent_calql_critic_4_2_8k_eta0.001_30steps/image_replay_buffer/*.tfrecord"

python find_traj_stats.py --tf_record_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_Graq_ascent_calql_critic_4_2_8k_eta0.001_10steps/image_replay_buffer/*.tfrecord"

python find_traj_stats.py --tf_record_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_Graq_ascent_calql_critic_4_2_8k_eta0.0001_50steps/image_replay_buffer/*.tfrecord"

python find_traj_stats.py --tf_record_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_Graq_ascent_calql_critic_4_2_8k_eta0.001_40steps/image_replay_buffer/*.tfrecord"

python find_traj_stats.py --tf_record_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_Graq_ascent_sarsa_critic_10_2_8k_eta0.001_30steps/image_replay_buffer/*.tfrecord"

python find_traj_stats.py --tf_record_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_Graq_ascent_sarsa_critic_10_2_8k_eta0.001_30steps_gradq_mean/image_replay_buffer/*.tfrecord"

python find_traj_stats.py --tf_record_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_Graq_ascent_sarsa_critic_10_2_8k_eta0.001_50steps_rmsprop_mean/image_replay_buffer/*.tfrecord"

python find_traj_stats.py --tf_record_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_Graq_ascent_sarsa_critic_10_2_8k_eta0.001_30steps_rmsprop_mean/image_replay_buffer/*.tfrecord"

python find_traj_stats.py --tf_record_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_Graq_ascent_sarsa_critic_10_2_8k_eta0.00001_50steps_rmsprop_mean/image_replay_buffer/*.tfrecord"

python find_traj_stats.py --tf_record_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_Graq_ascent_sarsa_critic_10_2_8k_eta0.001_10steps_adam_mean/image_replay_buffer/*.tfrecord"

python find_traj_stats.py --tf_record_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_Graq_ascent_sarsa_critic_10_2_8k_eta0.0001_20steps_adam_mean/image_replay_buffer/*.tfrecord"

python find_traj_stats.py --tf_record_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_Graq_ascent_sarsa_critic_10_2_8k_eta0.001_30steps_gradq_mean_bon8/image_replay_buffer/*.tfrecord"

python find_traj_stats.py --tf_record_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_Graq_ascent_sarsa_critic_10_2_8k_eta0.001_30steps_gradq_mean_bon16/image_replay_buffer/*.tfrecord"

python find_traj_stats.py --tf_record_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_Graq_ascent_sarsa_critic_10_2_8k_bon16/image_replay_buffer/*.tfrecord"

python find_traj_stats.py --tf_record_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_Graq_ascent_sarsa_critic_online_train_v4_200_bon16/image_replay_buffer/*.tfrecord"
```
```

### Policy extraction using TD3 + SARSA
```
```
rm -r /data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-residual_sum-0.3-clean_v4-residual_warmup && WANDB_ENTITY=shreyas-kowshik CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./train_pi_residual.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --pi_config_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k --seed=0 --task_name="put both moka pots on the stove" --config.batch_size=256 --environment_name=libero --wandb_experiment_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-residual_sum-0.3-clean_v4-residual_warmup --final_step_sparse_reward=False --online_trajectory_collection_frequency 1000 --num_trajectories_to_collect=2 --config.utd_ratio=1 --config.num_eval_episodes=10 --config.num_episodes_per_video=5 --config.eval_interval=1000 --reward_scale=1.0 --reward_bias=-0.1 --config.libero_tfrecord_regexp="/data/user_data/skowshik/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_base_rollouts/image_replay_buffer/*.tfrecord" --filter_successful_trajectories=False --config.save_dir="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-residual_sum-0.3-clean_v4-residual_warmup" --scale_success_alpha=2.0 --intermediate_reward_mul_factor 10.0 --warmup_steps 5000 --num_train_steps 5000000 --balance_offline_training_data=True --config.save_interval=1000 --critic_warmup_type=sarsa --config.image_replay_buffer_kwargs.load_action_samples=False --config.offline_image_replay_buffer_kwargs.load_action_samples=False --ws_critic=False --ws_edit_actor=True --params_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance/checkpoint_8000.pkl" --num_diffusion_samples=1 --edit_actor_utd_ratio=1 --exploration_epsilon=0.000005 --online_critic_update_type="sarsa" --on_policy=True --config.agent_kwargs.edit_action_scale=0.3 --ws_critic=False --ws_edit_actor=True --config.agent_kwargs.actor_lr=0.00001

cp -r /data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-residual_sum-0.3-clean_v2/seed_0/eval_gifs/step_999/ .
```

```
rm -r /data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-residual_sum-0.3-ws_hparam_tuning_v1

WANDB_ENTITY=shreyas-kowshik CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./train_pi_residual.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --pi_config_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k --seed=0 --task_name="put both moka pots on the stove" --config.batch_size=256 --environment_name=libero --wandb_experiment_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-residual_sum-0.3-slow_train_td3_v1 --final_step_sparse_reward=False --online_trajectory_collection_frequency 10 --num_trajectories_to_collect=5 --config.utd_ratio=1 --config.num_eval_episodes=10 --config.num_episodes_per_video=5 --config.eval_interval=1000 --reward_scale=1.0 --reward_bias=-0.1 --config.libero_tfrecord_regexp="/data/user_data/skowshik/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_base_rollouts/image_replay_buffer/*.tfrecord" --filter_successful_trajectories=False --config.save_dir="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-residual_sum-0.3-slow_train_td3_v1" --scale_success_alpha=2.0 --intermediate_reward_mul_factor 10.0 --warmup_steps 50 --num_train_steps 5000000 --balance_offline_training_data=True --config.save_interval=1000 --critic_warmup_type=sarsa --config.image_replay_buffer_kwargs.load_action_samples=False --config.offline_image_replay_buffer_kwargs.load_action_samples=False --params_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance/checkpoint_8000.pkl" --num_diffusion_samples=1 --edit_actor_utd_ratio=30 --exploration_epsilon=0.000005 --online_critic_update_type="sarsa" --on_policy=True --config.agent_kwargs.edit_action_scale=0.3 --ws_critic=False --ws_edit_actor=True
```

```
WANDB_ENTITY=shreyas-kowshik CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./train_pi_residual.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --pi_config_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k --seed=0 --task_name="put both moka pots on the stove" --config.batch_size=256 --environment_name=libero --wandb_experiment_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-residual_sum-0.3-slow_train_td3_v2 --final_step_sparse_reward=False --online_trajectory_collection_frequency 20 --num_trajectories_to_collect=5 --config.utd_ratio=1 --config.num_eval_episodes=10 --config.num_episodes_per_video=5 --config.eval_interval=1000 --reward_scale=1.0 --reward_bias=-0.1 --config.libero_tfrecord_regexp="/data/user_data/skowshik/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_base_rollouts/image_replay_buffer/*.tfrecord" --filter_successful_trajectories=False --config.save_dir="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-residual_sum-0.3-slow_train_td3_v2" --scale_success_alpha=2.0 --intermediate_reward_mul_factor 10.0 --warmup_steps 20 --num_train_steps 5000000 --balance_offline_training_data=True --config.save_interval=1000 --critic_warmup_type=sarsa --config.image_replay_buffer_kwargs.load_action_samples=False --config.offline_image_replay_buffer_kwargs.load_action_samples=False --params_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance/checkpoint_8000.pkl" --num_diffusion_samples=1 --edit_actor_utd_ratio=12 --exploration_epsilon=0.000005 --online_critic_update_type="sarsa" --on_policy=True --config.agent_kwargs.edit_action_scale=0.3 --ws_critic=False --ws_edit_actor=True
```
```

### GradQ Critic Train Continue
```
```
rm -r /data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-online_rl_v1

WANDB_ENTITY=shreyas-kowshik CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./train_pi_residual.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --pi_config_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k --seed=0 --task_name="put both moka pots on the stove" --config.batch_size=256 --environment_name=libero --wandb_experiment_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-online_rl_v1 --final_step_sparse_reward=False --online_trajectory_collection_frequency 100 --num_trajectories_to_collect=5 --config.utd_ratio=1 --config.num_eval_episodes=10 --config.num_episodes_per_video=5 --config.eval_interval=1000 --reward_scale=1.0 --reward_bias=-0.1 --config.libero_tfrecord_regexp="/data/user_data/skowshik/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_base_rollouts/image_replay_buffer/*.tfrecord" --filter_successful_trajectories=False --config.save_dir="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-online_rl_v1" --scale_success_alpha=2.0 --intermediate_reward_mul_factor 10.0 --warmup_steps 0 --num_train_steps 5000000 --balance_offline_training_data=True --config.save_interval=1000 --critic_warmup_type=sarsa --config.image_replay_buffer_kwargs.load_action_samples=False --config.offline_image_replay_buffer_kwargs.load_action_samples=False --params_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance/checkpoint_8000.pkl" --num_diffusion_samples=1 --edit_actor_utd_ratio=1 --exploration_epsilon=0.000005 --online_critic_update_type="sarsa" --on_policy=True --config.agent_kwargs.edit_action_scale=0.3 --ws_critic=False --ws_edit_actor=False --do_ascent=True --eta_ascent=0.001 --num_ascent_steps=30
```

```
rm -r /data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-online_rl_v2

WANDB_ENTITY=shreyas-kowshik CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./train_pi_residual.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --pi_config_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k --seed=0 --task_name="put both moka pots on the stove" --config.batch_size=256 --environment_name=libero --wandb_experiment_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-online_rl_v2 --final_step_sparse_reward=False --online_trajectory_collection_frequency 500 --num_trajectories_to_collect=5 --config.utd_ratio=1 --config.num_eval_episodes=10 --config.num_episodes_per_video=5 --config.eval_interval=1000 --reward_scale=1.0 --reward_bias=-0.1 --config.libero_tfrecord_regexp="/data/user_data/skowshik/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_base_rollouts/image_replay_buffer/*.tfrecord" --filter_successful_trajectories=False --config.save_dir="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-online_rl_v2" --scale_success_alpha=2.0 --intermediate_reward_mul_factor 10.0 --warmup_steps 0 --num_train_steps 5000000 --balance_offline_training_data=True --config.save_interval=1000 --critic_warmup_type=sarsa --config.image_replay_buffer_kwargs.load_action_samples=False --config.offline_image_replay_buffer_kwargs.load_action_samples=False --params_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance/checkpoint_8000.pkl" --num_diffusion_samples=1 --edit_actor_utd_ratio=1 --exploration_epsilon=0.000005 --online_critic_update_type="sarsa" --on_policy=True --config.agent_kwargs.edit_action_scale=0.3 --ws_critic=False --ws_edit_actor=False --do_ascent=True --eta_ascent=0.001 --num_ascent_steps=30
```

```
rm -r /data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-online_rl_v3

WANDB_ENTITY=shreyas-kowshik CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./train_pi_residual.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --pi_config_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k --seed=0 --task_name="put both moka pots on the stove" --config.batch_size=256 --environment_name=libero --wandb_experiment_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-online_rl_v3 --final_step_sparse_reward=False --online_trajectory_collection_frequency 20 --num_trajectories_to_collect=5 --config.utd_ratio=1 --config.num_eval_episodes=10 --config.num_episodes_per_video=5 --config.eval_interval=200 --reward_scale=1.0 --reward_bias=-0.1 --config.libero_tfrecord_regexp="/data/user_data/skowshik/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_base_rollouts/image_replay_buffer/*.tfrecord" --filter_successful_trajectories=False --config.save_dir="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-online_rl_v3" --scale_success_alpha=2.0 --intermediate_reward_mul_factor 10.0 --warmup_steps 0 --num_train_steps 5000000 --balance_offline_training_data=True --config.save_interval=200 --critic_warmup_type=sarsa --config.image_replay_buffer_kwargs.load_action_samples=False --config.offline_image_replay_buffer_kwargs.load_action_samples=False --params_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance/checkpoint_8000.pkl" --num_diffusion_samples=1 --edit_actor_utd_ratio=1 --exploration_epsilon=0.000005 --online_critic_update_type="sarsa" --on_policy=True --config.agent_kwargs.edit_action_scale=0.3 --ws_critic=False --ws_edit_actor=False --do_ascent=True --eta_ascent=0.001 --num_ascent_steps=30
```

```
rm -r /data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-online_rl_v4

WANDB_ENTITY=shreyas-kowshik CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./train_pi_residual.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --pi_config_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k --seed=0 --task_name="put both moka pots on the stove" --config.batch_size=256 --environment_name=libero --wandb_experiment_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-online_rl_v4 --final_step_sparse_reward=False --online_trajectory_collection_frequency 10 --num_trajectories_to_collect=5 --config.utd_ratio=1 --config.num_eval_episodes=10 --config.num_episodes_per_video=5 --config.eval_interval=1000 --reward_scale=1.0 --reward_bias=-0.1 --config.libero_tfrecord_regexp="/data/user_data/skowshik/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_base_rollouts/image_replay_buffer/*.tfrecord" --filter_successful_trajectories=False --config.save_dir="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-online_rl_v4" --scale_success_alpha=2.0 --intermediate_reward_mul_factor 10.0 --warmup_steps 0 --num_train_steps 5000000 --balance_offline_training_data=True --config.save_interval=200 --critic_warmup_type=sarsa --config.image_replay_buffer_kwargs.load_action_samples=False --config.offline_image_replay_buffer_kwargs.load_action_samples=False --params_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance/checkpoint_8000.pkl" --num_diffusion_samples=1 --edit_actor_utd_ratio=1 --exploration_epsilon=0.000005 --online_critic_update_type="sarsa" --on_policy=True --config.agent_kwargs.edit_action_scale=0.3 --ws_critic=False --ws_edit_actor=False --do_ascent=True --eta_ascent=0.001 --num_ascent_steps=30
```

```
rm -r /data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-online_rl_v5

WANDB_ENTITY=shreyas-kowshik CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./train_pi_residual.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --pi_config_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k --seed=0 --task_name="put both moka pots on the stove" --config.batch_size=256 --environment_name=libero --wandb_experiment_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-online_rl_v5 --final_step_sparse_reward=False --online_trajectory_collection_frequency 5000 --num_trajectories_to_collect=15 --config.utd_ratio=1 --config.num_eval_episodes=10 --config.num_episodes_per_video=5 --config.eval_interval=5000 --reward_scale=1.0 --reward_bias=-0.1 --config.libero_tfrecord_regexp="/data/user_data/skowshik/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_base_rollouts/image_replay_buffer/*.tfrecord" --filter_successful_trajectories=False --config.save_dir="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-online_rl_v5" --scale_success_alpha=2.0 --intermediate_reward_mul_factor 10.0 --warmup_steps 0 --num_train_steps 5000000 --balance_offline_training_data=False --config.save_interval=200 --critic_warmup_type=sarsa --config.image_replay_buffer_kwargs.load_action_samples=False --config.offline_image_replay_buffer_kwargs.load_action_samples=False --params_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance/checkpoint_8000.pkl" --num_diffusion_samples=1 --edit_actor_utd_ratio=1 --exploration_epsilon=0.000005 --online_critic_update_type="sarsa" --on_policy=True --config.agent_kwargs.edit_action_scale=0.3 --ws_critic=False --ws_edit_actor=False --do_ascent=True --eta_ascent=0.001 --num_ascent_steps=30
```

```
WANDB_ENTITY=shreyas-kowshik CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./train_pi_residual.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --pi_config_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k --seed=0 --task_name="put both moka pots on the stove" --config.batch_size=256 --environment_name=libero --wandb_experiment_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-online_rl_v6 --final_step_sparse_reward=False --online_trajectory_collection_frequency 1000 --num_trajectories_to_collect=15 --config.utd_ratio=1 --config.num_eval_episodes=10 --config.num_episodes_per_video=5 --config.eval_interval=5000 --reward_scale=1.0 --reward_bias=-0.1 --config.libero_tfrecord_regexp="/data/user_data/skowshik/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_base_rollouts/image_replay_buffer/*.tfrecord" --filter_successful_trajectories=False --config.save_dir="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-online_rl_v6" --scale_success_alpha=2.0 --intermediate_reward_mul_factor 10.0 --warmup_steps 0 --num_train_steps 5000000 --balance_offline_training_data=False --config.save_interval=200 --critic_warmup_type=sarsa --config.image_replay_buffer_kwargs.load_action_samples=False --config.offline_image_replay_buffer_kwargs.load_action_samples=False --params_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance/checkpoint_8000.pkl" --num_diffusion_samples=1 --edit_actor_utd_ratio=1 --exploration_epsilon=0.000005 --online_critic_update_type="sarsa" --on_policy=True --config.agent_kwargs.edit_action_scale=0.3 --ws_critic=False --ws_edit_actor=False --do_ascent=True --eta_ascent=0.001 --num_ascent_steps=30
```

```
WANDB_ENTITY=shreyas-kowshik CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./train_pi_residual.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --pi_config_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k --seed=0 --task_name="put both moka pots on the stove" --config.batch_size=256 --environment_name=libero --wandb_experiment_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-online_rl_v7 --final_step_sparse_reward=False --online_trajectory_collection_frequency 1000 --num_trajectories_to_collect=15 --config.utd_ratio=1 --config.num_eval_episodes=10 --config.num_episodes_per_video=5 --config.eval_interval=5000 --reward_scale=1.0 --reward_bias=-0.1 --config.libero_tfrecord_regexp="/data/user_data/skowshik/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_base_rollouts/image_replay_buffer/*.tfrecord" --filter_successful_trajectories=False --config.save_dir="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-online_rl_v7" --scale_success_alpha=2.0 --intermediate_reward_mul_factor 10.0 --warmup_steps 0 --num_train_steps 5000000 --balance_offline_training_data=False --config.save_interval=200 --critic_warmup_type=sarsa --config.image_replay_buffer_kwargs.load_action_samples=False --config.offline_image_replay_buffer_kwargs.load_action_samples=False --params_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance/checkpoint_8000.pkl" --num_diffusion_samples=1 --edit_actor_utd_ratio=1 --exploration_epsilon=0.000005 --online_critic_update_type="sarsa" --on_policy=True --config.agent_kwargs.edit_action_scale=0.3 --ws_critic=False --ws_edit_actor=False --do_ascent=True --eta_ascent=0.001 --num_ascent_steps=30 --critic_balance_classes=True
```

```
WANDB_ENTITY=shreyas-kowshik CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./train_pi_residual.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --pi_config_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k --seed=0 --task_name="put both moka pots on the stove" --config.batch_size=256 --environment_name=libero --wandb_experiment_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-online_rl_v8 --final_step_sparse_reward=False --online_trajectory_collection_frequency 5000 --num_trajectories_to_collect=15 --config.utd_ratio=1 --config.num_eval_episodes=10 --config.num_episodes_per_video=5 --config.eval_interval=5000 --reward_scale=1.0 --reward_bias=-0.1 --config.libero_tfrecord_regexp="/data/user_data/skowshik/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_base_rollouts/image_replay_buffer/*.tfrecord" --filter_successful_trajectories=False --config.save_dir="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-online_rl_v8" --scale_success_alpha=2.0 --intermediate_reward_mul_factor 10.0 --warmup_steps 0 --num_train_steps 5000000 --balance_offline_training_data=False --config.save_interval=200 --critic_warmup_type=sarsa --config.image_replay_buffer_kwargs.load_action_samples=False --config.offline_image_replay_buffer_kwargs.load_action_samples=False --params_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance/checkpoint_8000.pkl" --num_diffusion_samples=1 --edit_actor_utd_ratio=1 --exploration_epsilon=0.000005 --online_critic_update_type="sarsa" --on_policy=True --config.agent_kwargs.edit_action_scale=0.3 --ws_critic=False --ws_edit_actor=False --do_ascent=True --eta_ascent=0.001 --num_ascent_steps=30 --critic_balance_classes=True
```

### Minimal intuition experiments
```
WANDB_ENTITY=shreyas-kowshik CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./train_pi_residual.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --pi_config_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k --seed=0 --task_name="put both moka pots on the stove" --config.batch_size=256 --environment_name=libero --wandb_experiment_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-minimal_intuition-1k_online_steps --final_step_sparse_reward=False --online_trajectory_collection_frequency 1000 --num_trajectories_to_collect=15 --config.utd_ratio=1 --config.num_eval_episodes=10 --config.num_episodes_per_video=5 --config.eval_interval=10000 --reward_scale=1.0 --reward_bias=-0.1 --config.libero_tfrecord_regexp="/data/user_data/skowshik/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_base_rollouts/image_replay_buffer/*.tfrecord" --filter_successful_trajectories=False --config.save_dir="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-minimal_intuition-1k_online_steps" --scale_success_alpha=2.0 --intermediate_reward_mul_factor 10.0 --warmup_steps 0 --num_train_steps 5000000 --balance_offline_training_data=False --config.save_interval=1000 --critic_warmup_type=sarsa --config.image_replay_buffer_kwargs.load_action_samples=False --config.offline_image_replay_buffer_kwargs.load_action_samples=False --params_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance/checkpoint_8000.pkl" --num_diffusion_samples=1 --edit_actor_utd_ratio=1 --exploration_epsilon=0.000005 --online_critic_update_type="sarsa" --on_policy=True --config.agent_kwargs.edit_action_scale=0.3 --ws_critic=False --ws_edit_actor=False --do_ascent=True --eta_ascent=0.001 --num_ascent_steps=30
```

```
WANDB_ENTITY=shreyas-kowshik CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./train_pi_residual.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --pi_config_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k --seed=0 --task_name="put both moka pots on the stove" --config.batch_size=256 --environment_name=libero --wandb_experiment_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-minimal_intuition-1k_online_steps-critic_balance --final_step_sparse_reward=False --online_trajectory_collection_frequency 1000 --num_trajectories_to_collect=15 --config.utd_ratio=1 --config.num_eval_episodes=10 --config.num_episodes_per_video=5 --config.eval_interval=10000 --reward_scale=1.0 --reward_bias=-0.1 --config.libero_tfrecord_regexp="/data/user_data/skowshik/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_base_rollouts/image_replay_buffer/*.tfrecord" --filter_successful_trajectories=False --config.save_dir="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-minimal_intuition-1k_online_steps-critic_balance" --scale_success_alpha=2.0 --intermediate_reward_mul_factor 10.0 --warmup_steps 0 --num_train_steps 5000000 --balance_offline_training_data=False --config.save_interval=1000 --critic_warmup_type=sarsa --config.image_replay_buffer_kwargs.load_action_samples=False --config.offline_image_replay_buffer_kwargs.load_action_samples=False --params_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance/checkpoint_8000.pkl" --num_diffusion_samples=1 --edit_actor_utd_ratio=1 --exploration_epsilon=0.000005 --online_critic_update_type="sarsa" --on_policy=True --config.agent_kwargs.edit_action_scale=0.3 --ws_critic=False --ws_edit_actor=False --do_ascent=True --eta_ascent=0.001 --num_ascent_steps=30 --critic_balance_classes=True
```

```
WANDB_ENTITY=shreyas-kowshik CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./train_pi_residual.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --pi_config_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k --seed=0 --task_name="put both moka pots on the stove" --config.batch_size=256 --environment_name=libero --wandb_experiment_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-minimal_intuition-5k_online_steps-critic_balance --final_step_sparse_reward=False --online_trajectory_collection_frequency 5000 --num_trajectories_to_collect=15 --config.utd_ratio=1 --config.num_eval_episodes=10 --config.num_episodes_per_video=5 --config.eval_interval=10000 --reward_scale=1.0 --reward_bias=-0.1 --config.libero_tfrecord_regexp="/data/user_data/skowshik/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_base_rollouts/image_replay_buffer/*.tfrecord" --filter_successful_trajectories=False --config.save_dir="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-minimal_intuition-5k_online_steps-critic_balance" --scale_success_alpha=2.0 --intermediate_reward_mul_factor 10.0 --warmup_steps 0 --num_train_steps 5000000 --balance_offline_training_data=False --config.save_interval=1000 --critic_warmup_type=sarsa --config.image_replay_buffer_kwargs.load_action_samples=False --config.offline_image_replay_buffer_kwargs.load_action_samples=False --params_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance/checkpoint_8000.pkl" --num_diffusion_samples=1 --edit_actor_utd_ratio=1 --exploration_epsilon=0.000005 --online_critic_update_type="sarsa" --on_policy=True --config.agent_kwargs.edit_action_scale=0.3 --ws_critic=False --ws_edit_actor=False --do_ascent=True --eta_ascent=0.001 --num_ascent_steps=30 --critic_balance_classes=True

python dump_tf_record.py --tfrecord_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_8k_init-gradq_ascent_policy-minimal_intuition-5k_online_steps-critic_balance/seed_0/image_replay_buffer/episode_15.tfrecord" --output_gif_path "./debug15.mp4"
```
```

## Warmstart critic and residual actor
rm -r pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_CQLQL && WANDB_ENTITY=shreyas-kowshik CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./train_pi_residual.py --config=configs/libero_config.py:pi_residual_td3 --seed=0 --task_name="put both moka pots on the stove" --config.batch_size=256 --environment_name=libero --wandb_experiment_name=calql_debug_v1 --num_edit_samples=4 --num_actions_to_sample=4 --final_step_sparse_reward=False --online_trajectory_collection_frequency 5000000 --num_trajectories_to_collect=1 --config.utd_ratio=4 --config.num_eval_episodes=10 --config.num_episodes_per_video=5 --config.eval_interval=1000000 --reward_scale=1.0 --reward_bias=-0.1 --config.libero_tfrecord_regexp="/data/user_data/skowshik/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_base_rollouts/image_replay_buffer/*.tfrecord" --filter_successful_trajectories=False --config.save_dir="./pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_CQLQL" --scale_success_alpha=2.0 --intermediate_reward_mul_factor 10.0 --warmup_steps 50000 --num_train_steps 50000 --balance_offline_training_data=True --config.save_interval=200 --critic_warmup_type=calql --config.image_replay_buffer_kwargs.load_action_samples=True --config.offline_image_replay_buffer_kwargs.load_action_samples=True --calql_lower_bound=-200.0 --cql_alpha=0.1 --cql_temp=1.0


#########################
### TD3 GRPO
##### Warmstart Critic
```
rm -r pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_calql_10_2_v2 && WANDB_ENTITY=shreyas-kowshik CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./train_pi_residual.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --pi_config_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k --seed=0 --task_name="put both moka pots on the stove" --config.batch_size=256 --environment_name=libero --wandb_experiment_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_calql_10_2_v2 --num_edit_samples=4 --num_actions_to_sample=4 --final_step_sparse_reward=False --online_trajectory_collection_frequency 5000000 --num_trajectories_to_collect=1 --config.utd_ratio=1 --config.num_eval_episodes=10 --config.num_episodes_per_video=5 --config.eval_interval=1000000 --reward_scale=1.0 --reward_bias=-0.1 --config.libero_tfrecord_regexp="/data/user_data/skowshik/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_base_rollouts/image_replay_buffer/*.tfrecord" --filter_successful_trajectories=False --config.save_dir="./pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_calql_10_2_v2" --scale_success_alpha=2.0 --intermediate_reward_mul_factor 10.0 --warmup_steps 50000 --num_train_steps 50000 --balance_offline_training_data=True --config.save_interval=500 --critic_warmup_type=calql --config.image_replay_buffer_kwargs.load_action_samples=True --config.offline_image_replay_buffer_kwargs.load_action_samples=True --ws_critic=True --ws_edit_actor=False --calql_lower_bound=-300.0 --cql_alpha=0.1 --cql_temp=1.0 --calql_random_actions=10
```

```
rm -r pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_calql_10_2_v2_no_balance && WANDB_ENTITY=shreyas-kowshik CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./train_pi_residual.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --pi_config_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k --seed=0 --task_name="put both moka pots on the stove" --config.batch_size=256 --environment_name=libero --wandb_experiment_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_calql_10_2_v2_no_balance --num_edit_samples=4 --num_actions_to_sample=4 --final_step_sparse_reward=False --online_trajectory_collection_frequency 5000000 --num_trajectories_to_collect=1 --config.utd_ratio=1 --config.num_eval_episodes=10 --config.num_episodes_per_video=5 --config.eval_interval=1000000 --reward_scale=1.0 --reward_bias=-0.1 --config.libero_tfrecord_regexp="/data/user_data/skowshik/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_base_rollouts/image_replay_buffer/*.tfrecord" --filter_successful_trajectories=False --config.save_dir="./pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_calql_10_2_v2_no_balance" --scale_success_alpha=2.0 --intermediate_reward_mul_factor 10.0 --warmup_steps 50000 --num_train_steps 50000 --balance_offline_training_data=False --config.save_interval=1000 --critic_warmup_type=calql --config.image_replay_buffer_kwargs.load_action_samples=True --config.offline_image_replay_buffer_kwargs.load_action_samples=True --ws_critic=True --ws_edit_actor=False --calql_lower_bound=-300.0 --cql_alpha=0.1 --cql_temp=1.0 --calql_random_actions=10
```

```
rm -r pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance && WANDB_ENTITY=shreyas-kowshik CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./train_pi_residual.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --pi_config_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k --seed=0 --task_name="put both moka pots on the stove" --config.batch_size=256 --environment_name=libero --wandb_experiment_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance --num_edit_samples=4 --num_actions_to_sample=4 --final_step_sparse_reward=False --online_trajectory_collection_frequency 5000000 --num_trajectories_to_collect=1 --config.utd_ratio=1 --config.num_eval_episodes=10 --config.num_episodes_per_video=5 --config.eval_interval=1000000 --reward_scale=1.0 --reward_bias=-0.1 --config.libero_tfrecord_regexp="/data/user_data/skowshik/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_base_rollouts/image_replay_buffer/*.tfrecord" --filter_successful_trajectories=False --config.save_dir="./pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance" --scale_success_alpha=2.0 --intermediate_reward_mul_factor 10.0 --warmup_steps 50000 --num_train_steps 50000 --balance_offline_training_data=True --config.save_interval=1000 --critic_warmup_type=sarsa --config.image_replay_buffer_kwargs.load_action_samples=True --config.offline_image_replay_buffer_kwargs.load_action_samples=True --ws_critic=True --ws_edit_actor=False
```

```
rm -r pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_calql_10_2_clean_v2_balance && WANDB_ENTITY=shreyas-kowshik CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./train_pi_residual.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --pi_config_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k --seed=0 --task_name="put both moka pots on the stove" --config.batch_size=256 --environment_name=libero --wandb_experiment_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_calql_ws_10_2_clean_v2_balance --num_edit_samples=4 --num_actions_to_sample=4 --final_step_sparse_reward=False --online_trajectory_collection_frequency 5000000 --num_trajectories_to_collect=1 --config.utd_ratio=1 --config.num_eval_episodes=10 --config.num_episodes_per_video=5 --config.eval_interval=1000000 --reward_scale=1.0 --reward_bias=-0.1 --config.libero_tfrecord_regexp="/data/user_data/skowshik/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_base_rollouts/image_replay_buffer/*.tfrecord" --filter_successful_trajectories=False --config.save_dir="./pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_calql_10_2_clean_v2_balance" --scale_success_alpha=2.0 --intermediate_reward_mul_factor 10.0 --warmup_steps 50000 --num_train_steps 50000 --balance_offline_training_data=True --config.save_interval=500 --critic_warmup_type=calql --config.image_replay_buffer_kwargs.load_action_samples=True --config.offline_image_replay_buffer_kwargs.load_action_samples=True --ws_critic=True --ws_edit_actor=False --calql_lower_bound=-300.0 --cql_alpha=0.1 --cql_temp=1.0
```

##### Warmstart Edit Actor
```
rm -r pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_bc_edit_actor && WANDB_ENTITY=shreyas-kowshik CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./train_pi_residual.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --pi_config_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k --seed=0 --task_name="put both moka pots on the stove" --config.batch_size=256 --environment_name=libero --wandb_experiment_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_bc_edit_actor_ws_clean_v1 --num_edit_samples=4 --num_actions_to_sample=4 --final_step_sparse_reward=False --online_trajectory_collection_frequency 5000000 --num_trajectories_to_collect=1 --config.utd_ratio=1 --config.num_eval_episodes=10 --config.num_episodes_per_video=5 --config.eval_interval=1000000 --reward_scale=1.0 --reward_bias=-0.1 --config.libero_tfrecord_regexp="/data/user_data/skowshik/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_base_rollouts/image_replay_buffer/*.tfrecord" --filter_successful_trajectories=False --config.save_dir="./pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_bc_edit_actor" --scale_success_alpha=2.0 --intermediate_reward_mul_factor 10.0 --warmup_steps 50000 --num_train_steps 50000 --config.save_interval=1000 --critic_warmup_type=sarsa --config.image_replay_buffer_kwargs.load_action_samples=True --config.offline_image_replay_buffer_kwargs.load_action_samples=True --ws_critic=False --ws_edit_actor=True --params_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2/checkpoint_9000.pkl" --grpo_beta=50.0 --grpo_weight_threshold=50.0
```
#########################

##### Full Training
```
rm -r pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_td3_grpo_clean_v1_continued_4.8k_v2_3.4k && WANDB_ENTITY=shreyas-kowshik CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./train_pi_residual.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --pi_config_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k --seed=0 --task_name="put both moka pots on the stove" --config.batch_size=256 --environment_name=libero --wandb_experiment_name=full_training_td3_grpo_ep5_base_clean_v1_continued_4.8k --num_edit_samples=4 --num_actions_to_sample=4 --final_step_sparse_reward=False --online_trajectory_collection_frequency 500 --num_trajectories_to_collect=1 --config.utd_ratio=8 --config.num_eval_episodes=10 --config.num_episodes_per_video=5 --config.eval_interval=1000 --reward_scale=1.0 --reward_bias=-0.1 --config.libero_tfrecord_regexp="/data/user_data/skowshik/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_base_rollouts/image_replay_buffer/*.tfrecord" --filter_successful_trajectories=True --config.save_dir="./pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_td3_grpo_clean_v1_continued_4.8k_v2_3.4k" --scale_success_alpha=2.0 --intermediate_reward_mul_factor 10.0 --warmup_steps=0 --num_train_steps 50000 --config.save_interval=1000 --critic_warmup_type=sarsa --config.image_replay_buffer_kwargs.load_action_samples=True --config.offline_image_replay_buffer_kwargs.load_action_samples=True --ws_critic=False --ws_edit_actor=False --params_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_td3_grpo_clean_v1_continued_4.8k/checkpoint_3400.pkl" --num_diffusion_samples=16 --grpo_beta=50.0 --grpo_weight_threshold=50.0 --exploration_epsilon=0.2
```

```
rm -r pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_bc_ws && WANDB_ENTITY=shreyas-kowshik CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./train_pi_residual.py --config=configs/libero_config.py:pi_residual_td3 --agent_name=pi_residual_td3 --pi_config_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k --seed=0 --task_name="put both moka pots on the stove" --config.batch_size=256 --environment_name=libero --wandb_experiment_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_bc_ws --num_edit_samples=4 --num_actions_to_sample=4 --final_step_sparse_reward=False --online_trajectory_collection_frequency 5000000 --num_trajectories_to_collect=1 --config.utd_ratio=1 --config.num_eval_episodes=10 --config.num_episodes_per_video=5 --config.eval_interval=1000000 --reward_scale=1.0 --reward_bias=-0.1 --config.libero_tfrecord_regexp="/data/user_data/skowshik/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_base_rollouts/image_replay_buffer/*.tfrecord" --filter_successful_trajectories=True --config.save_dir="./pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_bc_ws" --scale_success_alpha=2.0 --intermediate_reward_mul_factor 10.0 --warmup_steps 50000 --num_train_steps 50000 --balance_offline_training_data=False --config.save_interval=1000 --critic_warmup_type=sarsa --config.image_replay_buffer_kwargs.load_action_samples=True --config.offline_image_replay_buffer_kwargs.load_action_samples=True --ws_critic=False --ws_edit_actor=True --params_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance/checkpoint_5000.pkl"
```

```
rm -r pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_bc_ws && WANDB_ENTITY=shreyas-kowshik CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./train_pi_residual.py --config=configs/libero_config.py:pi_residual_td3 --agent_name=pi_residual_td3 --pi_config_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k --seed=0 --task_name="put both moka pots on the stove" --config.batch_size=256 --environment_name=libero --wandb_experiment_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_bc_ws --num_edit_samples=4 --num_actions_to_sample=4 --final_step_sparse_reward=False --online_trajectory_collection_frequency 5000000 --num_trajectories_to_collect=1 --config.utd_ratio=1 --config.num_eval_episodes=10 --config.num_episodes_per_video=5 --config.eval_interval=1000000 --reward_scale=1.0 --reward_bias=-0.1 --config.libero_tfrecord_regexp="/data/user_data/skowshik/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_base_rollouts/image_replay_buffer/*.tfrecord" --filter_successful_trajectories=True --config.save_dir="./pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_bc_ws" --scale_success_alpha=2.0 --intermediate_reward_mul_factor 10.0 --warmup_steps 50000 --num_train_steps 50000 --balance_offline_training_data=False --config.save_interval=1000 --critic_warmup_type=sarsa --config.image_replay_buffer_kwargs.load_action_samples=True --config.offline_image_replay_buffer_kwargs.load_action_samples=True --ws_critic=False --ws_edit_actor=True --params_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance/checkpoint_5000.pkl"
```

```
rm -rf /data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_bc_ws_full_train-v2 && WANDB_ENTITY=shreyas-kowshik CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./train_pi_residual.py --config=configs/libero_config.py:pi_residual_td3 --agent_name=pi_residual_td3 --pi_config_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k --seed=0 --task_name="put both moka pots on the stove" --config.batch_size=256 --environment_name=libero --wandb_experiment_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_bc_ws_full_train-v2 --num_edit_samples=4 --num_actions_to_sample=4 --final_step_sparse_reward=False --online_trajectory_collection_frequency 500 --num_trajectories_to_collect=5 --config.utd_ratio=1 --config.num_eval_episodes=10 --config.num_episodes_per_video=5 --config.eval_interval=1000 --reward_scale=1.0 --reward_bias=-0.1 --config.libero_tfrecord_regexp="/data/user_data/skowshik/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_base_rollouts/image_replay_buffer/*.tfrecord" --filter_successful_trajectories=True --config.save_dir="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance_bc_ws_full_train-v2" --scale_success_alpha=2.0 --intermediate_reward_mul_factor 10.0 --warmup_steps 8000 --num_train_steps 5000000 --balance_offline_training_data=False --config.save_interval=1000 --critic_warmup_type=sarsa --config.image_replay_buffer_kwargs.load_action_samples=False --config.offline_image_replay_buffer_kwargs.load_action_samples=False --ws_critic=False --ws_edit_actor=True --params_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_clean_v2_balance/checkpoint_8000.pkl" --num_diffusion_samples=2 --edit_actor_utd_ratio=8 --exploration_epsilon=0.00005
```

##### Q* Training
```

```

```
rm -r pi05_libero_gradacc2_2k_q_star_sarsa_10_2_rng_fix && WANDB_ENTITY=shreyas-kowshik CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./train_pi_residual.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --pi_config_name=pi05_libero_gradacc2_2k --seed=0 --task_name="put both moka pots on the stove" --config.batch_size=256 --environment_name=libero --wandb_experiment_name=pi05_libero_gradacc2_2k_q_star_sarsa_10_2_rng_fix --num_edit_samples=4 --num_actions_to_sample=4 --final_step_sparse_reward=False --online_trajectory_collection_frequency 5000000 --num_trajectories_to_collect=1 --config.utd_ratio=1 --config.num_eval_episodes=10 --config.num_episodes_per_video=5 --config.eval_interval=1000000 --reward_scale=1.0 --reward_bias=-0.1 --config.libero_tfrecord_regexp="/data/hf_cache/models/pi05_libero_gradacc2_2k_base_trajectories_rng_fix/image_replay_buffer/*.tfrecord" --filter_successful_trajectories=False --config.save_dir="./pi05_libero_gradacc2_2k_q_star_sarsa_10_2_rng_fix" --scale_success_alpha=2.0 --intermediate_reward_mul_factor 10.0 --warmup_steps 50000 --num_train_steps 50000 --balance_offline_training_data=True --config.save_interval=500 --critic_warmup_type=sarsa --config.image_replay_buffer_kwargs.load_action_samples=True --config.offline_image_replay_buffer_kwargs.load_action_samples=True --ws_critic=True --ws_edit_actor=False
```



##### Evaluation
Baseline for trained model (base model)
```
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./evaluate_traj.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --seed=0 --environment_name=libero --reward_scale=1.0 --reward_bias=-0.1 --filter_successful_trajectories=False --config.save_dir="/home/skowshik/vla/codebase/PolicyAgnosticRL/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_vision_init_fullft_action_BASELINE" --num_trajectories_to_collect=50 --pi_config_name="pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_vision_init_fullft_action" --params_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_td3_grpo_clean_v1/checkpoint_4800.pkl" --base_actions=True
```

```
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./evaluate_traj.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --seed=0 --environment_name=libero --reward_scale=1.0 --reward_bias=-0.1 --filter_successful_trajectories=False --config.save_dir="/home/skowshik/vla/codebase/PolicyAgnosticRL/td3_grpo_full_training_clean_v1_7k_evaluation" --num_trajectories_to_collect=50 --pi_config_name="pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k" --params_path="/home/skowshik/vla/codebase/PolicyAgnosticRL/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_td3_grpo_clean_v1_continued_4.8k_v2_3.4k/checkpoint_7000.pkl"
```

## PPO warm start training
```
rm -r debug_refactor && WANDB_ENTITY=shreyas-kowshik CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./train_pi_residual.py --config=configs/libero_config.py:pi_residual_ppo --seed=0 --task_name="put both moka pots on the stove" --config.batch_size=256 --config.agent_kwargs.batch_size=256 --environment_name=libero --wandb_experiment_name=debug_res_td3_ws --num_edit_samples=4 --num_actions_to_sample=4 --final_step_sparse_reward=False --online_trajectory_collection_frequency 200 --num_trajectories_to_collect=10 --config.utd_ratio=1 --config.num_eval_episodes=10 --config.num_episodes_per_video=5 --config.eval_interval=1000000 --reward_scale=1.0 --reward_bias=-0.1 --config.libero_tfrecord_regexp="/home/skowshik/vla/codebase/PolicyAgnosticRL/debug_pi_res_td3/image_replay_buffer/*.tfrecord" --filter_successful_trajectories=True --config.save_dir="./debug_refactor" --scale_success_alpha=2.0 --intermediate_reward_mul_factor 10.0 --warmup_steps 5000 --num_train_steps 50000 --config.image_replay_buffer_kwargs.use_gae=True --balance_offline_training_data=False --config.save_interval=200 --on_policy=True --critic_success_wt=10.0 --bc_loss_coef=1000.0
```

### 6 Dimensional Action Trainings
```
WANDB_ENTITY=shreyas-kowshik CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./train_pi_residual.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --pi_config_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k --seed=0 --task_name="put both moka pots on the stove" --config.batch_size=256 --environment_name=libero --wandb_experiment_name=pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_6dim-clean_v2 --num_edit_samples=4 --num_actions_to_sample=4 --final_step_sparse_reward=False --online_trajectory_collection_frequency 5000000 --num_trajectories_to_collect=1 --config.utd_ratio=1 --config.num_eval_episodes=10 --config.num_episodes_per_video=5 --config.eval_interval=1000000 --reward_scale=1.0 --reward_bias=-0.1 --config.libero_tfrecord_regexp="/data/user_data/skowshik/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_base_rollouts/image_replay_buffer/*.tfrecord" --filter_successful_trajectories=False --config.save_dir="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_6dim-clean_v2" --scale_success_alpha=2.0 --intermediate_reward_mul_factor 10.0 --warmup_steps 50000 --num_train_steps 50000 --balance_offline_training_data=True --config.save_interval=1000 --critic_warmup_type=sarsa --config.image_replay_buffer_kwargs.load_action_samples=True --config.offline_image_replay_buffer_kwargs.load_action_samples=True --ws_critic=True --ws_edit_actor=False
```


```
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m pdb ./sample_base_traj.py --config=configs/libero_config.py:pi_residual_td3_grpo --agent_name=pi_residual_td3_grpo --seed=0 --environment_name=libero --reward_scale=1.0 --reward_bias=-0.1 --filter_successful_trajectories=False --config.save_dir="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_6dim-clean_v2-20k_0.001_5" --num_trajectories_to_collect=50 --pi_config_name="pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k" --num_diffusion_samples=1 --params_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_6dim-clean_v2/checkpoint_20000.pkl" --do_ascent=True --eta_ascent=0.001 --num_ascent_steps=5
```

```
python dump_tf_record.py --tfrecord_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_6dim-clean_v2-20k_0.001_5/image_replay_buffer/episode_0.tfrecord" --output_gif_path "./debug_v2_20k_5_ep0.mp4"
```

```
python find_traj_stats.py --tf_record_path="/data/hf_cache/models/pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_ws_sarsa_10_2_6dim-clean_v2-20k_0.001_5/image_replay_buffer/*.tfrecord"
```

