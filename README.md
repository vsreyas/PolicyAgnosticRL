# Policy Agnostic RL

Jax codebase for [Policy Agnostic RL: Offline RL and Online RL Fine-Tuning of Any Policy Class and Backbone](https://arxiv.org/abs/2412.06685).

## Environment
```
conda create -n parl python=3.11
conda activate parl
pip install -e .
pip install -r requirements.txt
```

If you run into GL/glew.h: No such file or directory, run this:
```
conda install -c conda-forge glew
conda install -c conda-forge mesalib
conda install -c menpo glfw3
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


XLA_PYTHON_CLIENT_PREALLOCATE=false env -u PYOPENGL_PLATFORM python ./train_pi.py --environment_name=libero --wandb_experiment_name=pi0_trial --config=./configs/libero_config.py:pi0 --num_offline_epochs=1 --num_online_epochs=5 --seed=0

XLA_PYTHON_CLIENT_PREALLOCATE=false env -u PYOPENGL_PLATFORM python ./train_pi.py --environment_name=libero --wandb_experiment_name=pi0_trial_parl --config=./configs/libero_config.py:parl_calql_pi0 --seed=0 --reward_bias=-1 --config.agent_kwargs.critic_network_kwargs.hidden_dims=256,256,256,256 --config.base_policy_path=pi0 --config.agent_kwargs.cql_alpha=0.005 --num_offline_epochs=1000 --num_online_epochs=1000 --config.agent_kwargs.distributional_critic_kwargs.q_min=-100 --config.agent_kwargs.critic_ensemble_size=2 --config.data_collection_particle_choosing_strategy="max_q_value" --config.evaluation_particle_choosing_strategy="max_q_value"

XLA_PYTHON_CLIENT_PREALLOCATE=false env -u PYOPENGL_PLATFORM python ./train_pi.py --environment_name=libero --wandb_experiment_name=pi0_trial_parl --config=./configs/libero_config.py:parl_calql_pi0 --seed=0 --reward_bias=-1 --config.agent_kwargs.critic_network_kwargs.hidden_dims=256,256,256,256 --config.base_policy_path=pi0 --config.agent_kwargs.cql_alpha=0.005 --num_offline_epochs=1 --num_online_epochs=1 --config.agent_kwargs.distributional_critic_kwargs.q_min=-100 --config.agent_kwargs.critic_ensemble_size=2 --config.data_collection_particle_choosing_strategy="max_q_value" --config.evaluation_particle_choosing_strategy="max_q_value"

XLA_PYTHON_CLIENT_PREALLOCATE=false env -u PYOPENGL_PLATFORM python ./train_pi.py --environment_name=libero --wandb_experiment_name=pi0_trial_parl --config=./configs/libero_config.py:parl_calql_pi0 --seed=0 --config.agent_kwargs.critic_network_kwargs.hidden_dims=256,256,256,256 --config.base_policy_path=pi0 --config.agent_kwargs.cql_alpha=0.005 --num_offline_epochs=1 --num_online_epochs=1 --config.agent_kwargs.distributional_critic_kwargs.q_min=-100 --config.agent_kwargs.critic_ensemble_size=2 --config.data_collection_particle_choosing_strategy="max_q_value" --config.evaluation_particle_choosing_strategy="max_q_value"


XLA_PYTHON_CLIENT_PREALLOCATE=false env -u PYOPENGL_PLATFORM python ./q_vs_mc_returns.py --environment_name=libero \
                  --wandb_experiment_name=pi0_5_finetuning_parl_qchunking_no_pessimism \
                  --config=./configs/libero_config.py:parl_calql_pi0 \
                  --seed=0\
                  --config.agent_kwargs.critic_network_kwargs.hidden_dims=256,256,256,256 \
                  --config.base_policy_path=pi0 --config.agent_kwargs.cql_alpha=0.000 \
                  --num_offline_epochs=500 --num_online_epochs=50 --config.agent_kwargs.distributional_critic_kwargs.q_min=-100 \
                  --config.agent_kwargs.critic_ensemble_size=2 --config.data_collection_particle_choosing_strategy="max_q_value" \
                  --config.evaluation_particle_choosing_strategy="max_q_value"


XLA_PYTHON_CLIENT_PREALLOCATE=false env -u PYOPENGL_PLATFORM python ./critic_warmstart_debug.py --environment_name=libero \
                  --wandb_experiment_name=pi0_5_finetuning_parl_qchunking_SARSA_Qwarmstart_resume_from_20 \
                  --config=./configs/libero_config.py:parl_calql_pi0 \
                  --seed=0\
                  --config.agent_kwargs.critic_network_kwargs.hidden_dims=256,256,256,256 \
                  --config.base_policy_path=pi0 --config.agent_kwargs.cql_alpha=0.000 \
                  --num_offline_epochs=500 --num_online_epochs=50 \
                  --config.agent_kwargs.critic_ensemble_size=2 \
                  --config.data_collection_particle_choosing_strategy="max_q_value" \
                  --config.evaluation_particle_choosing_strategy="max_q_value"

XLA_PYTHON_CLIENT_PREALLOCATE=false env -u PYOPENGL_PLATFORM python ./train_fbc_pi.py --environment_name=libero --wandb_experiment_name=pi0_trial --config=./configs/libero_config.py:pi0 --num_offline_epochs=1 --num_online_epochs=5 --seed=0 --config.save_dir=results-debug-pi-bc-finetuning/ --config.num_eval_episodes=4
