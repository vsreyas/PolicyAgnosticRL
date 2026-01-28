# PolicyAgnosticRL Codebase Index

## Overview
Jax codebase for **Policy Agnostic RL: Offline RL and Online RL Fine-Tuning of Any Policy Class and Backbone**.
Paper: https://arxiv.org/abs/2412.06685

This codebase supports training and fine-tuning various RL agents (including Diffusion Policies, Pi0/Pi0.5, OpenVLA) using policy-agnostic reinforcement learning methods across multiple simulation and real-world environments.

---

## Directory Structure

### 📁 Root Level Files

#### Main Training Scripts
- `train.py` - Main training script for general RL agents
- `train_pi.py` - Training script for Pi0/Pi0.5 policies
- `train_pi_residual.py` - Training script for residual actor learning on top of Pi0.5
- `train_filtered_bc.py` - Behavioral cloning training with filtered data
- `evaluate_traj.py` - Trajectory evaluation script

#### Utility Scripts
- `sample_base_traj.py` - Sample trajectories from base policies for warmup
- `visualize_traj_predictions.py` - Visualize trajectory predictions
- `dump_tf_record.py` - Dump TFRecord files to video/images
- `dump_vlm_actions.py` - Dump VLM (Vision-Language Model) actions
- `find_traj_stats.py` - Calculate trajectory statistics
- `merge_pkl_files.py` - Merge pickle files
- `filter_pkl.py` - Filter pickle files
- `umap_action_vis.py` - UMAP visualization for actions
- `critic_ws_pg.py` - Critic warmstart policy gradient
- `noop.py` - No-op baseline script
- `debug_*.py` - Various debugging scripts

#### Configuration Files
- `environment.yml` - Conda environment specification
- `setup.py` - Package setup file
- `requirements*.txt` - Python dependencies (multiple variants)
- `*.slurm` - SLURM batch job scripts
- `run.sh` - Bash execution script

#### Documentation
- `../README.md` - Main documentation with setup and usage instructions
- `../INSTALLATION_NOTES.md` - Additional installation notes
- `code_index/` - Comprehensive codebase documentation (INDEX, ARCHITECTURE_GUIDE, FILE_REFERENCE, DEVELOPMENT_GUIDE)

---

### 📁 jaxrl_m/ - Core JAX RL Library

#### jaxrl_m/agents/
Agent implementations for reinforcement learning.

**jaxrl_m/agents/continuous/**
- `base_policy.py` - Base policy interface
- `bc.py` - Behavioral Cloning
- `ddpm_bc.py` - DDPM (Diffusion) + Behavioral Cloning
- `pi_0.py` - Pi0 policy implementation
- `openvla.py` - OpenVLA integration
- `sac.py` - Soft Actor-Critic
- `cql.py` - Conservative Q-Learning
- `calql.py` - CQL with Actor Loss
- `iql.py` - Implicit Q-Learning
- `parl_calql.py` - Policy Agnostic RL with CalQL
- `parl_iql.py` - Policy Agnostic RL with IQL
- `diffusion_q_learning.py` - Diffusion-based Q-learning
- `awr_residual.py` - Advantage-Weighted Regression for residual policies
- `action_optimization.py` - Action optimization utilities
- `auto_regressive_transformer.py` - Transformer-based autoregressive policies
- `expo_pi.py` - Exponential policy utilities
- `expo_pi_cache.py` - Cached version of exponential policies

**jaxrl_m/agents/continuous/pi_vlm_cached/**
Residual learning on top of VLM policies:
- `residual_base.py` - Base residual policy
- `residual_ppo.py` - PPO-based residual learning
- `residual_td3.py` - TD3-based residual learning

#### jaxrl_m/networks/
Neural network architectures:
- `mlp.py` - Multi-Layer Perceptrons
- `actor_critic_nets.py` - Actor-Critic networks
- `diffusion_nets.py` - Diffusion model networks
- `distributional.py` - Distributional RL networks
- `discrete_nets.py` - Discrete action networks
- `lagrange.py` - Lagrange multiplier networks

#### jaxrl_m/vision/
Vision encoders and processing:
- `resnet_v1.py` - ResNet V1 encoder
- `impala.py` - IMPALA encoder
- `small_encoders.py` - Small CNN encoders
- `robofume_encoders.py` - RoboFuME encoders
- `bigvision_*.py` - BigVision utilities and ResNetV2
- `cvae.py` - Conditional VAE
- `data_augmentations.py` - Data augmentation utilities
- `film_conditioning_layer.py` - FiLM conditioning

#### jaxrl_m/data/
Dataset handling and replay buffers:
- `dataset.py` - General dataset interface
- `replay_buffer.py` - Standard replay buffer
- `image_replay_buffer.py` - Image-based replay buffer
- `image_replay_buffer_pi.py` - Pi0 image replay buffer
- `img_replay_buffer_pi_bc.py` - BC version of Pi replay buffer
- `goal_conditioned_buffer.py` - Goal-conditioned replay buffer
- `bridge_dataset.py` - Bridge Data V2 dataset
- `libero_dataset.py` - LIBERO dataset
- `calvin_dataset.py` - CALVIN dataset
- `calvin_to_lerobot.py` - CALVIN to LeRobot conversion
- `roboverse_dataset.py` - Roboverse dataset
- `tf_augmentations.py` - TensorFlow augmentations
- `tf_goal_relabeling.py` - TensorFlow goal relabeling
- `np_goal_relabeling.py` - NumPy goal relabeling
- `text_processing.py` - Text processing utilities

#### jaxrl_m/envs/
Environment wrappers and configurations:
- `d4rl.py` - D4RL environments
- `libero.py` - LIBERO environment
- `calvin.py` - CALVIN environment
- `metaworld.py` - Meta-World environment
- `pointmaze.py` - PointMaze environment
- `robot_env.py` - Real robot environment
- `franka_env.py` - Franka robot environment
- `robot_manual_reward_functions.py` - Manual reward functions
- `procgen.py` - Procgen environments
- `discretize_continuous_action.py` - Action discretization

**jaxrl_m/envs/wrappers/**
Environment wrappers:
- `video_recorder.py` - Video recording
- `reward_scale.py` - Reward scaling
- `norm.py` - Observation/action normalization
- `mujoco.py` - MuJoCo wrappers
- `chunking.py` - Action chunking
- `remap.py` - Action/observation remapping
- `dmcgym.py` - DeepMind Control Suite wrapper
- `roboverse.py` - Roboverse wrapper

#### jaxrl_m/transformers/
Transformer-based models:
- `q_functions.py` - Q-function transformers

**jaxrl_m/transformers/models/**
- `infrastructure.py` - Transformer infrastructure
- `q_transformer.py` - Q-Transformer implementation

**jaxrl_m/transformers/utils/**
- Utility functions for transformers

#### jaxrl_m/common/
Common utilities:
- `common.py` - Common functions
- `evaluation.py` - Evaluation utilities
- `encoding.py` - Encoding utilities
- `initialization.py` - Weight initialization
- `optimizers.py` - Optimizer configurations
- `wandb.py` - Weights & Biases integration
- `typing.py` - Type definitions
- `traj.py` - Trajectory utilities

#### jaxrl_m/utils/
Additional utilities:
- `train_utils.py` - Training utilities
- `train_state_utils.py` - Training state management
- `pi0_train_utils.py` - Pi0-specific training utilities
- `jax_utils.py` - JAX utilities
- `sim_utils.py` - Simulation utilities
- `visualization_utils.py` - Visualization utilities
- `expo_utils.py` - Exponential utilities
- `timer_utils.py` - Timing utilities
- `gym_text_patch.py` - Gym text patching

---

### 📁 configs/
Configuration files for different environments and experiments:
- `base_config.py` - Base configuration
- `state_config.py` - State-based environments (AntMaze, etc.)
- `libero_config.py` - LIBERO robot manipulation tasks
- `calvin_config.py` - CALVIN environment
- `bridgedata_config.py` - Bridge Data V2 real robot tasks
- `real_config.py` - Real robot deployment config

---

### 📁 scripts/
Helper scripts and documentation:
- `example_usage.py` / `example_usage.sh` - Example usage
- `openvla_caching_worker.py` - OpenVLA action caching worker
- `bridgedata_raw_to_numpy.py` - Bridge data preprocessing
- `bridgedata_numpy_to_tfrecord.py` - Convert to TFRecord format
- `load_vlm_actions.py` - Load VLM actions
- `maze2d_utils.py` - Maze2D utilities
- `*.md` - Documentation files (QUICK_START, SUMMARY, CHANGES, etc.)

---

### 📁 data_info/
Information about datasets (contents not indexed)

### 📁 pkl_files/
Pickle files for trajectories and checkpoints

### 📁 logs/ & outputs/
Training logs and outputs

### 📁 results/
Experiment results and checkpoints

### 📁 plots/
Generated plots and visualizations

### 📁 eval_gifs/
Evaluation visualization GIFs

### 📁 perturbation_analysis/
Perturbation analysis results

---

## Key Workflows

### 1. **Training a Diffusion Policy with BC**
```bash
python ./train.py --environment_name=antmaze-large-diverse-v2 \
  --config=./configs/state_config.py:ddpm --num_offline_epochs=3000
```

### 2. **Training PA-RL (Offline + Online)**
```bash
python ./train.py --environment_name=antmaze-large-diverse-v2 \
  --config=./configs/state_config.py:parl_calql \
  --config.base_policy_path=ddpm:./results/.../checkpoint_3000/
```

### 3. **Training Pi0 on LIBERO**
```bash
python ./train_pi.py --environment_name=libero \
  --config=./configs/libero_config.py:pi0 --num_offline_epochs=30
```

### 4. **Residual Learning Workflow**
a. Sample base trajectories:
```bash
python ./sample_base_traj.py --config=configs/libero_config.py:pi_residual_td3
```

b. Train residual actor:
```bash
python ./train_pi_residual.py --config=configs/libero_config.py:pi_residual_td3
```

### 5. **OpenVLA Fine-Tuning**
Requires running caching workers in parallel:
```bash
python scripts/openvla_caching_worker.py --checkpoints_dir=... --instruction="..."
```

### 6. **Trajectory Visualization**
```bash
python visualize_traj_predictions.py --tfrecord_path=... --checkpoint_path=...
```

---

## Supported Environments

### Simulation
- **D4RL**: AntMaze, HalfCheetah, Hopper, Walker2D, etc.
- **Meta-World**: Multi-task robot manipulation
- **PointMaze**: Navigation tasks
- **Procgen**: Procedurally generated games
- **LIBERO**: Robot manipulation benchmarks
- **CALVIN**: Language-conditioned robot tasks

### Real Robot
- **Bridge Data Robot**: Real Franka robot experiments
- Custom real robot environments

---

## Supported Policy Types

1. **Diffusion Policies** (DDPM)
2. **Pi0 / Pi0.5** - Vision-Language-Action models
3. **OpenVLA** - Open-source VLA model
4. **Transformer Policies** - Autoregressive transformers
5. **Standard RL Policies** - SAC, TD3, PPO, etc.

---

## Key Agent Types

### Offline RL Agents
- **BC** - Behavioral Cloning
- **CQL** - Conservative Q-Learning
- **IQL** - Implicit Q-Learning
- **CalQL** - CQL with Actor Loss

### Policy Agnostic RL Agents
- **PARL-CalQL** - PA-RL with CalQL
- **PARL-IQL** - PA-RL with IQL

### Residual Learning
- **Residual TD3** - TD3-based residual learning
- **Residual PPO** - PPO-based residual learning
- **AWR Residual** - Advantage-weighted residual

---

## Data Formats

- **TFRecord**: Primary format for training data
- **Pickle (.pkl)**: Checkpoints and trajectory data
- **NumPy**: Intermediate data processing
- **MP4/GIF**: Trajectory visualizations

---

## Training Outputs

Each experiment creates:
- `agent_checkpoints/` - Agent model checkpoints
- `base_policy_checkpoints_from_agent_trainer/` - Base policy checkpoints
- `image_replay_buffer/` - Collected trajectories as TFRecords
- Logs and metrics in W&B

---

## Dependencies

Key packages:
- JAX + Flax (neural networks)
- TensorFlow (data pipeline)
- Gymnasium/Gym (environments)
- MuJoCo (physics simulation)
- OpenCV (image processing)
- Weights & Biases (experiment tracking)
- LIBERO, CALVIN (benchmark environments)

See `requirements*.txt` for complete lists.

---

## Common Commands

### Environment Setup
```bash
conda create -n parl python=3.11
conda activate parl
pip install uv
uv pip install -e .
uv pip install -r requirements_clean.txt
```

### SLURM Batch Jobs
- `train.slurm` - Standard training
- `train_libero.slurm` - LIBERO training
- `train_calvin.slurm` - CALVIN training
- `noop.slurm` - Baseline experiments

---

## Notes

- Uses JAX for efficient GPU/TPU computation
- TensorFlow for data pipeline only (not training)
- Supports distributed training on TPUs
- Real robot experiments based on Bridge Data Robot repo
- Extensive configuration system via ML Collections

---

## References

- Paper: https://arxiv.org/abs/2412.06685
- Based on BridgeData V2 codebase
- Contact: maxsobolmark at cmu dot edu

---

*Index generated on: 2026-01-28*
