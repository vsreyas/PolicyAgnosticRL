# Quick File Reference

A concise reference of what each major file does.

## 🎯 Main Entry Points

| File | Purpose | Key Arguments |
|------|---------|--------------|
| `train.py` | Main training script for general RL agents | `--environment_name`, `--config`, `--num_offline_epochs`, `--num_online_epochs` |
| `train_pi.py` | Training script for Pi0/Pi0.5 policies | `--task_name`, `--config`, `--pi_config_name` |
| `train_pi_residual.py` | Train residual actors on top of base policies | `--num_edit_samples`, `--on_policy`, `--bc_loss_coef` |
| `evaluate_traj.py` | Evaluate trained policies and generate metrics | `--checkpoint_path`, `--num_episodes` |

## 📊 Data & Evaluation

| File | Purpose |
|------|---------|
| `sample_base_traj.py` | Collect trajectories from base policy for warmstart |
| `dump_tf_record.py` | Convert TFRecord to video/images |
| `dump_vlm_actions.py` | Extract VLM actions from trajectories |
| `visualize_traj_predictions.py` | Visualize predicted vs actual trajectories |
| `find_traj_stats.py` | Compute statistics on trajectory datasets |
| `umap_action_vis.py` | Create UMAP visualizations of action distributions |

## 🔧 Utilities

| File | Purpose |
|------|---------|
| `merge_pkl_files.py` | Merge multiple pickle checkpoint files |
| `filter_pkl.py` | Filter trajectories by criteria |
| `critic_ws_pg.py` | Critic warmstart with policy gradient |
| `noop.py` | No-operation baseline for comparison |
| `debug_*.py` | Various debugging scripts |

---

## 🤖 Agents (`jaxrl_m/agents/continuous/`)

### Base Policies
| File | Description | Use Case |
|------|-------------|----------|
| `base_policy.py` | Abstract base class for all policies | Interface definition |
| `bc.py` | Behavioral Cloning | Imitation learning from demos |
| `ddpm_bc.py` | Diffusion Policy (DDPM) + BC | High-quality trajectory modeling |
| `pi_0.py` | Pi0/Pi0.5 Vision-Language-Action model | Pre-trained VLA policies |
| `openvla.py` | OpenVLA integration | Open-source VLA fine-tuning |
| `auto_regressive_transformer.py` | Transformer-based policy | Sequence modeling |

### RL Algorithms
| File | Description | Algorithm Type |
|------|-------------|----------------|
| `sac.py` | Soft Actor-Critic | Online RL |
| `cql.py` | Conservative Q-Learning | Offline RL |
| `calql.py` | CQL with Actor Loss | Offline RL |
| `iql.py` | Implicit Q-Learning | Offline RL |
| `diffusion_q_learning.py` | Diffusion-based Q-learning | Offline RL |

### Policy Agnostic RL
| File | Description |
|------|-------------|
| `parl_calql.py` | PA-RL with CalQL critic |
| `parl_iql.py` | PA-RL with IQL critic |
| `expo_pi.py` | Exponential policy utilities |
| `expo_pi_cache.py` | Cached exponential policies |

### Residual Learning (`pi_vlm_cached/`)
| File | Description |
|------|-------------|
| `residual_base.py` | Base class for residual learning |
| `residual_td3.py` | TD3-based residual actor |
| `residual_ppo.py` | PPO-based residual actor |

### Other
| File | Description |
|------|-------------|
| `awr_residual.py` | Advantage-Weighted Regression for residual policies |
| `action_optimization.py` | Cross-entropy method and other action optimizers |

---

## 🧠 Networks (`jaxrl_m/networks/`)

| File | Purpose | Used By |
|------|---------|---------|
| `mlp.py` | Multi-Layer Perceptrons | All agents |
| `actor_critic_nets.py` | Actor and critic network architectures | SAC, TD3, PPO |
| `diffusion_nets.py` | U-Net and score networks for diffusion | DDPM agents |
| `distributional.py` | Distributional RL (quantile networks) | PARL critics |
| `discrete_nets.py` | Networks for discrete action spaces | Discrete agents |
| `lagrange.py` | Lagrange multiplier networks | Constrained RL |

---

## 👁️ Vision Encoders (`jaxrl_m/vision/`)

| File | Architecture | Use Case |
|------|--------------|----------|
| `resnet_v1.py` | ResNet V1 | Standard vision encoder |
| `impala.py` | IMPALA CNN | Fast, lightweight encoder |
| `small_encoders.py` | Small CNNs | Low-dimensional observations |
| `robofume_encoders.py` | RoboFuME pre-trained encoders | Transfer learning |
| `bigvision_resnetv2.py` | ResNet V2 from BigVision | Large-scale pre-training |
| `cvae.py` | Conditional VAE | Generative modeling |
| `data_augmentations.py` | Image augmentations | Data efficiency |
| `film_conditioning_layer.py` | FiLM layers | Conditional encoding |

---

## 🗂️ Data (`jaxrl_m/data/`)

### Datasets
| File | Dataset | Domain |
|------|---------|--------|
| `bridge_dataset.py` | Bridge Data V2 | Real robot manipulation |
| `libero_dataset.py` | LIBERO | Sim robot benchmarks |
| `calvin_dataset.py` | CALVIN | Language-conditioned manipulation |
| `roboverse_dataset.py` | Roboverse | Sim robot tasks |
| `dataset.py` | Generic dataset interface | All |

### Replay Buffers
| File | Purpose |
|------|---------|
| `replay_buffer.py` | Standard replay buffer |
| `image_replay_buffer.py` | Image-based replay buffer with augmentations |
| `image_replay_buffer_pi.py` | Pi0-specific replay buffer |
| `img_replay_buffer_pi_bc.py` | BC version with trajectory sampling |
| `goal_conditioned_buffer.py` | For goal-conditioned RL |

### Processing
| File | Purpose |
|------|---------|
| `tf_augmentations.py` | TensorFlow-based image augmentations |
| `text_processing.py` | Natural language instruction processing |
| `tf_goal_relabeling.py` | Hindsight Experience Replay (HER) |
| `np_goal_relabeling.py` | NumPy-based goal relabeling |
| `calvin_to_lerobot.py` | Convert CALVIN to LeRobot format |

---

## 🌍 Environments (`jaxrl_m/envs/`)

### Environment Implementations
| File | Environment | Type |
|------|-------------|------|
| `d4rl.py` | D4RL benchmarks (AntMaze, etc.) | Sim |
| `libero.py` | LIBERO manipulation tasks | Sim |
| `calvin.py` | CALVIN language tasks | Sim |
| `metaworld.py` | Meta-World multi-task | Sim |
| `pointmaze.py` | PointMaze navigation | Sim |
| `robot_env.py` | Real robot interface | Real |
| `franka_env.py` | Franka Panda robot | Real |
| `procgen.py` | Procgen games | Sim |

### Wrappers (`envs/wrappers/`)
| File | Purpose |
|------|---------|
| `video_recorder.py` | Record episode videos |
| `reward_scale.py` | Scale and shift rewards |
| `norm.py` | Normalize observations/actions |
| `chunking.py` | Action chunking for temporal smoothness |
| `mujoco.py` | MuJoCo-specific wrappers |
| `remap.py` | Remap action/observation spaces |
| `dmcgym.py` | DeepMind Control Suite wrapper |

### Configuration
| Directory/File | Purpose |
|----------------|---------|
| `calvin_config/` | CALVIN environment configs |
| `libero_config/` | LIBERO task configs |
| `robot_manual_reward_functions.py` | Custom reward shaping |
| `discretize_continuous_action.py` | Action discretization |

---

## 🔮 Transformers (`jaxrl_m/transformers/`)

| File | Purpose |
|------|---------|
| `q_functions.py` | Q-function with transformer backbone |
| `models/infrastructure.py` | Transformer building blocks |
| `models/q_transformer.py` | Q-Transformer implementation |
| `utils/misc.py` | Transformer utilities |

---

## 🛠️ Common Utilities (`jaxrl_m/common/`)

| File | Purpose |
|------|---------|
| `common.py` | General-purpose helper functions |
| `evaluation.py` | Evaluation loop and metrics |
| `encoding.py` | Encoder factory and utilities |
| `initialization.py` | Weight initialization schemes |
| `optimizers.py` | Optimizer configurations (Adam, etc.) |
| `wandb.py` | Weights & Biases logging |
| `typing.py` | Type definitions and aliases |
| `traj.py` | Trajectory data structures |

---

## ⚙️ Utils (`jaxrl_m/utils/`)

| File | Purpose |
|------|---------|
| `train_utils.py` | Training loop utilities |
| `train_state_utils.py` | JAX train state management |
| `pi0_train_utils.py` | Pi0-specific training helpers |
| `jax_utils.py` | JAX convenience functions |
| `sim_utils.py` | Simulation utilities |
| `visualization_utils.py` | Plotting and visualization |
| `expo_utils.py` | Exponential utilities |
| `timer_utils.py` | Performance timing |
| `gym_text_patch.py` | Patch Gym for text observations |

---

## ⚙️ Configs (`configs/`)

| File | Purpose | Environments |
|------|---------|--------------|
| `base_config.py` | Base configuration class | All |
| `state_config.py` | State-based environments | D4RL, AntMaze, etc. |
| `libero_config.py` | LIBERO robot tasks | LIBERO |
| `calvin_config.py` | CALVIN language tasks | CALVIN |
| `bridgedata_config.py` | Real robot tasks | Bridge Data V2 |
| `real_config.py` | Real robot deployment | Real Franka robot |

### Config Structure
Each config file typically contains:
- **Agent configs**: `ddpm`, `bc`, `parl_calql`, `pi0`, etc.
- **Environment settings**: observation space, action space, tasks
- **Data settings**: dataset paths, augmentations, batch size
- **Training hyperparameters**: learning rates, epochs, UTD ratio
- **Evaluation settings**: number of episodes, video recording

---

## 📜 Scripts (`scripts/`)

| File | Purpose |
|------|---------|
| `example_usage.py` / `.sh` | Example training commands |
| `openvla_caching_worker.py` | Parallel worker for OpenVLA action caching |
| `bridgedata_raw_to_numpy.py` | Preprocess Bridge Data demos |
| `bridgedata_numpy_to_tfrecord.py` | Convert to TFRecord format |
| `load_vlm_actions.py` | Load cached VLM actions |
| `maze2d_utils.py` | Utilities for Maze2D environments |

### Documentation
| File | Purpose |
|------|---------|
| `QUICK_START.md` | Quick start guide |
| `SUMMARY.md` | Project summary |
| `CHANGES.md` | Change log |
| `FINAL_SUMMARY.md` | Final documentation |
| `CRITIC_CACHE_USAGE.md` | How to use critic caching |

---

## 🎨 Output Files & Directories

| Path | Contents |
|------|----------|
| `results/` | Experiment checkpoints and logs |
| `logs/` | Training logs |
| `outputs/` | Hydra outputs |
| `plots/` | Generated plots |
| `eval_gifs/` | Evaluation visualizations |
| `pkl_files/` | Trajectory pickle files |
| `data_info/` | Dataset metadata |
| `perturbation_analysis/` | Robustness analysis results |

---

## 🔑 Key Code Patterns

### Loading a Checkpoint
```python
# In evaluate_traj.py or other scripts
with open('checkpoint.pkl', 'rb') as f:
    checkpoint_data = pickle.load(f)
agent = create_agent(...)
agent = agent.replace(state=checkpoint_data['state'])
```

### Creating an Agent
```python
# In train.py
from jaxrl_m import agents
agent = agents[config.agent_name].create(
    rng=rng,
    observations=sample_obs,
    actions=sample_actions,
    **config.agent_kwargs
)
```

### Environment Loop
```python
obs = env.reset()
for step in range(max_steps):
    action = agent.sample_actions(obs)
    obs, reward, done, info = env.step(action)
    if done:
        break
```

### Replay Buffer Sampling
```python
batch = replay_buffer.sample(batch_size)
# batch contains: observations, actions, rewards, next_observations, dones
agent, update_info = agent.update(batch)
```

---

## 🔍 Finding Specific Functionality

| Want to... | Look in... |
|------------|------------|
| Add a new RL algorithm | `jaxrl_m/agents/continuous/` |
| Change the network architecture | `jaxrl_m/networks/` |
| Add a new vision encoder | `jaxrl_m/vision/` |
| Modify data augmentations | `jaxrl_m/data/tf_augmentations.py` |
| Add a new environment | `jaxrl_m/envs/` |
| Change training hyperparameters | `configs/*.py` |
| Modify the training loop | `train.py`, `train_pi.py`, `train_pi_residual.py` |
| Add evaluation metrics | `jaxrl_m/common/evaluation.py` |
| Change logging | `jaxrl_m/common/wandb.py` |
| Add environment wrappers | `jaxrl_m/envs/wrappers/` |

---

## 📖 Reading Order for New Contributors

1. **Start with README.md** - Understand project goals and setup
2. **configs/base_config.py** - See configuration structure
3. **train.py** - Main training loop (first 200 lines)
4. **jaxrl_m/agents/continuous/base_policy.py** - Agent interface
5. **jaxrl_m/agents/continuous/bc.py** - Simple agent example
6. **jaxrl_m/agents/continuous/parl_calql.py** - PA-RL implementation
7. **jaxrl_m/common/evaluation.py** - How evaluation works
8. **jaxrl_m/data/image_replay_buffer.py** - Data handling

---

*Last updated: 2026-01-28*
