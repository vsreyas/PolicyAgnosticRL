# PolicyAgnosticRL Architecture Guide

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Scripts Layer                    │
│  train.py | train_pi.py | train_pi_residual.py              │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    Configuration Layer                       │
│  configs/{state,libero,calvin,bridgedata,real}_config.py    │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                      jaxrl_m Package                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Agents   │  │ Networks │  │ Envs     │  │ Data     │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Vision   │  │ Trans-   │  │ Utils    │  │ Common   │   │
│  │          │  │ formers  │  │          │  │          │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Interaction Flow

### 1. Training Pipeline

```
Configuration File
       ↓
Training Script (train.py / train_pi.py / train_pi_residual.py)
       ↓
   ┌───────────────┐
   │ Agent Factory │ ← Creates agent based on config
   └───────┬───────┘
           ↓
   ┌───────────────────────────────────┐
   │        Agent (e.g., PARL)         │
   │  ┌─────────────┐  ┌─────────────┐│
   │  │ Base Policy │  │   Critic    ││
   │  └─────────────┘  └─────────────┘│
   └───────────────────────────────────┘
           ↓
   ┌───────────────┐
   │  Environment  │ ← Wrapped with various wrappers
   └───────────────┘
           ↓
   ┌───────────────┐
   │ Replay Buffer │ ← Stores trajectories
   └───────────────┘
```

### 2. Base Policy Types

```
Base Policy Interface (base_policy.py)
         │
         ├─── DDPM Policy (ddpm_bc.py)
         │    └─── Diffusion network (diffusion_nets.py)
         │
         ├─── Pi0 Policy (pi_0.py)
         │    └─── Vision encoder + Transformer
         │
         ├─── OpenVLA Policy (openvla.py)
         │    └─── External VLA model
         │
         ├─── Transformer Policy (auto_regressive_transformer.py)
         │    └─── Q-Transformer (transformers/models/)
         │
         └─── Standard RL (sac.py, cql.py, etc.)
              └─── Actor-Critic networks
```

### 3. Policy Agnostic RL Wrapper

```
┌──────────────────────────────────────────┐
│         PARL Agent (parl_calql.py)       │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │      Base Policy (Frozen/FT)       │ │ ← Any policy type
│  │  (DDPM / Pi0 / OpenVLA / etc.)     │ │
│  └────────────────┬───────────────────┘ │
│                   │ Actions             │
│                   ↓                     │
│  ┌────────────────────────────────────┐ │
│  │    Distributional Q-Ensemble       │ │ ← Learns Q-values
│  │    (10 critics, CQL loss)          │ │
│  └────────────────────────────────────┘ │
│                   │                     │
│                   ↓                     │
│  ┌────────────────────────────────────┐ │
│  │   Action Selection Strategy        │ │
│  │   - Max Q-value                    │ │
│  │   - Mean                           │ │
│  │   - Uncertainty-based              │ │
│  └────────────────────────────────────┘ │
└──────────────────────────────────────────┘
```

### 4. Residual Learning Architecture

```
┌──────────────────────────────────────────┐
│   Residual Agent (residual_td3.py)      │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │   Base Policy (Pi0.5) [Frozen]    │ │ ← Pre-trained
│  │   Outputs: a_base                  │ │
│  └────────────────┬───────────────────┘ │
│                   │                     │
│                   ↓                     │
│  ┌────────────────────────────────────┐ │
│  │      Residual Actor Network        │ │ ← Learnable
│  │      Outputs: Δa (delta action)    │ │
│  └────────────────┬───────────────────┘ │
│                   │                     │
│                   ↓                     │
│           a_final = a_base + Δa        │
│                   │                     │
│                   ↓                     │
│  ┌────────────────────────────────────┐ │
│  │      TD3/PPO Critic Network        │ │ ← Learns Q(s,a)
│  │      Trained with RL objectives    │ │
│  └────────────────────────────────────┘ │
└──────────────────────────────────────────┘
```

---

## Module Dependencies

### Agent Dependencies

```
agents/continuous/*.py
    ↓ depends on
networks/{actor_critic_nets, diffusion_nets, mlp}.py
    ↓ depends on
vision/{resnet_v1, impala, small_encoders}.py
    ↓ depends on
common/{encoding, initialization, optimizers}.py
```

### Data Pipeline Dependencies

```
Training Script
    ↓ uses
data/{libero_dataset, calvin_dataset, bridge_dataset}.py
    ↓ creates
data/image_replay_buffer.py
    ↓ uses
data/{tf_augmentations, text_processing}.py
    ↓ outputs
TFRecord files
```

### Environment Pipeline

```
Training Script
    ↓ creates
envs/{libero, calvin, d4rl, robot_env}.py
    ↓ wrapped by
envs/wrappers/{video_recorder, reward_scale, chunking}.py
    ↓ interfaces with
External Simulators (MuJoCo, PyBullet)
```

---

## Key Design Patterns

### 1. Configuration-Driven Design
- All experiments configured via `ml_collections.ConfigDict`
- Configs defined in `configs/*.py`
- Command-line flags override config values
- Enables reproducibility and hyperparameter sweeps

### 2. Factory Pattern for Agents
```python
# In train.py
agent = agents[config.agent_name].create(
    rng=rng,
    observations=observations,
    actions=actions,
    **config.agent_kwargs
)
```

### 3. Environment Wrapper Composition
```python
env = Environment()
env = RewardScaleWrapper(env, scale=config.reward_scale)
env = VideoRecorderWrapper(env, save_dir=...)
env = ChunkingWrapper(env, action_horizon=...)
```

### 4. Modular Policy Design
- Base policies implement common interface
- Policies can be loaded from checkpoints
- Policies can be frozen or fine-tuned
- Enables policy-agnostic RL

---

## Data Flow

### Training Step

```
1. Sample batch from replay buffer
   └─ image_replay_buffer.sample() → observations, actions, rewards

2. Agent update
   └─ agent.update(batch) → {
        'critic_loss': ...,
        'actor_loss': ...,
        'q_values': ...
      }

3. Environment interaction (online RL)
   └─ base_policy.sample_actions(obs) → actions
   └─ critic.select_best_action(actions, obs) → selected_action
   └─ env.step(selected_action) → next_obs, reward, done

4. Store transition
   └─ replay_buffer.insert(obs, action, reward, next_obs, done)

5. Log metrics
   └─ wandb.log(metrics)
```

### Evaluation Step

```
1. Reset environment
   └─ obs = env.reset()

2. Rollout loop
   for t in range(max_steps):
       └─ action = policy.sample_actions(obs, eval_mode=True)
       └─ obs, reward, done, info = env.step(action)
       └─ Accumulate rewards and success flags

3. Aggregate metrics
   └─ success_rate = sum(successes) / num_episodes
   └─ average_return = sum(returns) / num_episodes

4. Save videos/gifs
   └─ video_wrapper.save_video()
```

---

## Key Abstractions

### Agent Interface
All agents implement:
- `create()` - Factory method
- `update()` - Training step
- `sample_actions()` - Action sampling
- `get_state()` / `set_state()` - Checkpointing

### Environment Interface
All environments provide:
- `reset()` - Initialize episode
- `step(action)` - Take action
- `observation_space` - Observation spec
- `action_space` - Action spec

### Replay Buffer Interface
All buffers provide:
- `insert()` - Add transition
- `sample()` - Sample batch
- `get_dataset()` - Get full dataset (for offline RL)

---

## Extension Points

### Adding a New Environment
1. Create `envs/my_env.py` with Gym interface
2. Add config in `configs/my_env_config.py`
3. Register in `train.py` environment factory
4. Add dataset class in `data/my_env_dataset.py`

### Adding a New Agent
1. Create `agents/continuous/my_agent.py`
2. Implement agent interface methods
3. Define agent config in appropriate config file
4. Register in `agents/__init__.py`

### Adding a New Base Policy
1. Implement policy in `agents/continuous/my_policy.py`
2. Inherit from base policy interface
3. Add loading logic for checkpoints
4. Configure in PARL agent

### Adding a New Vision Encoder
1. Create `vision/my_encoder.py`
2. Implement `__call__` method
3. Register in config's `encoder_type` options
4. Update `encoding.py` to use new encoder

---

## File Size and Complexity

### Large/Complex Files (>500 lines)
- `train.py` (~800 lines) - Main training loop
- `train_pi_residual.py` (~700 lines) - Residual training
- `evaluate_traj.py` (~750 lines) - Evaluation
- `agents/continuous/parl_calql.py` - PARL implementation
- `data/image_replay_buffer.py` - Replay buffer with images

### Core Interface Files
- `agents/continuous/base_policy.py` - Base policy interface
- `common/common.py` - Common utilities
- `networks/actor_critic_nets.py` - Network definitions

---

## Common Workflows Summary

| Workflow | Entry Point | Key Components | Output |
|----------|-------------|----------------|--------|
| BC Pretraining | `train.py` | `bc.py`, `ddpm_bc.py` | Base policy checkpoint |
| PA-RL Training | `train.py` | `parl_calql.py`, base policy | Fine-tuned policy + critic |
| Pi0 Training | `train_pi.py` | `pi_0.py`, `pi0_train_utils.py` | Pi0 checkpoint |
| Residual Learning | `train_pi_residual.py` | `residual_td3.py`, `residual_ppo.py` | Residual actor + critic |
| Evaluation | `evaluate_traj.py` | Agent checkpoint, environment | Metrics + videos |
| Data Collection | `sample_base_traj.py` | Base policy, environment | TFRecord trajectories |
| Visualization | `visualize_traj_predictions.py` | TFRecord, checkpoint | MP4 visualization |

---

*Last updated: 2026-01-28*
