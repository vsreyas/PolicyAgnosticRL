# Development Guide

A practical guide for common development tasks in PolicyAgnosticRL.

---

## 🚀 Quick Start Workflows

### 1. Train a Diffusion Policy from Scratch

```bash
# Set environment variables
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=""

# Train on AntMaze
python ./train.py \
  --environment_name=antmaze-medium-diverse-v2 \
  --wandb_experiment_name=my_ddpm_experiment \
  --config=./configs/state_config.py:ddpm \
  --num_offline_epochs=3000 \
  --num_online_epochs=0 \
  --seed=0
```

**What happens:**
- Loads D4RL dataset
- Trains DDPM policy with behavioral cloning
- Saves checkpoints to `results/PA-RL/my_ddpm_experiment/seed_0/`
- Logs to Weights & Biases

### 2. Fine-tune with PA-RL

```bash
python ./train.py \
  --environment_name=antmaze-medium-diverse-v2 \
  --wandb_experiment_name=my_parl_experiment \
  --config=./configs/state_config.py:parl_calql \
  --config.base_policy_path=ddpm:./results/PA-RL/my_ddpm_experiment/seed_0/agent_checkpoints/checkpoint_3000/ \
  --num_offline_epochs=1000 \
  --num_online_epochs=1000 \
  --seed=0
```

**What happens:**
- Loads pre-trained DDPM policy
- Trains distributional Q-ensemble with CQL
- Performs offline RL (1000 epochs)
- Performs online fine-tuning (1000 epochs)
- Uses Q-values to select actions from diffusion samples

### 3. Train Pi0 on LIBERO

```bash
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0

python ./train_pi.py \
  --environment_name=libero \
  --wandb_experiment_name=pi0_libero_experiment \
  --config=./configs/libero_config.py:pi0 \
  --task_name="put both moka pots on the stove" \
  --num_offline_epochs=30 \
  --seed=0
```

**What happens:**
- Loads LIBERO environment and dataset
- Trains Pi0 vision-language-action policy
- Evaluates on specified task

### 4. Residual Learning on Pi0.5

**Step A: Sample base trajectories**
```bash
python ./sample_base_traj.py \
  --config=configs/libero_config.py:pi_residual_td3 \
  --environment_name=libero \
  --config.save_dir="/path/to/base_trajectories" \
  --num_trajectories_to_collect=40 \
  --pi_config_name="pi05_libero_gradacc2_2k" \
  --seed=0
```

**Step B: Train residual actor**
```bash
python ./train_pi_residual.py \
  --config=configs/libero_config.py:pi_residual_td3 \
  --environment_name=libero \
  --task_name="put both moka pots on the stove" \
  --config.libero_tfrecord_regexp="/path/to/base_trajectories/image_replay_buffer/*.tfrecord" \
  --config.save_dir="./residual_training" \
  --num_edit_samples=4 \
  --warmup_steps=50000 \
  --num_train_steps=50000 \
  --seed=0
```

**What happens:**
- Warmstarts critic using base policy trajectories
- Trains residual actor to modify base policy actions
- Residual actor learns: `a_final = a_base + Δa`

---

## 🔧 Common Development Tasks

### Task 1: Add a New Environment

**1. Create environment file**

Create `jaxrl_m/envs/my_new_env.py`:

```python
import gymnasium as gym

def make_my_env(task_name="default", **kwargs):
    """Factory function to create environment."""
    # Your environment creation logic
    env = gym.make(f"MyEnv-{task_name}-v0")
    return env
```

**2. Create dataset loader**

Create `jaxrl_m/data/my_new_env_dataset.py`:

```python
import tensorflow as tf

def make_my_env_dataset(data_path, **kwargs):
    """Load dataset from disk."""
    # Load and preprocess data
    dataset = tf.data.TFRecordDataset(data_path)
    # Apply transformations
    return dataset
```

**3. Add config**

Create `configs/my_new_env_config.py`:

```python
import ml_collections

def get_config(config_string="ddpm"):
    base_config = get_base_config()
    
    # Environment settings
    base_config.env_kwargs = {
        "task_name": "my_task",
    }
    
    # Dataset settings
    base_config.dataset_kwargs = {
        "data_path": "/path/to/data",
    }
    
    # Agent-specific configs
    if config_string == "ddpm":
        return get_ddpm_config(base_config)
    # ... more configs
    
    return base_config
```

**4. Update training script**

In `train.py`, add to environment factory:

```python
elif config.environment_name == "my_new_env":
    from jaxrl_m.envs.my_new_env import make_my_env
    env = make_my_env(**config.env_kwargs)
```

**5. Test**

```bash
python ./train.py \
  --environment_name=my_new_env \
  --config=./configs/my_new_env_config.py:ddpm \
  --num_offline_epochs=10 \
  --seed=0
```

---

### Task 2: Add a New Agent

**1. Create agent file**

Create `jaxrl_m/agents/continuous/my_agent.py`:

```python
import jax
import jax.numpy as jnp
from jaxrl_m.agents.continuous.base_policy import BasePolicy

class MyAgent(BasePolicy):
    """My custom RL agent."""
    
    @classmethod
    def create(cls, rng, observations, actions, **kwargs):
        """Factory method to create agent."""
        # Initialize networks
        # Create optimizer states
        # Return agent instance
        pass
    
    def update(self, batch):
        """Training step."""
        # Compute losses
        # Update parameters
        # Return updated agent and metrics
        pass
    
    def sample_actions(self, observations, **kwargs):
        """Sample actions from policy."""
        # Forward pass through networks
        # Return actions
        pass
```

**2. Register agent**

In `jaxrl_m/agents/__init__.py`:

```python
from jaxrl_m.agents.continuous.my_agent import MyAgent

agents = {
    "my_agent": MyAgent,
    # ... existing agents
}
```

**3. Add config**

In your config file:

```python
def get_my_agent_config(base_config):
    base_config.agent_name = "my_agent"
    base_config.agent_kwargs = {
        "learning_rate": 3e-4,
        "hidden_dims": (256, 256),
        # ... agent-specific hyperparameters
    }
    return base_config
```

**4. Test**

```bash
python ./train.py \
  --environment_name=antmaze-medium-diverse-v2 \
  --config=./configs/state_config.py:my_agent \
  --num_offline_epochs=100 \
  --seed=0
```

---

### Task 3: Add a New Vision Encoder

**1. Create encoder file**

Create `jaxrl_m/vision/my_encoder.py`:

```python
import flax.linen as nn
import jax.numpy as jnp

class MyEncoder(nn.Module):
    """My custom vision encoder."""
    
    features: int = 256
    
    @nn.compact
    def __call__(self, observations, train=False):
        # Process images
        x = observations["image"]
        
        # Your encoder architecture
        x = nn.Conv(32, (3, 3))(x)
        x = nn.relu(x)
        # ... more layers
        
        # Flatten and project
        x = x.reshape(x.shape[0], -1)
        x = nn.Dense(self.features)(x)
        
        return x
```

**2. Register encoder**

In `jaxrl_m/common/encoding.py`:

```python
from jaxrl_m.vision.my_encoder import MyEncoder

def make_encoder(encoder_type, **kwargs):
    if encoder_type == "my_encoder":
        return MyEncoder(**kwargs)
    # ... existing encoders
```

**3. Use in config**

```python
base_config.encoder_type = "my_encoder"
base_config.encoder_kwargs = {
    "features": 256,
}
```

---

### Task 4: Modify Training Loop

The main training loop is in `train.py` (or `train_pi.py`, `train_pi_residual.py`).

**Key sections:**

```python
# 1. Setup (lines ~100-300)
# - Create environment
# - Load dataset
# - Create agent

# 2. Offline training (lines ~300-400)
for epoch in range(config.num_offline_epochs):
    for batch in dataset:
        agent, update_info = agent.update(batch)
        # Log metrics
    
    # Periodic evaluation
    if epoch % config.eval_interval == 0:
        eval_metrics = evaluate(agent, env, num_episodes=10)

# 3. Online training (lines ~400-500)
for epoch in range(config.num_online_epochs):
    # Collect trajectories
    for step in range(steps_per_epoch):
        action = agent.sample_actions(obs)
        obs, reward, done, info = env.step(action)
        replay_buffer.insert(transition)
    
    # Train on replay buffer
    for batch in replay_buffer.sample_batches():
        agent, update_info = agent.update(batch)
```

**To modify:**
1. Edit the relevant section
2. Add new hyperparameters to config
3. Test thoroughly

---

### Task 5: Implement a New Reward Function

**For simulation environments:**

Edit the environment file (e.g., `jaxrl_m/envs/libero.py`):

```python
def custom_reward(obs, action, next_obs, info):
    """Custom reward function."""
    # Dense reward based on distance to goal
    distance = np.linalg.norm(next_obs["desired_goal"] - next_obs["achieved_goal"])
    dense_reward = -distance
    
    # Sparse reward on success
    sparse_reward = 1.0 if info["success"] else 0.0
    
    # Combine
    reward = 0.1 * dense_reward + sparse_reward
    return reward
```

Then wrap the environment:

```python
env = MyEnv()
env = RewardWrapper(env, reward_fn=custom_reward)
```

**For real robot:**

Edit `jaxrl_m/envs/robot_manual_reward_functions.py` and register your function.

---

### Task 6: Debug with Visualization

**Option 1: Record videos**

```python
from jaxrl_m.envs.wrappers.video_recorder import VideoRecorder

env = VideoRecorder(
    env,
    save_folder="./videos",
    save_prefix="eval",
    fps=20
)

# Videos are automatically saved after each episode
```

**Option 2: Visualize trajectories**

```bash
python visualize_traj_predictions.py \
  --tfrecord_path="/path/to/episode.tfrecord" \
  --checkpoint_path="/path/to/checkpoint.pkl" \
  --output_path="./visualization.mp4"
```

**Option 3: Dump TFRecord to video**

```bash
python dump_tf_record.py \
  --tfrecord_path="/path/to/episode.tfrecord" \
  --output_gif_path="./output.mp4"
```

---

### Task 7: Hyperparameter Tuning

**Using W&B Sweeps:**

1. Create sweep config `sweep_config.yaml`:

```yaml
program: train.py
method: bayes
metric:
  name: evaluation/success_rate
  goal: maximize
parameters:
  config.agent_kwargs.learning_rate:
    distribution: log_uniform_values
    min: 1e-5
    max: 1e-3
  config.agent_kwargs.critic_ensemble_size:
    values: [2, 5, 10]
  seed:
    values: [0, 1, 2, 3, 4]
```

2. Initialize and run sweep:

```bash
wandb sweep sweep_config.yaml
wandb agent <sweep_id>
```

**Manual grid search:**

Use a bash script:

```bash
#!/bin/bash
for lr in 1e-4 3e-4 1e-3; do
  for seed in 0 1 2; do
    python ./train.py \
      --config=./configs/state_config.py:parl_calql \
      --config.agent_kwargs.learning_rate=$lr \
      --seed=$seed
  done
done
```

---

### Task 8: Multi-GPU/TPU Training

**For multi-GPU (JAX handles automatically):**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./train.py \
  --config=./configs/libero_config.py:ddpm \
  --num_offline_epochs=100
```

JAX will automatically use all visible GPUs.

**For TPU:**

```python
# Install TPU support
pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Run training
python ./train.py \
  --config=./configs/libero_config.py:ddpm \
  --num_offline_epochs=100
```

---

### Task 9: Checkpoint Management

**Save checkpoint manually:**

```python
import pickle

checkpoint_data = {
    'state': agent.state,
    'config': config.to_dict(),
    'step': step,
}

with open('checkpoint.pkl', 'wb') as f:
    pickle.dump(checkpoint_data, f)
```

**Load checkpoint:**

```python
with open('checkpoint.pkl', 'rb') as f:
    checkpoint_data = pickle.load(f)

agent = agent.replace(state=checkpoint_data['state'])
```

**Resume training:**

```bash
python ./train.py \
  --config=./configs/state_config.py:ddpm \
  --resume_path=./results/experiment/seed_0/agent_checkpoints/checkpoint_1000.pkl
```

---

### Task 10: Profile Performance

**JAX profiling:**

```python
import jax.profiler

# In training loop
with jax.profiler.trace("/tmp/jax-trace", create_perfetto_trace=True):
    for _ in range(100):
        agent, update_info = agent.update(batch)
```

View trace at `chrome://tracing`.

**Time specific operations:**

```python
from jaxrl_m.utils.timer_utils import Timer

timer = Timer()

with timer("agent_update"):
    agent, update_info = agent.update(batch)

with timer("env_step"):
    obs, reward, done, info = env.step(action)

print(timer.get_average_times())
```

---

## 🧪 Testing Checklist

Before committing changes:

- [ ] Run linter: `flake8 jaxrl_m/`
- [ ] Check types: `mypy jaxrl_m/` (if using type hints)
- [ ] Test on small config (10 epochs):
  ```bash
  python ./train.py --config=... --num_offline_epochs=10 --seed=0
  ```
- [ ] Check W&B logging (all metrics appear correctly)
- [ ] Test evaluation runs without errors
- [ ] Test checkpoint save/load
- [ ] Run on multiple seeds (0, 1, 2)
- [ ] Document changes in commit message

---

## 🐛 Common Issues and Solutions

### Issue 1: Out of Memory (OOM)

**Solutions:**
1. Reduce batch size: `--config.batch_size=128`
2. Reduce critic ensemble: `--config.agent_kwargs.critic_ensemble_size=2`
3. Reduce UTD ratio: `--config.utd_ratio=1`
4. Enable memory preallocation: `export XLA_PYTHON_CLIENT_PREALLOCATE=true`
5. Use gradient accumulation (if implemented)

### Issue 2: NaN losses

**Debug steps:**
1. Check learning rates (reduce if too high)
2. Enable gradient clipping: `--config.agent_kwargs.max_grad_norm=10.0`
3. Check reward scaling: `--reward_scale=1.0 --reward_bias=0.0`
4. Inspect batch statistics (print min/max values)
5. Use mixed precision carefully

### Issue 3: Poor performance

**Debug steps:**
1. Verify environment is correct (render a few episodes)
2. Check dataset quality (visualize trajectories)
3. Verify reward function is correct
4. Try standard hyperparameters first
5. Increase training duration
6. Check evaluation code (sometimes eval is buggy, not training)

### Issue 4: Slow training

**Solutions:**
1. Profile to find bottlenecks
2. Use faster encoder (IMPALA instead of ResNet)
3. Reduce image size: `--config.image_size=64`
4. Reduce augmentation overhead
5. Use XLA compilation: JAX does this automatically
6. Check data loading (might be CPU bottleneck)

### Issue 5: Environment not rendering

**For EGL (headless):**
```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
```

**For Xvfb (virtual display):**
```bash
xvfb-run -a python ./train.py ...
```

---

## 📚 Learning Resources

### Understanding the Codebase
1. Start with `../README.md`
2. Read `ARCHITECTURE_GUIDE.md` (architecture overview)
3. Read `FILE_REFERENCE.md` (what each file does)
4. Study a simple agent: `jaxrl_m/agents/continuous/bc.py`
5. Study the training loop: `train.py` (first 300 lines)

### JAX + Flax
- [JAX Tutorial](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html)
- [Flax Documentation](https://flax.readthedocs.io/)
- [JAX for RL](https://github.com/google/jax/discussions/8189)

### RL Algorithms
- PA-RL paper: https://arxiv.org/abs/2412.06685
- CQL paper: https://arxiv.org/abs/2006.04779
- IQL paper: https://arxiv.org/abs/2110.06169
- Diffusion Policy: https://arxiv.org/abs/2303.04137

---

*Last updated: 2026-01-28*
