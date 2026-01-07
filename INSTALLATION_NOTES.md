# Installation Notes for PolicyAgnosticRL

## Quick Start

For a clean installation with all working versions:

```bash
conda activate parl
uv pip install -r requirements_clean.txt
```

## Known Issues and Manual Fixes

### 1. egl-probe CMake Compatibility Issue

**Problem**: `egl-probe==1.0.2` (dependency of robomimic) fails to build with modern CMake (4.x) due to outdated CMakeLists.txt requiring CMake 2.8.12.

**Solution**: Patch and install manually:

```bash
# Download source
pip download egl-probe==1.0.2 --no-binary :all:

# Extract
tar -xzf egl_probe-1.0.2.tar.gz
cd egl_probe-1.0.2

# Patch CMakeLists.txt
sed -i 's/cmake_minimum_required(VERSION 2.8.12)/cmake_minimum_required(VERSION 3.5)/' egl_probe/CMakeLists.txt

# Install
pip install .
cd ..
rm -rf egl_probe-1.0.2*
```

**Automated workaround**: If using `requirements_clean.txt` with uv/pip and it fails on egl-probe, run the above commands manually, then rerun the installation.

### 2. JAX/Flax Version Compatibility

**Problem**: The original requirements specified JAX 0.5.3 and Flax 0.8.0, which are incompatible due to removed APIs in JAX 0.5.x.

**Solution**: Use JAX 0.4.20 with Flax 0.8.0 (already specified in requirements_clean.txt)

### 3. NumPy 2.0 Incompatibility

**Problem**: JAX 0.4.x requires NumPy 1.x, but newer packages default to NumPy 2.x.

**Solution**: Pin numpy<2.0 (already done in requirements_clean.txt)

### 4. MuJoCo 210 Binary Required for D4RL

**Problem**: D4RL expects MuJoCo 210 binary at `~/.mujoco/mujoco210/`

**Solution**: Download and install MuJoCo 210 separately:

```bash
# Download MuJoCo 210
mkdir -p ~/.mujoco
cd ~/.mujoco
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
tar -xzf mujoco210-linux-x86_64.tar.gz
rm mujoco210-linux-x86_64.tar.gz

# Set environment variable (add to ~/.bashrc)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
```

**Note**: This is only needed if you plan to use D4RL environments. The package will import but environments requiring MuJoCo 210 won't work without this.

## Version Differences from Original requirements.txt

| Package | Original | Clean | Reason |
|---------|----------|-------|--------|
| jax[cuda12] | 0.5.3 | 0.4.20 | Flax 0.8.0 incompatibility |
| jaxlib | 0.5.3 | 0.4.20 | Match JAX version |
| numpy | (auto) | 1.26.4 | JAX 0.4.x requires numpy<2 |
| scipy | >=1.6.0 | 1.12.0 | Compatibility with JAX stack |
| orbax-checkpoint | 0.11.13 | 0.5.16 | Auto-downgraded for JAX 0.4.20 |

## Verification

After installation, verify key packages:

```bash
python -c "
import gym, jax, flax, tensorflow, torch
import robomimic, robosuite, wandb
print('Core packages OK')
print(f'JAX: {jax.__version__}')
print(f'Flax: {flax.__version__}')
print(f'TensorFlow: {tensorflow.__version__}')
print(f'PyTorch: {torch.__version__}')
"
```

## Installation Order (if installing manually)

If you need to install packages individually:

1. Core numerical stack: numpy, scipy
2. JAX ecosystem: jax, jaxlib, flax, distrax, chex, optax
3. TensorFlow stack: tensorflow, tensorflow-probability
4. RL environments: gym, mujoco, dm_control, robosuite
5. Robotics: robomimic (with manual egl-probe fix if needed)
6. Everything else: transformers, wandb, plotting tools, etc.

## Environment

- Python: 3.11.14
- CUDA: 12.x (for JAX/PyTorch GPU support)
- OS: Linux (tested on Ubuntu with kernel 6.14.0)

## Troubleshooting

### Import errors with JAX/Flax

If you see `ImportError: cannot import name 'maps' from 'jax.experimental'`, you have incompatible JAX/Flax versions. Ensure you're using JAX 0.4.20 with Flax 0.8.0.

### NumPy version conflicts

If you see numpy array API errors, downgrade to numpy 1.26.4.

### CUDA warnings

Warnings like "Unable to register cuFFT factory" are normal and can be ignored. They don't affect functionality.

### Gym deprecation warnings

Gym 0.23.1 shows deprecation warnings about numpy 2.0. This is expected and doesn't affect functionality since we use numpy 1.26.4.
