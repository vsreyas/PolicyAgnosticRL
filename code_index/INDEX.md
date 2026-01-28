# PolicyAgnosticRL - Master Index

Welcome to the PolicyAgnosticRL codebase! This index will help you navigate the documentation and understand the structure of the project.

---

## 📚 Documentation Overview

This codebase is documented through several interconnected guides. Choose the one that best fits your needs:

### 🗺️ [CODEBASE_INDEX.md](./CODEBASE_INDEX.md)
**Comprehensive directory and file structure**

Use this when you want to:
- Understand the overall organization of the codebase
- Find where specific functionality lives
- See what files and directories exist
- Learn about supported environments and policy types
- Find example training commands

**Best for:** Getting oriented, finding files, understanding scope

---

### 🏗️ [ARCHITECTURE_GUIDE.md](./ARCHITECTURE_GUIDE.md)
**System architecture and design patterns**

Use this when you want to:
- Understand how components interact
- Learn the data flow through the system
- See the agent architecture (base policy + critic)
- Understand module dependencies
- Learn about design patterns used

**Best for:** Understanding how the system works, making architectural changes

---

### 📖 [FILE_REFERENCE.md](./FILE_REFERENCE.md)
**Quick reference for every major file**

Use this when you want to:
- Quickly look up what a specific file does
- Find which file implements specific functionality
- Understand file organization at a glance
- Get reading recommendations for learning the codebase

**Best for:** Quick lookups, finding specific implementations

---

### 🛠️ [DEVELOPMENT_GUIDE.md](./DEVELOPMENT_GUIDE.md)
**Practical guide for common tasks**

Use this when you want to:
- Train models (quick start commands)
- Add new environments, agents, or encoders
- Debug and visualize results
- Tune hyperparameters
- Solve common issues

**Best for:** Hands-on development, implementing features, debugging

---

## 🚀 Quick Start Paths

### I want to run experiments
1. Read the [README.md](../README.md) for setup instructions
2. Jump to "Quick Start Workflows" in [DEVELOPMENT_GUIDE.md](./DEVELOPMENT_GUIDE.md)
3. Run example commands and modify as needed

### I want to understand the code
1. Read [CODEBASE_INDEX.md](./CODEBASE_INDEX.md) for overview
2. Read [ARCHITECTURE_GUIDE.md](./ARCHITECTURE_GUIDE.md) for system design
3. Follow "Reading Order for New Contributors" in [FILE_REFERENCE.md](./FILE_REFERENCE.md)
4. Read specific files mentioned

### I want to add a new feature
1. Check [CODEBASE_INDEX.md](./CODEBASE_INDEX.md) to find similar existing features
2. Read relevant "Common Development Tasks" in [DEVELOPMENT_GUIDE.md](./DEVELOPMENT_GUIDE.md)
3. Use [FILE_REFERENCE.md](./FILE_REFERENCE.md) to find where to add code
4. Refer to [ARCHITECTURE_GUIDE.md](./ARCHITECTURE_GUIDE.md) for design patterns

### I have a problem
1. Check "Common Issues and Solutions" in [DEVELOPMENT_GUIDE.md](./DEVELOPMENT_GUIDE.md)
2. Use [FILE_REFERENCE.md](./FILE_REFERENCE.md) to find relevant debugging files
3. Check visualization tools in [CODEBASE_INDEX.md](./CODEBASE_INDEX.md)

---

## 🎯 Project Overview

**Policy Agnostic RL (PA-RL)** is a framework for offline RL and online fine-tuning that works with any policy class and backbone.

### Key Capabilities

**Policy Types Supported:**
- Diffusion Policies (DDPM)
- Vision-Language-Action models (Pi0, Pi0.5, OpenVLA)
- Transformer policies
- Standard RL policies (SAC, TD3, PPO)

**Training Modes:**
- Behavioral Cloning (imitation learning)
- Offline RL (learning from fixed datasets)
- Online Fine-Tuning (environment interaction)
- Residual Learning (learning corrections to base policies)

**Environments:**
- **Simulation:** D4RL (AntMaze), LIBERO, CALVIN, Meta-World, PointMaze
- **Real Robot:** Franka Panda arm with Bridge Data Robot integration

### Core Innovation

PA-RL trains a **distributional Q-ensemble** on top of any base policy:
1. Base policy generates action candidates
2. Q-ensemble evaluates each candidate
3. Best action (by Q-value) is selected
4. Works with frozen OR fine-tuned base policies

This enables fine-tuning large pre-trained models (like OpenVLA) efficiently!

---

## 📁 Directory Structure (High-Level)

```
PolicyAgnosticRL/
├── 📄 README.md (setup and basic usage)
├── 📄 INSTALLATION_NOTES.md
│
├── 📁 code_index/ (documentation)
│   ├── INDEX.md (you are here)
│   ├── CODEBASE_INDEX.md
│   ├── ARCHITECTURE_GUIDE.md
│   ├── FILE_REFERENCE.md
│   └── DEVELOPMENT_GUIDE.md
│
├── 🐍 train.py (main training script)
├── 🐍 train_pi.py (Pi0/Pi0.5 training)
├── 🐍 train_pi_residual.py (residual learning)
├── 🐍 evaluate_traj.py (evaluation)
│
├── 📁 configs/ (experiment configurations)
│   ├── base_config.py
│   ├── state_config.py (D4RL, AntMaze)
│   ├── libero_config.py (LIBERO)
│   ├── calvin_config.py (CALVIN)
│   ├── bridgedata_config.py (real robot)
│   └── real_config.py (real robot deployment)
│
├── 📁 jaxrl_m/ (core library)
│   ├── 📁 agents/ (RL algorithms)
│   │   └── continuous/ (continuous action agents)
│   │       ├── parl_calql.py (PA-RL main agent)
│   │       ├── bc.py, ddpm_bc.py, pi_0.py
│   │       └── pi_vlm_cached/ (residual learning)
│   ├── 📁 networks/ (neural networks)
│   ├── 📁 vision/ (vision encoders)
│   ├── 📁 data/ (datasets and replay buffers)
│   ├── 📁 envs/ (environment wrappers)
│   ├── 📁 transformers/ (transformer models)
│   ├── 📁 common/ (utilities)
│   └── 📁 utils/ (helper functions)
│
├── 📁 scripts/ (helper scripts)
├── 📁 results/ (experiment outputs)
├── 📁 logs/ (training logs)
└── 📁 data_info/ (dataset metadata)
```

---

## 🔑 Key Concepts

### Agent Types

| Agent | Description | Use Case |
|-------|-------------|----------|
| **BC** | Behavioral Cloning | Imitation learning baseline |
| **DDPM** | Diffusion Policy | High-quality trajectory modeling |
| **CQL/CalQL** | Conservative Q-Learning | Offline RL on fixed datasets |
| **IQL** | Implicit Q-Learning | Offline RL without explicit policy |
| **PARL** | Policy Agnostic RL | Fine-tune any policy with Q-ensemble |
| **Residual** | Residual Learning | Learn corrections to base policy |

### Training Phases

1. **Offline Pre-training**: Learn from fixed dataset
2. **Online Fine-tuning**: Interact with environment
3. **Evaluation**: Measure success rate and returns

### Base Policy + Critic Architecture

```
Observation → Base Policy → N action samples
                                    ↓
                            Q-Ensemble (10 critics)
                                    ↓
                            Select best action
                                    ↓
                              Environment
```

---

## 🎓 Learning Path

### Beginner (Day 1)
1. ✅ Read [README.md](../README.md) - Setup environment
2. ✅ Skim [CODEBASE_INDEX.md](./CODEBASE_INDEX.md) - See what exists
3. ✅ Run first example from [DEVELOPMENT_GUIDE.md](./DEVELOPMENT_GUIDE.md)
4. ✅ Visualize a trajectory with `visualize_traj_predictions.py`

### Intermediate (Week 1)
1. ✅ Read [ARCHITECTURE_GUIDE.md](./ARCHITECTURE_GUIDE.md) - Understand system
2. ✅ Study `jaxrl_m/agents/continuous/bc.py` - Simple agent
3. ✅ Study `train.py` (first 300 lines) - Training loop
4. ✅ Modify hyperparameters and re-run experiments
5. ✅ Add a custom reward function

### Advanced (Month 1)
1. ✅ Read [FILE_REFERENCE.md](./FILE_REFERENCE.md) - Know where everything is
2. ✅ Study `jaxrl_m/agents/continuous/parl_calql.py` - PA-RL agent
3. ✅ Implement a new environment (follow [DEVELOPMENT_GUIDE.md](./DEVELOPMENT_GUIDE.md))
4. ✅ Implement a new vision encoder
5. ✅ Modify the training loop for your use case

### Expert (Month 3+)
1. ✅ Implement a new RL algorithm
2. ✅ Contribute to the codebase
3. ✅ Run experiments on real robots
4. ✅ Publish results!

---

## 🔍 Finding What You Need

Use this decision tree:

```
What do you want to know?
│
├─ "Where is X implemented?"
│  └─ Check FILE_REFERENCE.md → Search for X
│
├─ "How does X work?"
│  └─ Check ARCHITECTURE_GUIDE.md → Find X's data flow
│
├─ "How do I do X?"
│  └─ Check DEVELOPMENT_GUIDE.md → Find task for X
│
├─ "What files are in directory Y?"
│  └─ Check CODEBASE_INDEX.md → Find directory Y
│
└─ "How do I run experiment Z?"
   └─ Check ../README.md or DEVELOPMENT_GUIDE.md → Training workflows
```

---

## 📊 File Size Guide

**Small files (<200 lines):** Easy to read entirely
- `setup.py`, `noop.py`, `filter_pkl.py`

**Medium files (200-500 lines):** Read key functions
- Most agent files, network files, environment files

**Large files (>500 lines):** Navigate by section
- `train.py`, `train_pi_residual.py`, `evaluate_traj.py`
- Use documentation and comments to navigate

---

## 🤝 Contributing

Before contributing:
1. Read [ARCHITECTURE_GUIDE.md](./ARCHITECTURE_GUIDE.md) to understand design patterns
2. Check [DEVELOPMENT_GUIDE.md](./DEVELOPMENT_GUIDE.md) for testing checklist
3. Follow existing code style (see similar files)
4. Add documentation for new features
5. Test thoroughly on multiple seeds

---

## 📞 Getting Help

1. **Documentation:** Check the 4 guides above
2. **Code examples:** Look at existing implementations in codebase
3. **Issues:** Search GitHub issues (if applicable)
4. **Contact:** maxsobolmark at cmu dot edu (from README)

---

## 🔗 External Resources

- **Paper:** https://arxiv.org/abs/2412.06685
- **JAX:** https://jax.readthedocs.io/
- **Flax:** https://flax.readthedocs.io/
- **Gymnasium:** https://gymnasium.farama.org/
- **LIBERO:** https://github.com/Lifelong-Robot-Learning/LIBERO
- **CALVIN:** https://github.com/mees/calvin
- **Bridge Data Robot:** https://github.com/MaxSobolMark/bridge_data_robot

---

## 📝 Index Summary

| Document | Purpose | When to Use |
|----------|---------|-------------|
| **INDEX.md** (this file) | Navigation and overview | Starting point, getting oriented |
| **CODEBASE_INDEX.md** | Complete file/directory listing | Finding files, understanding scope |
| **ARCHITECTURE_GUIDE.md** | System design and patterns | Understanding architecture |
| **FILE_REFERENCE.md** | File descriptions | Quick lookups |
| **DEVELOPMENT_GUIDE.md** | Practical tasks | Implementation, debugging |
| **../README.md** | Setup and basic usage | Installation, first run |

---

## ✅ Index Status

- ✅ Directory structure indexed
- ✅ All Python files catalogued
- ✅ Architecture documented
- ✅ Development workflows documented
- ✅ Cross-references created
- ✅ Navigation guide created

**Index Version:** 1.0  
**Last Updated:** 2026-01-28  
**Total Files Indexed:** 129 Python files + configs + scripts  

---

*Happy coding! 🚀*
