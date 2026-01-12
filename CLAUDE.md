# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LearningHumanoidWalking is a PyTorch-based deep reinforcement learning system for training humanoid robots to walk using MuJoCo physics simulation and PPO (Proximal Policy Optimization).

## Commands

### Package Management (uv)
```bash
uv sync                    # Install dependencies from uv.lock
uv add <package>           # Add a new dependency
uv lock                    # Update lock file after pyproject.toml changes
```

### Training
```bash
python run_experiment.py train --logdir <path_to_exp_dir> --num_procs <num_cpu_procs> --env <environment_name>
```

### Evaluation
```bash
python run_experiment.py eval --logdir <path_to_actor_pt>
```

### Environment Names
- `h1` - Basic standing task (H1 robot)
- `jvrc_walk` - Basic walking task (JVRC robot)
- `jvrc_step` - Stepping task using planned footsteps (JVRC robot)

### Key Training Hyperparameters
- `--n-itr`: Training iterations (default: 20000)
- `--lr`: Learning rate (default: 1e-4)
- `--gamma`: MDP discount (default: 0.99)
- `--lam`: GAE discount (default: 0.95)
- `--clip`: PPO clipping (default: 0.2)
- `--epochs`: Optimization epochs per update (default: 3)
- `--max-traj-len`: Episode horizon (default: 400)

## Architecture

### Core Components

**Entry Point**: `run_experiment.py` - CLI with `train` and `eval` subcommands

**RL Framework** (`rl/`):
- `algos/ppo.py` - PPO implementation with GAE, gradient clipping, entropy regularization
- `policies/actor.py` - Gaussian FF and LSTM actor networks
- `policies/critic.py` - Feed-forward and LSTM value functions
- `storage/rollout_storage.py` - PPO trajectory buffer
- `envs/` - Wrappers for normalization and symmetry-based learning

**Environments** (`envs/`):
- `common/mujoco_env.py` - Base MuJoCo environment wrapper
- `common/robot_interface.py` - Robot-specific abstraction (motor dynamics, contacts)
- `jvrc/` and `h1/` - Robot-specific environment implementations

**Tasks** (`tasks/`):
- `walking_task.py`, `stepping_task.py` - Task specifications
- `rewards.py` - Modular reward functions (velocity tracking, torque minimization, etc.)

### Key Patterns

- **Parallel Training**: Uses Ray for distributed training across multiple CPUs
- **Symmetry Learning**: `SymmetricEnv` wrapper leverages left-right body symmetry for data augmentation
- **Observation Normalization**: Online Welford algorithm in environment wrappers
- **Motor Dynamics**: JIT-compiled PyTorch networks for realistic motor simulation

### Robot Models

Located in `models/` with git submodules:
- `jvrc_mj_description/` - JVRC humanoid
- `cassie_mj_description/` - Cassie bipedal robot
- `mujoco_menagerie/` - Google DeepMind robot collection

## Git Conventions

- Brief commit messages without automated attribution
- Main branch: `main`
- Initialize submodules: `git submodule update --init --recursive`
