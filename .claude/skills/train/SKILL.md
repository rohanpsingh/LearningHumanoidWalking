---
name: train
description: Launch a training run for a robot environment using PPO
disable-model-invocation: true
allowed-tools: Bash, Read, Glob
argument-hint: [env-name] [hyperparams...]
---

# /train — Launch a PPO Training Run

Parse the user's request from `$ARGUMENTS` and construct a training command.

## Command Template

```
RAY_ADDRESS= uv run python run_experiment.py train --env <ENV> --logdir <LOGDIR> [OPTIONS...]
```

## Available Environments

| Name | Description |
|------|-------------|
| `cartpole` | Cartpole swing-up (simplest, good for testing) |
| `h1` | Unitree H1 standing task |
| `jvrc_walk` | JVRC humanoid basic walking |
| `jvrc_step` | JVRC humanoid stepping with planned footsteps |

## Hyperparameters (defaults)

| Flag | Default | Description |
|------|---------|-------------|
| `--n-itr` | 20000 | Training iterations |
| `--lr` | 1e-4 | Learning rate |
| `--gamma` | 0.99 | Discount factor |
| `--std-dev` | 0.223 | Action noise |
| `--learn-std` | off | Learn action noise (flag) |
| `--entropy-coeff` | 0.0 | Entropy regularization |
| `--clip` | 0.2 | PPO clipping |
| `--minibatch-size` | 64 | Minibatch size |
| `--epochs` | 3 | Optimization epochs per update |
| `--num-procs` | 12 | Parallel workers |
| `--num-envs-per-worker` | 1 | Vectorized envs per worker |
| `--max-grad-norm` | 0.05 | Gradient clipping |
| `--max-traj-len` | 400 | Episode horizon |
| `--eval-freq` | 100 | Eval every N iterations |
| `--seed` | None | Random seed |
| `--device` | auto | Training device (auto/cpu/cuda) |
| `--no-mirror` | off | Disable symmetry wrapper (flag) |
| `--recurrent` | off | Use LSTM policy (flag) |
| `--continued` | None | Path to pretrained weights |

## Instructions

1. Determine the environment name from the user's request. If ambiguous, ask.
2. Use `--logdir /tmp/training_runs` unless the user specifies a different path.
3. Only include flags that differ from defaults — keep the command clean.
4. Show the user the full command you're about to run.
5. Run the command in the background using `run_in_background: true` on the Bash tool. Set a generous timeout (600000ms).
6. After launching, tell the user the logdir path and how to check progress (you can tail the output using the task ID).
7. If the user asks to check on training, use `TaskOutput` with `block: false` to check the latest output.

## Cartpole-Specific Defaults

For cartpole, these settings are known to work well with the current defaults
(`--lr 3e-4 --max-grad-norm 0.5 --lam 0.95 --gamma 0.99`):
- `--minibatch-size 256`
- `--std-dev 0.15 --learn-std --entropy-coeff 0.01`
- `--max-traj-len 500 --n-itr 500 --num-procs 12`
- `--no-mirror` (cartpole has no body symmetry)

Suggest these defaults when the user trains cartpole, but let them override.
