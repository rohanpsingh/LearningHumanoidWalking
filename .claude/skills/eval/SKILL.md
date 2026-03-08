---
name: eval
description: Evaluate a trained checkpoint with visualization
disable-model-invocation: true
allowed-tools: Bash, Read, Glob
argument-hint: [path-or-logdir]
---

# /eval — Evaluate a Trained Checkpoint

Parse the user's request from `$ARGUMENTS` and run evaluation.

## Command Template

```
uv run python run_experiment.py eval --path <PATH> [OPTIONS...]
```

## Path Resolution

The user may provide:
- **A .pt file**: Use directly (`--path /tmp/.../actor_999.pt`)
- **A run directory**: Contains actor*.pt files (`--path /tmp/.../26-03-07-00-26-36_cartpole/`)
- **A logdir**: Contains timestamped run subdirectories (`--logdir /tmp/training_runs`)

If no path is given, check `/tmp/training_runs` for the most recent run.

Use `Glob` to verify the path exists and resolve it before running.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--ep-len` | 10 | Episode length in seconds |
| `--seed` | None | Random seed for reproducible eval |
| `--out-dir` | None | Directory to save videos |

## Instructions

1. Resolve the model path from the user's input. If ambiguous, list available checkpoints and ask.
2. Show the user which checkpoint will be evaluated (full path).
3. Run the eval command. This opens an interactive MuJoCo viewer window — it is NOT a background job.
4. Report the results when done.
