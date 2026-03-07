---
name: experiment
description: Plan and run a series of training experiments, then compare results
disable-model-invocation: true
allowed-tools: Bash, Read, Glob, Grep
argument-hint: [experiment description]
---

# /experiment — Run and Compare Training Experiments

Plan, execute, and analyze a series of training runs based on the user's experiment description in `$ARGUMENTS`.

## Workflow

### Phase 1: Plan

1. Parse the user's experiment goal and identify what variables to sweep.
2. Design a set of training runs, each with a clear name and description of what it tests.
3. Present the plan as a numbered table:
   ```
   | Run | Name | Key Changes | Command |
   ```
4. **Wait for user approval before running anything.**

### Phase 2: Execute

**CRITICAL: Run training jobs SEQUENTIALLY, one at a time. NEVER run jobs in parallel — the machine is compute-limited and parallel training will degrade performance for all runs.**

For each run:
1. Announce which run is starting (e.g., "Starting Run 2/4: high_gamma").
2. Construct the training command following the `/train` skill conventions:
   - Use `RAY_ADDRESS= uv run python run_experiment.py train --env <ENV> ...`
   - Use `--logdir /tmp/experiments/<experiment_name>/<run_name>` for organized output
3. Run the command in the **foreground** (do NOT use `run_in_background`). Use a generous timeout (600000ms / 10 min).
4. After the run completes, immediately parse its logs (see Phase 3 below).
5. Give a brief status update before starting the next run.

### Phase 3: Parse Logs

After each run completes, extract these metrics from the training stdout:

**Per-iteration metrics** (from the table printed each iteration):
- `Mean Eprew` — episode reward
- `Mean Eplen` — episode length
- `Actor loss`, `Critic loss`
- `Mean KL Div` — policy divergence
- `Mean Entropy` — exploration
- `Clip Fraction` — PPO clipping rate
- `Mean noise std` — action noise

**Summary metrics** (from eval and timing lines):
- `fps` — frames per second
- Eval reward and episode length
- Total training time

**Anomaly detection** — flag these issues:
- `nan` or `inf` in any metric
- Critic loss > 1000 (possible divergence)
- KL divergence > 0.05 (policy changing too fast)
- Clip fraction > 0.3 (clipping too aggressively)
- Entropy collapsing to near-zero
- Reward decreasing over last 20% of training

For each completed run, report:
- Final mean reward (last 10% of iterations)
- Peak eval reward and which iteration
- Whether training appeared stable or showed issues

### Phase 4: Compare Runs

After all runs complete, produce a comparison summary:

**Comparison table:**
```markdown
| Run | Final Reward | Peak Eval Reward | Peak Iter | Stable? | Key Hyperparam Diffs |
|-----|-------------|-----------------|-----------|---------|---------------------|
```

**Analysis:**
- Which run performed best and why
- Which hyperparameter changes had the most impact
- Any runs that diverged or showed instability
- Recommendations for further experiments

## Tips

- For short experiments (testing quickly), suggest `--n-itr 100-500` with `--eval-freq 50`
- For cartpole, always include `--no-mirror`
- Keep `--num-procs` consistent across runs in the same experiment for fair FPS comparison
- Use descriptive run names that encode the key variable (e.g., `gamma095`, `lr1e3`)
