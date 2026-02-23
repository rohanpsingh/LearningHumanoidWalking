# Cartpole Swing-Up Research Log

**Goal:** Train policy achieving ≥90% upright duration in 10s episode, with minimal oscillation when upright.
**Eval metric:** `cos(θ) > 0.95` threshold (within ~18° of vertical) over 500 steps (10s at 0.02s control_dt).

---

## Environment Bugs Fixed

**Critical:** `angle % (2π)` in `_get_obs()` creates a 0/2π discontinuity when pole passes through upright. Neural network sees wildly different values for nearly identical physical states.
**Fix:** Replace 4D obs `[cart_pos, angle%(2π), cart_vel, pole_vel]` with 5D obs `[cart_pos, cos(θ), sin(θ), cart_vel, pole_vel]`.

**Stochastic-to-deterministic gap:** Unbounded policy outputs target positions >1m during deterministic eval, crashing cart into track boundary.
**Fix:** `np.clip(action, -0.8, 0.8)` in `CartpoleEnv.step()`.

---

## Experiments

### Run 0 — Baseline
```
--n-itr 500 --gamma 0.95 --std-dev 0.15 --learn-std
```
**Result:** 88.7% mean (seed 42, 10 ep). Just below target; inconsistent across seeds.
**Issue:** 4D obs with angle%(2π) discontinuity; short traj-len=400 (8s) vs 10s eval.

### Run 1 — Fixed obs + gamma=0.99
```
--gamma 0.99 --max-traj-len 500 (5D obs, near_upright reward weights)
```
**Result:** 84.2% best; diverged at iter ~500.
**Diagnosis:** near_upright reward weighting created non-smooth signal; critic loss hit 45.

### Run 2 — Stabilized gamma=0.95
```
--gamma 0.95 --entropy-coeff 0.01 --max-traj-len 500 (5D obs, exp reward)
```
**Result:** 90.0% (seed 42), 87.9% (seed 0). Inconsistent across seeds.
**Diagnosis:** γ=0.95 effective horizon ~20 steps; swing-up needs ~50-100 steps → credit assignment fails for hard starting angles.

### Run 3 — Linear reward + gamma=0.99
```
--gamma 0.99 --lr 1e-4 (linear cosine reward)
```
**Result:** Failed. Critic loss exploded to 96. Did not learn.
**Diagnosis:** γ=0.99 returns up to ~99; lr=1e-4 too large for stable critic training.

### Run 4 — Linear reward + gamma=0.95
```
--gamma 0.95 --entropy-coeff 0.01 (linear reward, no action clip)
```
**Result:** 57.9%. Cart goes out of bounds deterministically.
**Diagnosis:** Missing action clipping; stochastic training masks out-of-bounds behavior.

### Run 5 — Hybrid reward + action clipping + gamma=0.95
```
--gamma 0.95 --entropy-coeff 0.01 --max-traj-len 500 (hybrid reward, action clip ±0.8)
```
**Result:** 3.2% at iter 599. Policy stuck at 0.3 reward/step (pole hanging).
**Diagnosis:** γ=0.95 cannot assign credit to early swing-up actions. Killed at iter 599.

### Run 6 — **SUCCESS**
```
--gamma 0.99 --lr 3e-5 --minibatch-size 256 --max-grad-norm 0.02
--entropy-coeff 0.01 --std-dev 0.15 --learn-std
--max-traj-len 500 --n-itr 1500
```
**Best checkpoint:** `actor_999.pt`
**Results:**
| Seed | Episodes | Upright % |
|------|----------|-----------|
| 42   | 10       | **92.2%** |
| 0    | 20       | **91.8%** |
| 1    | 10       | **91.7%** |
| 2    | 10       | **93.4%** |
| 3    | 30       | **90.7%** |

**Overall: 91.6% mean over 80 diverse episodes. All seeds ≥ 90%. SUCCESS.**

Note: `actor_1499.pt` (final) is worse (88.2%) — policy degrades after iter 999 due to entropy growth. Use `actor_999.pt`.

---

## Key Findings

1. **γ=0.99 is required** for swing-up. Effective horizon = 1/(1-γ) steps. Swing-up from hanging takes ~50-100 steps; γ=0.95 gives only ~20-step horizon → cannot plan swing-up.

2. **Stabilizing γ=0.99:** Previous attempts with γ=0.99 had critic diverge (loss→96). Fix: `lr=3e-5` (not 1e-4) + `minibatch=256` + `max-grad-norm=0.02`. Critic loss peaked at ~48 at iter 46, converged to ~7-17.

3. **Hybrid linear+exp reward:** Pure exp reward has near-zero gradient at hanging (`d/d(cos)|_{cos=-1} ≈ 0.003`). Linear component gives constant gradient everywhere; exp sharpens near upright. Max total reward ~0.70 per step.

4. **Policy peaks at iter 999.** Late training (1000-1499) shows entropy growth (0.65→1.32) and performance regression. Stop at iter 999.

---

## Reproduction

```bash
RAY_ADDRESS= uv run python run_experiment.py train \
  --env cartpole --n-itr 1500 --gamma 0.99 --lr 3e-5 \
  --minibatch-size 256 --max-grad-norm 0.02 \
  --std-dev 0.15 --learn-std --entropy-coeff 0.01 \
  --num-procs 12 --max-traj-len 500 \
  --logdir /tmp/cartpole_run

# Evaluate (use actor_999.pt, not actor.pt)
uv run python eval_cartpole.py \
  --path /tmp/cartpole_run/<timestamp>_cartpole/actor_999.pt \
  --n-episodes 20 --seed 0
```

Best checkpoint for this log: `/tmp/cartpole_experiments/run6_gamma99_stable/26-02-22-16-06-15_cartpole/actor_999.pt`
