"""Headless cartpole evaluation script.

Measures how long the pole stays upright during a 10s episode.
No GUI required.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from envs.cartpole import CartpoleEnv


@torch.no_grad()
def evaluate_policy(actor_path, n_episodes=10, ep_len_sec=10.0, seed=42):
    """Evaluate a saved policy checkpoint.

    Args:
        actor_path: Path to actor .pt file
        n_episodes: Number of episodes to evaluate
        ep_len_sec: Episode length in seconds
        seed: Random seed for reproducibility

    Returns:
        dict with evaluation metrics
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    policy = torch.load(actor_path, weights_only=False, map_location="cpu")
    policy.eval()

    env = CartpoleEnv()
    control_dt = env.model.opt.timestep * env.frame_skip  # 0.005 * 4 = 0.02s
    max_steps = int(ep_len_sec / control_dt)  # 10 / 0.02 = 500 steps

    # Detect policy's expected observation dimension
    policy_obs_dim = policy.obs_mean.shape[0] if torch.is_tensor(policy.obs_mean) else 5
    env_obs_dim = env.observation_space.shape[0]
    if policy_obs_dim != env_obs_dim:
        print(f"  Note: policy was trained with {policy_obs_dim}D obs, env returns {env_obs_dim}D obs.")
        print("  Will use compatibility mode for old 4D policy.")
        use_legacy_obs = policy_obs_dim == 4
    else:
        use_legacy_obs = False

    # upright threshold: cos(angle) > 0.95 ≈ within ~18° of vertical
    upright_threshold = 0.95

    results = []
    for ep in range(n_episodes):
        obs = env.reset()
        upright_steps = 0
        total_steps = 0
        ep_reward = 0.0

        for _step in range(max_steps):
            if use_legacy_obs:
                # Old 4D obs: [cart_pos, angle%(2pi), cart_vel, pole_vel]
                raw_angle = env.data.qpos[env._hinge_qpos_idx]
                legacy_obs = np.array(
                    [
                        obs[0],
                        raw_angle % (2 * np.pi),
                        obs[3],  # cart_vel is at index 3 in new 5D obs
                        obs[4],  # pole_vel is at index 4 in new 5D obs
                    ]
                )
                obs_tensor = torch.tensor(legacy_obs, dtype=torch.float32)
            else:
                obs_tensor = torch.tensor(obs, dtype=torch.float32)
            action = policy.forward(obs_tensor, deterministic=True).numpy()
            obs, reward, done, info = env.step(action.copy())

            total_steps += 1
            ep_reward += reward

            # Check if pole is upright: obs[1] = cos(angle) in new 5D representation
            cos_angle = obs[1]
            if cos_angle > upright_threshold:
                upright_steps += 1

            if done:
                break

        upright_fraction = upright_steps / total_steps
        results.append(
            {
                "ep": ep,
                "total_steps": total_steps,
                "upright_steps": upright_steps,
                "upright_fraction": upright_fraction,
                "ep_reward": ep_reward,
            }
        )
        print(
            f"  Ep {ep}: {upright_steps}/{total_steps} steps upright "
            f"({100 * upright_fraction:.1f}%), reward={ep_reward:.1f}"
        )

    mean_upright = np.mean([r["upright_fraction"] for r in results])
    mean_reward = np.mean([r["ep_reward"] for r in results])
    mean_ep_len = np.mean([r["total_steps"] for r in results])

    print(f"\nSummary over {n_episodes} episodes:")
    print(f"  Mean upright fraction: {100 * mean_upright:.1f}%  (target: >=90%)")
    print(f"  Mean episode length: {mean_ep_len:.1f}/{max_steps} steps")
    print(f"  Mean episode reward: {mean_reward:.1f}")
    print(f"  SUCCESS: {'YES' if mean_upright >= 0.90 else 'NO'}")

    return {
        "mean_upright_fraction": mean_upright,
        "mean_reward": mean_reward,
        "mean_ep_len": mean_ep_len,
        "success": mean_upright >= 0.90,
        "episodes": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True, help="Path to actor .pt file or run directory")
    parser.add_argument("--n-episodes", type=int, default=10, help="Number of eval episodes")
    parser.add_argument("--ep-len", type=float, default=10.0, help="Episode length in seconds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.95, help="cos(angle) threshold for 'upright'")
    args = parser.parse_args()

    actor_path = args.path
    if actor_path.is_dir():
        # Find best checkpoint
        checkpoints = list(actor_path.glob("actor_*.pt"))
        if checkpoints:
            import re

            actor_path = max(checkpoints, key=lambda f: int(re.search(r"actor_(\d+)\.pt", f.name).group(1)))
        else:
            actor_path = actor_path / "actor.pt"

    print(f"Loading: {actor_path}")
    evaluate_policy(actor_path, args.n_episodes, args.ep_len, args.seed)
