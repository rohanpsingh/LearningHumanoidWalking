#!/usr/bin/env python
"""Test script to compare contact behavior between MuJoCo versions.

Loads each environment, simulates for 60 seconds to reach static state,
then reports contact information for comparison.

Usage:
    python scripts/test_contact_behavior.py                    # Run all envs, no viewer
    python scripts/test_contact_behavior.py --viewer           # Run all envs with viewer
    python scripts/test_contact_behavior.py --env jvrc_walk    # Run specific env
    python scripts/test_contact_behavior.py --env h1 --viewer  # Run specific env with viewer
"""

import argparse
import time

import mujoco
import numpy as np


def get_contact_info(env):
    """Extract contact info using env's built-in diagnostics plus foot-specific data."""
    # Use the environment's built-in contact summary
    info = env.get_contact_summary()

    # Add foot-specific info from the interface
    rfoot_contacts = env.interface.get_rfoot_floor_contacts()
    lfoot_contacts = env.interface.get_lfoot_floor_contacts()
    info["rfoot_contact_count"] = len(rfoot_contacts)
    info["lfoot_contact_count"] = len(lfoot_contacts)
    info["rfoot_grf"] = env.interface.get_rfoot_grf()
    info["lfoot_grf"] = env.interface.get_lfoot_grf()
    info["total_grf"] = info["rfoot_grf"] + info["lfoot_grf"]

    return info


def simulate_to_static(env, sim_seconds=60.0, use_viewer=False):
    """Simulate environment for given seconds with zero action."""
    env.reset()
    action = np.zeros(env.action_space.shape[0])

    # Calculate number of steps needed
    dt = env.dt
    n_steps = int(sim_seconds / dt)

    if use_viewer:
        env.render()  # Initialize viewer

    for _step in range(n_steps):
        obs, reward, done, info = env.step(action)
        if done:
            env.reset()

        if use_viewer:
            env.render()
            # Sync with real time
            time.sleep(dt)

    return env


def print_contact_info(info):
    """Print contact information."""
    print("\n--- Contact Summary ---")
    print(f"Total contacts: {info['ncon']} (max: {info['nconmax']})")
    print(f"Constraints: {info['nefc']} (max: {info['njmax']})")
    print(f"Right foot contacts: {info['rfoot_contact_count']}")
    print(f"Left foot contacts: {info['lfoot_contact_count']}")
    print(f"Right foot GRF: {info['rfoot_grf']:.2f} N")
    print(f"Left foot GRF: {info['lfoot_grf']:.2f} N")
    print(f"Total GRF: {info['total_grf']:.2f} N")

    print("\n--- Contact Pairs ---")
    for i, con in enumerate(info["contacts"]):
        print(f"  [{i}] {con['geom1']} <-> {con['geom2']}: dist={con['dist']:.6f}, force={con['force']:.2f} N")


def test_environment(env_name, env_class, use_viewer=False):
    """Test a single environment and return contact info."""
    print(f"\n{'=' * 60}")
    print(f"Testing: {env_name}")
    print(f"{'=' * 60}")

    env = env_class()

    if use_viewer:
        print("Simulating with viewer... (close window or Ctrl+C to continue)")
        env.reset()
        env.render()

        # Enable contact point and force visualization
        with env.viewer.lock():
            env.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            env.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

        action = np.zeros(env.action_space.shape[0])
        dt = env.dt

        try:
            while env.viewer.is_running():
                start_time = time.time()

                obs, reward, done, info = env.step(action)
                if done:
                    env.reset()

                env.render()

                # Print contact info periodically
                if int(time.time()) % 5 == 0:
                    contact_info = get_contact_info(env)
                    print(
                        f"\rContacts: {contact_info['ncon']}, "
                        f"R-GRF: {contact_info['rfoot_grf']:.1f}N, "
                        f"L-GRF: {contact_info['lfoot_grf']:.1f}N",
                        end="",
                        flush=True,
                    )

                # Sync with real time
                elapsed = time.time() - start_time
                sleep_time = max(0, dt - elapsed)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nStopped by user")

        # Get final contact info
        contact_info = get_contact_info(env)
        print_contact_info(contact_info)
        env.close()
        return contact_info

    else:
        print("Simulating for 60 seconds (no viewer)...")
        env = simulate_to_static(env, sim_seconds=60.0, use_viewer=False)

        # Get contact info
        contact_info = get_contact_info(env)
        print_contact_info(contact_info)
        env.close()
        return contact_info


def main():
    parser = argparse.ArgumentParser(description="Test contact behavior across MuJoCo versions")
    parser.add_argument("--viewer", action="store_true", help="Show viewer for visual inspection")
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        choices=["jvrc_walk", "jvrc_step", "h1"],
        help="Test specific environment (default: all)",
    )
    args = parser.parse_args()

    print(f"MuJoCo version: {mujoco.__version__}")
    print("=" * 60)

    # Import environments
    from envs.h1.h1_env import H1Env
    from envs.jvrc.jvrc_step import JvrcStepEnv
    from envs.jvrc.jvrc_walk import JvrcWalkEnv

    all_environments = {
        "jvrc_walk": JvrcWalkEnv,
        "jvrc_step": JvrcStepEnv,
        "h1": H1Env,
    }

    # Select environments to test
    if args.env:
        environments = [(args.env, all_environments[args.env])]
    else:
        environments = list(all_environments.items())

    results = {}
    for env_name, env_class in environments:
        try:
            results[env_name] = test_environment(env_name, env_class, use_viewer=args.viewer)
        except Exception as e:
            print(f"ERROR testing {env_name}: {e}")
            import traceback

            traceback.print_exc()
            results[env_name] = {"error": str(e)}

    # Print summary comparison table
    print(f"\n\n{'=' * 60}")
    print("SUMMARY TABLE")
    print(f"{'=' * 60}")
    print(f"{'Environment':<15} {'Contacts':<10} {'R-Foot':<10} {'L-Foot':<10} {'Total GRF':<12}")
    print("-" * 60)
    for env_name, info in results.items():
        if "error" in info:
            print(f"{env_name:<15} ERROR: {info['error']}")
        else:
            print(
                f"{env_name:<15} {info['ncon']:<10} "
                f"{info['rfoot_contact_count']:<10} {info['lfoot_contact_count']:<10} "
                f"{info['total_grf']:<12.2f}"
            )

    return results


if __name__ == "__main__":
    main()
