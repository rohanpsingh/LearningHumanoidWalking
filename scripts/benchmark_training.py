#!/usr/bin/env python3
"""Benchmark script for comparing training speed across branches.

This script runs training for a small number of iterations and captures
timing and reward metrics for comparison.

Usage:
    python scripts/benchmark_training.py --env jvrc_walk --n-itr 10
    python scripts/benchmark_training.py --env h1 --n-itr 10 --num-procs 4
"""

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def run_training(env: str, n_itr: int, num_procs: int, logdir: Path, device: str = "auto") -> dict:
    """Run training and capture metrics.

    Args:
        env: Environment name (e.g., 'jvrc_walk', 'h1')
        n_itr: Number of training iterations
        num_procs: Number of parallel processes
        logdir: Directory for logs
        device: Device for training ('auto', 'cpu', 'cuda')

    Returns:
        Dictionary containing benchmark results
    """
    cmd = [
        sys.executable,
        "run_experiment.py",
        "train",
        "--env",
        env,
        "--n-itr",
        str(n_itr),
        "--num-procs",
        str(num_procs),
        "--logdir",
        str(logdir),
        "--no-mirror",  # Disable mirror for consistency
        "--eval-freq",
        str(n_itr + 1),  # Disable eval during benchmark
        "--device",
        device,
    ]

    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)

    start_time = time.time()

    # Run training and capture output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines = []
    iteration_times = []
    rewards = []
    episode_lengths = []
    fps_values = []

    # Patterns for parsing output
    iter_pattern = re.compile(r"\*+ Iteration (\d+) \*+")
    reward_pattern = re.compile(r"\|\s+Mean Eprew\s+\|\s+([\d.e+-]+)\s+\|")
    eplen_pattern = re.compile(r"\|\s+Mean Eplen\s+\|\s+([\d.e+-]+)\s+\|")
    time_pattern = re.compile(r"Total time elapsed: ([\d.]+)s.*fps=([\d.]+)")
    sample_time_pattern = re.compile(r"Sampling took ([\d.]+)s for (\d+) steps")

    current_iter_start = None
    sample_times = []
    optimizer_times = []

    for line in process.stdout:
        print(line, end="")
        output_lines.append(line)

        # Parse iteration number
        iter_match = iter_pattern.search(line)
        if iter_match:
            if current_iter_start is not None:
                iteration_times.append(time.time() - current_iter_start)
            current_iter_start = time.time()

        # Parse reward
        reward_match = reward_pattern.search(line)
        if reward_match:
            rewards.append(float(reward_match.group(1)))

        # Parse episode length
        eplen_match = eplen_pattern.search(line)
        if eplen_match:
            episode_lengths.append(float(eplen_match.group(1)))

        # Parse FPS
        time_match = time_pattern.search(line)
        if time_match:
            fps_str = time_match.group(2).rstrip(".")
            fps_values.append(float(fps_str))

        # Parse sample time
        sample_match = sample_time_pattern.search(line)
        if sample_match:
            sample_times.append(float(sample_match.group(1)))

        # Parse optimizer time
        if "Optimizer took:" in line:
            opt_time = float(line.split("Optimizer took:")[1].split("s")[0].strip())
            optimizer_times.append(opt_time)

    # Capture last iteration time
    if current_iter_start is not None:
        iteration_times.append(time.time() - current_iter_start)

    process.wait()
    total_time = time.time() - start_time

    # Compute statistics
    results = {
        "env": env,
        "n_itr": n_itr,
        "num_procs": num_procs,
        "device": device,
        "total_time_seconds": round(total_time, 2),
        "avg_iteration_time": round(sum(iteration_times) / len(iteration_times), 3) if iteration_times else 0,
        "avg_sample_time": round(sum(sample_times) / len(sample_times), 3) if sample_times else 0,
        "avg_optimizer_time": round(sum(optimizer_times) / len(optimizer_times), 3) if optimizer_times else 0,
        "avg_fps": round(sum(fps_values) / len(fps_values), 1) if fps_values else 0,
        "final_fps": fps_values[-1] if fps_values else 0,
        "rewards": {
            "first": rewards[0] if rewards else None,
            "last": rewards[-1] if rewards else None,
            "mean": round(sum(rewards) / len(rewards), 2) if rewards else None,
            "all": rewards,
        },
        "episode_lengths": {
            "first": episode_lengths[0] if episode_lengths else None,
            "last": episode_lengths[-1] if episode_lengths else None,
            "mean": round(sum(episode_lengths) / len(episode_lengths), 2) if episode_lengths else None,
        },
        "return_code": process.returncode,
        "timestamp": datetime.now().isoformat(),
    }

    return results


def print_summary(results: dict):
    """Print a formatted summary of benchmark results."""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 60)
    print(f"Environment:        {results['env']}")
    print(f"Iterations:         {results['n_itr']}")
    print(f"Num processes:      {results['num_procs']}")
    print("-" * 60)
    print(f"Total time:         {results['total_time_seconds']:.2f}s")
    print(f"Avg iteration time: {results['avg_iteration_time']:.3f}s")
    print(f"Avg sample time:    {results['avg_sample_time']:.3f}s")
    print(f"Avg optimizer time: {results['avg_optimizer_time']:.3f}s")
    print(f"Average FPS:        {results['avg_fps']:.1f}")
    print(f"Final FPS:          {results['final_fps']:.1f}")
    print("-" * 60)
    print(f"First reward:       {results['rewards']['first']}")
    print(f"Last reward:        {results['rewards']['last']}")
    print(f"Mean reward:        {results['rewards']['mean']}")
    print(f"First ep length:    {results['episode_lengths']['first']}")
    print(f"Last ep length:     {results['episode_lengths']['last']}")
    print("=" * 60)


def compare_results(results1: dict, results2: dict, label1: str, label2: str):
    """Compare two benchmark results and print a comparison table."""
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<25} {label1:>20} {label2:>20}")
    print("-" * 70)

    metrics = [
        ("Total time (s)", "total_time_seconds"),
        ("Avg iteration time (s)", "avg_iteration_time"),
        ("Avg sample time (s)", "avg_sample_time"),
        ("Avg optimizer time (s)", "avg_optimizer_time"),
        ("Average FPS", "avg_fps"),
        ("Final FPS", "final_fps"),
    ]

    for label, key in metrics:
        v1 = results1.get(key, 0)
        v2 = results2.get(key, 0)
        diff = ((v2 - v1) / v1 * 100) if v1 != 0 else 0
        diff_str = f"({diff:+.1f}%)" if diff != 0 else ""
        print(f"{label:<25} {v1:>20.2f} {v2:>17.2f} {diff_str}")

    print("-" * 70)
    print(f"{'First reward':<25} {results1['rewards']['first']:>20.2f} {results2['rewards']['first']:>20.2f}")
    print(f"{'Last reward':<25} {results1['rewards']['last']:>20.2f} {results2['rewards']['last']:>20.2f}")
    print(f"{'Mean reward':<25} {results1['rewards']['mean']:>20.2f} {results2['rewards']['mean']:>20.2f}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Benchmark training speed")
    parser.add_argument("--env", type=str, default="jvrc_walk", help="Environment name (default: jvrc_walk)")
    parser.add_argument("--n-itr", type=int, default=10, help="Number of training iterations (default: 10)")
    parser.add_argument("--num-procs", type=int, default=12, help="Number of parallel processes (default: 12)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file for results")
    parser.add_argument("--compare", type=str, default=None, help="JSON file with previous results to compare against")
    parser.add_argument("--label", type=str, default=None, help="Label for this benchmark run")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for training: 'auto', 'cpu', or 'cuda' (default: auto)",
    )
    args = parser.parse_args()

    # Create unique log directory
    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
    logdir = Path(f"/tmp/benchmark-{timestamp}")

    print("=" * 60)
    print("TRAINING SPEED BENCHMARK")
    print("=" * 60)
    print(f"Environment: {args.env}")
    print(f"Iterations:  {args.n_itr}")
    print(f"Processes:   {args.num_procs}")
    print(f"Device:      {args.device}")
    print(f"Log dir:     {logdir}")
    print("=" * 60 + "\n")

    # Run benchmark
    results = run_training(args.env, args.n_itr, args.num_procs, logdir, args.device)

    if args.label:
        results["label"] = args.label

    # Print summary
    print_summary(results)

    # Save results if output specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    # Compare with previous results if specified
    if args.compare:
        compare_path = Path(args.compare)
        if compare_path.exists():
            with open(compare_path) as f:
                prev_results = json.load(f)
            label1 = prev_results.get("label", "Previous")
            label2 = args.label or "Current"
            compare_results(prev_results, results, label1, label2)
        else:
            print(f"Warning: Comparison file not found: {compare_path}")

    return results


if __name__ == "__main__":
    main()
