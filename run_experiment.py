import argparse
import os
import pickle
import platform
import shutil
import sys
from datetime import datetime
from functools import partial
from pathlib import Path

import mujoco
import ray
import torch

from rl.algos.ppo import PPO
from rl.envs.wrappers import SymmetricEnv
from rl.utils.eval import EvaluateEnv
from rl.utils.seeding import set_global_seeds


def print_system_info(args, training=True):
    """Print system and training configuration info."""
    print("=" * 60)
    print("System Information")
    print("=" * 60)
    print(f"MuJoCo version: {mujoco.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Ray version: {ray.__version__}")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"CPU: {platform.processor() or 'N/A'}")
    print(f"CPU count: {os.cpu_count()}")
    if training:
        print("-" * 60)
        print("Training Configuration")
        print("-" * 60)
        print(f"Environment: {args.env}")
        print(f"Log directory: {args.logdir}")
        print(f"Num processes: {args.num_procs}")
        print(f"Learning rate: {args.lr}")
        print(f"Max trajectory length: {args.max_traj_len}")
        print(f"Iterations: {args.n_itr}")
        if hasattr(args, "seed") and args.seed is not None:
            print(f"Seed: {args.seed} (deterministic)")
        else:
            print("Seed: None (non-deterministic)")
    print("=" * 60)


def get_latest_run_dir(logdir):
    """Find the most recent run subdirectory under logdir."""
    logdir = Path(logdir)
    if not logdir.exists():
        return None

    subdirs = [d for d in logdir.iterdir() if d.is_dir()]
    if not subdirs:
        return None

    # Sort by modification time, most recent first
    subdirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return subdirs[0]


def import_env(env_name_str):
    if env_name_str == "jvrc_walk":
        from envs.jvrc import JvrcWalkEnv as Env
    elif env_name_str == "jvrc_step":
        from envs.jvrc import JvrcStepEnv as Env
    elif env_name_str == "h1":
        from envs.h1 import H1Env as Env
    elif env_name_str == "cartpole":
        from envs.cartpole import CartpoleEnv as Env
    else:
        raise Exception("Check env name!")
    return Env


def run_experiment(args):
    # Create timestamped subdirectory: yy-mm-dd-hh-mm-ss_env_name
    timestamp = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    run_name = f"{timestamp}_{args.env}"
    args.logdir = Path(args.logdir) / run_name

    # Print system info before training
    print_system_info(args)

    # import the correct environment
    Env = import_env(args.env)

    # wrapper function for creating parallelized envs
    env_fn = partial(Env, path_to_yaml=args.yaml)
    _env = env_fn()
    if not args.no_mirror:
        try:
            print("Wrapping in SymmetricEnv.")
            env_fn = partial(
                SymmetricEnv,
                env_fn,
                mirrored_obs=_env.robot.mirrored_obs,
                mirrored_act=_env.robot.mirrored_acts,
                clock_inds=_env.robot.clock_inds,
            )
        except AttributeError as e:
            print("Warning! Cannot use SymmetricEnv.", e)

    # Set up Parallelism
    # os.environ['OMP_NUM_THREADS'] = '1'  # [TODO: Is this needed?]
    if not ray.is_initialized():
        ray.init(num_cpus=args.num_procs)

    # dump hyperparameters
    Path.mkdir(args.logdir, parents=True, exist_ok=True)
    pkl_path = Path(args.logdir, "experiment.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(args, f)

    # copy config file
    if args.yaml:
        config_out_path = Path(args.logdir, "config.yaml")
        shutil.copyfile(args.yaml, config_out_path)

    algo = PPO(env_fn, args, seed=getattr(args, "seed", None))
    algo.train(env_fn, args.n_itr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    if sys.argv[1] == "train":
        sys.argv.remove(sys.argv[1])

        parser.add_argument("--env", required=True, type=str)
        parser.add_argument("--logdir", default=Path("/tmp/logs"), type=Path, help="Path to save weights and logs")
        parser.add_argument("--input-norm-steps", type=int, default=100000)
        parser.add_argument("--n-itr", type=int, default=20000, help="Number of iterations of the learning algorithm")
        parser.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate")  # Xie
        parser.add_argument("--eps", type=float, default=1e-5, help="Adam epsilon (for numerical stability)")
        parser.add_argument("--gamma", type=float, default=0.99, help="MDP discount")
        parser.add_argument("--std-dev", type=float, default=0.223, help="Action noise for exploration")
        parser.add_argument("--learn-std", action="store_true", help="Exploration noise will be learned")
        parser.add_argument("--entropy-coeff", type=float, default=0.0, help="Coefficient for entropy regularization")
        parser.add_argument("--clip", type=float, default=0.2, help="Clipping parameter for PPO surrogate loss")
        parser.add_argument("--minibatch-size", type=int, default=64, help="Batch size for PPO updates")
        parser.add_argument("--epochs", type=int, default=3, help="Number of optimization epochs per PPO update")  # Xie
        parser.add_argument("--num-procs", type=int, default=12, help="Number of threads to train on")
        parser.add_argument("--max-grad-norm", type=float, default=0.05, help="Value to clip gradients at")
        parser.add_argument("--max-traj-len", type=int, default=400, help="Max episode horizon")
        parser.add_argument("--no-mirror", required=False, action="store_true", help="to use SymmetricEnv")
        parser.add_argument("--mirror-coeff", required=False, default=0.4, type=float, help="weight for mirror loss")
        parser.add_argument(
            "--eval-freq", required=False, default=100, type=int, help="Frequency of performing evaluation"
        )
        parser.add_argument("--continued", required=False, type=Path, help="path to pretrained weights")
        parser.add_argument("--recurrent", required=False, action="store_true", help="use LSTM instead of FF")
        parser.add_argument("--imitate", required=False, type=str, default=None, help="Policy to imitate")
        parser.add_argument(
            "--imitate-coeff", required=False, type=float, default=0.3, help="Coefficient for imitation loss"
        )
        parser.add_argument(
            "--yaml", required=False, type=str, default=None, help="Path to config file passed to Env class"
        )
        parser.add_argument(
            "--device",
            required=False,
            type=str,
            default="auto",
            choices=["auto", "cpu", "cuda"],
            help="Device for training: 'auto' (use GPU if available), 'cpu', or 'cuda'",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help="Random seed for reproducibility. If not set, training is non-deterministic.",
        )
        args = parser.parse_args()

        # Apply global seeding before any randomness
        if args.seed is not None:
            set_global_seeds(args.seed, cuda_deterministic=True)
            print(f"Deterministic mode enabled with seed: {args.seed}")

        run_experiment(args)

    elif sys.argv[1] == "eval":
        sys.argv.remove(sys.argv[1])

        parser.add_argument(
            "--path",
            required=False,
            type=Path,
            default=None,
            help="Path to trained model (actor.pt file or directory containing it)",
        )
        parser.add_argument(
            "--logdir",
            required=False,
            type=Path,
            default=None,
            help="Path to log directory; will use actor.pt from most recent run",
        )
        parser.add_argument(
            "--out-dir", required=False, type=Path, default=None, help="Path to directory to save videos"
        )
        parser.add_argument(
            "--ep-len", required=False, type=int, default=10, help="Episode length to play (in seconds)"
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help="Random seed for reproducible evaluation",
        )
        args = parser.parse_args()

        # Apply global seeding before any randomness
        if args.seed is not None:
            set_global_seeds(args.seed, cuda_deterministic=True)

        # Determine path to actor
        path_to_actor = None
        if args.path is not None:
            # Use --path if provided
            if args.path.is_file() and args.path.suffix == ".pt":
                path_to_actor = args.path
            elif args.path.is_dir():
                path_to_actor = Path(args.path, "actor.pt")
            else:
                raise Exception("Invalid path to actor module: ", args.path)
        elif args.logdir is not None:
            # Use --logdir: find most recent run subdirectory
            latest_run = get_latest_run_dir(args.logdir)
            if latest_run is None:
                raise Exception(f"No run directories found under: {args.logdir}")
            path_to_actor = Path(latest_run, "actor.pt")
            print(f"Using most recent run: {latest_run}")
        else:
            raise Exception("Must provide either --path or --logdir")

        # Set args.path for use in EvaluateEnv
        args.path = path_to_actor

        path_to_critic = Path(path_to_actor.parent, "critic" + str(path_to_actor).split("actor")[1])
        path_to_pkl = Path(path_to_actor.parent, "experiment.pkl")

        # load experiment args
        with open(path_to_pkl, "rb") as f:
            run_args = pickle.load(f)
        # load trained policy
        policy = torch.load(path_to_actor, weights_only=False)
        critic = torch.load(path_to_critic, weights_only=False)
        policy.eval()
        critic.eval()

        # Print system info
        print_system_info(args, training=False)

        # import the correct environment
        Env = import_env(run_args.env)
        if "yaml" in run_args and run_args.yaml is not None:
            yaml_path = Path(run_args.yaml)
        else:
            yaml_path = None
        env = partial(Env, yaml_path)()

        # run
        e = EvaluateEnv(env, policy, args)
        e.run()
