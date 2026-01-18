"""Proximal Policy Optimization (clip objective)."""

import datetime
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import ray
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from rl.envs.normalize import RunningMeanStd
from rl.policies.actor import Gaussian_FF_Actor, Gaussian_LSTM_Actor
from rl.policies.critic import FF_V, LSTM_V
from rl.utils import ModelCheckpointer, TrainingLogger
from rl.utils.seeding import get_worker_seed
from rl.workers import RolloutWorker


class PPO:
    def __init__(self, env_fn, args, seed=None):
        self.seed = seed
        self.gamma = args.gamma
        self.lr = args.lr
        self.eps = args.eps
        self.ent_coeff = args.entropy_coeff
        self.clip = args.clip
        self.minibatch_size = args.minibatch_size
        self.epochs = args.epochs
        self.max_traj_len = args.max_traj_len
        self.n_proc = args.num_procs
        self.grad_clip = args.max_grad_norm
        self.mirror_coeff = args.mirror_coeff
        self.eval_freq = args.eval_freq
        self.recurrent = args.recurrent
        self.imitate_coeff = args.imitate_coeff

        # batch_size depends on number of parallel envs
        self.batch_size = self.n_proc * self.max_traj_len

        self.total_steps = 0

        # counter for training iterations
        self.iteration_count = 0

        # directory for saving weights
        self.save_path = Path(args.logdir)

        # create logger and checkpointer
        self.logger = TrainingLogger(self.save_path, flush_secs=10)
        self.checkpointer = ModelCheckpointer(self.save_path)

        # create networks or load up pretrained
        obs_dim = env_fn().observation_space.shape[0]
        action_dim = env_fn().action_space.shape[0]
        if args.continued:
            path_to_actor = args.continued
            path_to_critic = Path(args.continued.parent, "critic" + str(args.continued).split("actor")[1])
            policy = torch.load(path_to_actor, weights_only=False)
            critic = torch.load(path_to_critic, weights_only=False)
            # policy action noise parameters are initialized from scratch and not loaded
            if args.learn_std:
                policy.stds = torch.nn.Parameter(args.std_dev * torch.ones(action_dim))
            else:
                policy.stds = args.std_dev * torch.ones(action_dim)
            print("Loaded (pre-trained) actor from: ", path_to_actor)
            print("Loaded (pre-trained) critic from: ", path_to_critic)
            # Pretrained models already have obs normalization embedded
            self.obs_rms = None
        else:
            if args.recurrent:
                policy = Gaussian_LSTM_Actor(obs_dim, action_dim, init_std=args.std_dev, learn_std=args.learn_std)
                critic = LSTM_V(obs_dim)
            else:
                policy = Gaussian_FF_Actor(
                    obs_dim, action_dim, init_std=args.std_dev, learn_std=args.learn_std, bounded=False
                )
                critic = FF_V(obs_dim)

            # Setup observation normalization
            env_instance = env_fn()
            if hasattr(env_instance, "obs_mean") and hasattr(env_instance, "obs_std"):
                # Use fixed normalization params from environment
                obs_mean, obs_std = env_instance.obs_mean, env_instance.obs_std
                self.obs_rms = None  # No running stats needed
                print("Using fixed observation normalization from environment.")
            else:
                # Use running mean/std that will be updated during training
                self.obs_rms = RunningMeanStd(shape=(obs_dim,))
                obs_mean, obs_std = self.obs_rms.mean, self.obs_rms.std
                print("Using running observation normalization (will update during training).")

            with torch.no_grad():
                policy.obs_mean, policy.obs_std = map(torch.Tensor, (obs_mean, obs_std))
                critic.obs_mean = policy.obs_mean
                critic.obs_std = policy.obs_std

        base_policy = None
        if args.imitate:
            base_policy = torch.load(args.imitate, weights_only=False)

        # Device setup (from args or auto-detect)
        device_arg = getattr(args, "device", "auto")
        if device_arg == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_arg)

        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                print("Warning: CUDA requested but not available, falling back to CPU")
                self.device = torch.device("cpu")
            else:
                print(f"GPU detected: {torch.cuda.get_device_name(0)}")
                print("Moving policy and critic to GPU...")
                policy = policy.to(self.device)
                critic = critic.to(self.device)
                # Also move non-parameter tensors to GPU
                policy.obs_mean = policy.obs_mean.to(self.device)
                policy.obs_std = policy.obs_std.to(self.device)
                critic.obs_mean = critic.obs_mean.to(self.device)
                critic.obs_std = critic.obs_std.to(self.device)
                # Move stds if it's a plain tensor (not nn.Parameter)
                if not isinstance(policy.stds, torch.nn.Parameter):
                    policy.stds = policy.stds.to(self.device)

        if self.device.type == "cpu":
            print("Using CPU for training")

        self.old_policy = deepcopy(policy)
        self.policy = policy
        self.critic = critic
        self.base_policy = base_policy

        # Store env_fn for later use
        self.env_fn = env_fn

        # Create persistent worker actors - this is the key optimization.
        # Each worker creates its environment ONCE and reuses it across all iterations,
        # avoiding expensive MuJoCo model recompilation.
        # Workers always use CPU (they do single-sample inference, no batching benefit)
        print(f"Creating {self.n_proc} persistent rollout workers...")

        # Create CPU copies for workers (deepcopy to avoid reference issues)
        if self.device.type == "cuda":
            # Networks are on GPU, need CPU copies for workers
            policy_cpu = deepcopy(self.policy).cpu()
            critic_cpu = deepcopy(self.critic).cpu()
            # Move non-parameter tensors to CPU
            policy_cpu.obs_mean = policy_cpu.obs_mean.cpu()
            policy_cpu.obs_std = policy_cpu.obs_std.cpu()
            critic_cpu.obs_mean = critic_cpu.obs_mean.cpu()
            critic_cpu.obs_std = critic_cpu.obs_std.cpu()
            if not isinstance(policy_cpu.stds, torch.nn.Parameter):
                policy_cpu.stds = policy_cpu.stds.cpu()
        else:
            # Already on CPU
            policy_cpu = self.policy
            critic_cpu = self.critic

        self.workers = [
            RolloutWorker.remote(
                env_fn,
                policy_cpu,
                critic_cpu,
                seed=get_worker_seed(self.seed, i) if self.seed is not None else None,
                worker_id=i,
            )
            for i in range(self.n_proc)
        ]
        print("Workers created successfully.")

    def _sync_obs_normalization(self, obs_mean, obs_std, include_old_policy=True):
        """Sync observation normalization params to all networks.

        This is the single point of truth for updating normalization parameters,
        avoiding scattered manual synchronization throughout the codebase.

        Args:
            obs_mean: Observation mean tensor
            obs_std: Observation std tensor
            include_old_policy: Whether to also sync to old_policy (for PPO ratio)
        """
        self.policy.obs_mean = obs_mean
        self.policy.obs_std = obs_std
        self.critic.obs_mean = obs_mean
        self.critic.obs_std = obs_std
        if include_old_policy:
            self.old_policy.obs_mean = obs_mean.clone()
            self.old_policy.obs_std = obs_std.clone()

    def sample_parallel_with_workers(self, deterministic=False):
        """Sample trajectories using persistent worker actors.

        This method uses pre-created Ray actors that maintain persistent environments,
        avoiding the expensive environment recreation that happens with stateless tasks.
        """
        max_steps = self.batch_size // self.n_proc

        # Get state dicts and obs normalization, move to CPU for workers
        # (Workers always run on CPU, even if main process is on GPU)
        policy_state_dict = {k: v.cpu() for k, v in self.policy.state_dict().items()}
        critic_state_dict = {k: v.cpu() for k, v in self.critic.state_dict().items()}
        obs_mean_cpu = self.policy.obs_mean.cpu()
        obs_std_cpu = self.policy.obs_std.cpu()

        # Use ray.put() to store in object store once, avoiding redundant
        # serialization when broadcasting to multiple workers
        policy_ref = ray.put(policy_state_dict)
        critic_ref = ray.put(critic_state_dict)
        obs_mean_ref = ray.put(obs_mean_cpu)
        obs_std_ref = ray.put(obs_std_cpu)

        # Sync all state to workers in a single call (weights, normalization, iteration)
        sync_futures = [
            w.sync_state.remote(policy_ref, critic_ref, obs_mean_ref, obs_std_ref, self.iteration_count)
            for w in self.workers
        ]
        ray.get(sync_futures)

        # Collect samples from all workers in parallel
        sample_futures = [
            w.sample.remote(self.gamma, max_steps, self.max_traj_len, deterministic) for w in self.workers
        ]
        result = ray.get(sample_futures)

        return self._aggregate_results(result)

    def _aggregate_results(self, result):
        # Aggregate results - handle traj_idx specially for recurrent policies
        # (indices need to be offset to reference correct positions in concatenated data)
        data_keys = ["states", "actions", "rewards", "values", "returns", "dones"]
        aggregated_data = {k: torch.cat([r[k] for r in result]) for k in data_keys}

        # Concatenate scalar metrics directly
        aggregated_data["ep_lens"] = torch.cat([r["ep_lens"] for r in result])
        aggregated_data["ep_rewards"] = torch.cat([r["ep_rewards"] for r in result])

        # Fix traj_idx: offset each worker's indices by cumulative sample count
        if self.recurrent:
            traj_idx_list = []
            offset = 0
            for r in result:
                # Skip the first 0 from subsequent workers (it's redundant)
                worker_traj_idx = r["traj_idx"]
                if offset > 0:
                    worker_traj_idx = worker_traj_idx[1:]  # Skip leading 0
                traj_idx_list.append(worker_traj_idx + offset)
                offset += len(r["states"])
            aggregated_data["traj_idx"] = torch.cat(traj_idx_list)
        else:
            aggregated_data["traj_idx"] = torch.cat([r["traj_idx"] for r in result])

        class Data:
            def __init__(self, data):
                for key, value in data.items():
                    setattr(self, key, value)

        data = Data(aggregated_data)
        return data

    def update_actor_critic(
        self, obs_batch, action_batch, return_batch, advantage_batch, mask, mirror_observation=None, mirror_action=None
    ):
        pdf = self.policy.distribution(obs_batch)
        log_probs = pdf.log_prob(action_batch).sum(-1, keepdim=True)

        old_pdf = self.old_policy.distribution(obs_batch)
        old_log_probs = old_pdf.log_prob(action_batch).sum(-1, keepdim=True)

        # ratio between old and new policy, should be one at the first iteration
        ratio = (log_probs - old_log_probs).exp()

        # clipped surrogate loss
        cpi_loss = ratio * advantage_batch * mask
        clip_loss = ratio.clamp(1.0 - self.clip, 1.0 + self.clip) * advantage_batch * mask
        actor_loss = -torch.min(cpi_loss, clip_loss).mean()

        # only used for logging
        clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip).float()).item()

        # Value loss using the TD(gae_lambda) target
        values = self.critic(obs_batch)
        critic_loss = F.mse_loss(return_batch, values)

        # Entropy loss favor exploration
        entropy_penalty = -(pdf.entropy() * mask).mean()

        # Mirror Symmetry Loss
        # Reuse mean from distribution instead of redundant forward pass
        deterministic_actions = pdf.mean
        if mirror_observation is not None and mirror_action is not None:
            if self.recurrent:
                mir_obs = torch.stack([mirror_observation(obs_batch[i, :, :]) for i in range(obs_batch.shape[0])])
                mirror_actions = self.policy(mir_obs)
            else:
                mir_obs = mirror_observation(obs_batch)
                mirror_actions = self.policy(mir_obs)
            mirror_actions = mirror_action(mirror_actions)
            mirror_loss = (deterministic_actions - mirror_actions).pow(2).mean()
        else:
            mirror_loss = torch.zeros_like(actor_loss)

        # imitation loss
        if self.base_policy is not None:
            imitation_loss = (self.base_policy(obs_batch) - deterministic_actions).pow(2).mean()
        else:
            imitation_loss = torch.zeros_like(actor_loss)

        # Calculate approximate form of reverse KL Divergence for early stopping
        # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
        # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
        # and Schulman blog: http://joschu.net/blog/kl-approx.html
        with torch.no_grad():
            log_ratio = log_probs - old_log_probs
            approx_kl_div = torch.mean((ratio - 1) - log_ratio)

        # Combined loss for single backward pass (avoids retain_graph=True overhead)
        actor_total_loss = (
            actor_loss
            + self.mirror_coeff * mirror_loss
            + self.imitate_coeff * imitation_loss
            + self.ent_coeff * entropy_penalty
        )
        total_loss = actor_total_loss + critic_loss

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        total_loss.backward()

        # Clip the gradient norm to prevent "unlucky" minibatches from
        # causing pathological updates
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return (
            actor_loss,
            entropy_penalty,
            critic_loss,
            approx_kl_div,
            mirror_loss,
            imitation_loss,
            clip_fraction,
        )

    def evaluate(self, env_fn, nets, itr, num_batches=5):
        # set all nets to .eval() mode
        for net in nets.values():
            net.eval()

        # collect some batches of data using persistent workers
        eval_batches = []
        for _ in range(num_batches):
            batch = self.sample_parallel_with_workers(deterministic=True)
            eval_batches.append(batch)

        # calculate average evaluation reward
        eval_ep_rewards = [float(i) for batch in eval_batches for i in batch.ep_rewards]
        avg_eval_ep_rewards = np.mean(eval_ep_rewards)

        # save checkpoint - saves with suffix and as best if improved
        self.checkpointer.save_if_best(nets, avg_eval_ep_rewards, itr)

        return eval_batches

    def train(self, env_fn, n_itr):
        self.actor_optimizer = optim.Adam(self.policy.parameters(), lr=self.lr, eps=self.eps)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr, eps=self.eps)

        train_start_time = time.time()

        obs_mirr, act_mirr = None, None
        if hasattr(env_fn(), "mirror_observation"):
            obs_mirr = env_fn().mirror_clock_observation

        if hasattr(env_fn(), "mirror_action"):
            act_mirr = env_fn().mirror_action

        # Warmup phase for running observation normalization
        if self.obs_rms is not None:
            print("Warming up observation normalization...")
            print(f"  Initial policy norm - mean: {self.policy.obs_mean[:3]}..., std: {self.policy.obs_std[:3]}...")
            warmup_batches = 5
            for i in range(warmup_batches):
                batch = self.sample_parallel_with_workers()
                self.obs_rms.update(batch.states.numpy())
                print(f"  Warmup batch {i + 1}: {len(batch.states)} samples, obs_rms count: {self.obs_rms.count:.0f}")
            # Sync warmed-up normalization to all networks
            with torch.no_grad():
                obs_mean = torch.from_numpy(self.obs_rms.mean).float().to(self.device)
                obs_std = torch.from_numpy(self.obs_rms.std).float().to(self.device)
                self._sync_obs_normalization(obs_mean, obs_std)
            print(f"Normalization initialized with {self.obs_rms.count:.0f} samples")
            print(f"  obs_mean range: [{obs_mean.min():.4f}, {obs_mean.max():.4f}]")
            print(f"  obs_std range: [{obs_std.min():.4f}, {obs_std.max():.4f}]")

        for itr in range(n_itr):
            print(f"********** Iteration {itr} ************")

            self.policy.train()
            self.critic.train()

            # set iteration count (could be used for curriculum training)
            self.iteration_count = itr

            sample_start_time = time.time()
            # Use persistent workers instead of stateless Ray tasks
            # This avoids expensive environment recreation each iteration
            batch = self.sample_parallel_with_workers()

            # Move batch to device for training
            observations = batch.states.float().to(self.device)
            actions = batch.actions.float().to(self.device)
            returns = batch.returns.float().to(self.device)
            values = batch.values.float().to(self.device)

            num_samples = len(observations)
            sample_time = time.time() - sample_start_time
            print(f"Sampling took {sample_time:.2f}s for {num_samples} steps.")

            # Normalize advantage (on device)
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps)

            minibatch_size = self.minibatch_size or num_samples
            self.total_steps += num_samples

            self.old_policy.load_state_dict(self.policy.state_dict())
            # Sync obs normalization to old_policy (not in state_dict, policy/critic already correct)
            self.old_policy.obs_mean = self.policy.obs_mean.clone()
            self.old_policy.obs_std = self.policy.obs_std.clone()

            optimizer_start_time = time.time()

            actor_losses = []
            entropies = []
            critic_losses = []
            kls = []
            mirror_losses = []
            imitation_losses = []
            clip_fractions = []
            for epoch in range(self.epochs):
                # Create seeded generator for deterministic batch sampling
                if self.seed is not None:
                    g = torch.Generator()
                    g.manual_seed(self.seed + itr * self.epochs + epoch)
                else:
                    g = None

                if self.recurrent:
                    random_indices = SubsetRandomSampler(range(len(batch.traj_idx) - 1), generator=g)
                    sampler = BatchSampler(random_indices, minibatch_size, drop_last=False)
                else:
                    random_indices = SubsetRandomSampler(range(num_samples), generator=g)
                    sampler = BatchSampler(random_indices, minibatch_size, drop_last=True)

                for indices in sampler:
                    if self.recurrent:
                        obs_batch = [observations[int(batch.traj_idx[i]) : int(batch.traj_idx[i + 1])] for i in indices]
                        action_batch = [actions[int(batch.traj_idx[i]) : int(batch.traj_idx[i + 1])] for i in indices]
                        return_batch = [returns[int(batch.traj_idx[i]) : int(batch.traj_idx[i + 1])] for i in indices]
                        advantage_batch = [
                            advantages[int(batch.traj_idx[i]) : int(batch.traj_idx[i + 1])] for i in indices
                        ]
                        mask = [torch.ones_like(r) for r in return_batch]

                        obs_batch = pad_sequence(obs_batch, batch_first=False)
                        action_batch = pad_sequence(action_batch, batch_first=False)
                        return_batch = pad_sequence(return_batch, batch_first=False)
                        advantage_batch = pad_sequence(advantage_batch, batch_first=False)
                        mask = pad_sequence(mask, batch_first=False)
                    else:
                        obs_batch = observations[indices]
                        action_batch = actions[indices]
                        return_batch = returns[indices]
                        advantage_batch = advantages[indices]
                        mask = 1

                    scalars = self.update_actor_critic(
                        obs_batch,
                        action_batch,
                        return_batch,
                        advantage_batch,
                        mask,
                        mirror_observation=obs_mirr,
                        mirror_action=act_mirr,
                    )
                    (
                        actor_loss,
                        entropy_penalty,
                        critic_loss,
                        approx_kl_div,
                        mirror_loss,
                        imitation_loss,
                        clip_fraction,
                    ) = scalars

                    actor_losses.append(actor_loss.item())
                    entropies.append(entropy_penalty.item())
                    critic_losses.append(critic_loss.item())
                    kls.append(approx_kl_div.item())
                    mirror_losses.append(mirror_loss.item())
                    imitation_losses.append(imitation_loss.item())
                    clip_fractions.append(clip_fraction)

            optimize_time = time.time() - optimizer_start_time
            print(f"Optimizer took: {optimize_time:.2f}s")

            action_noise = self.policy.stds.data.tolist()

            sys.stdout.write("-" * 37 + "\n")
            sys.stdout.write(f"| {'Mean Eprew':>15} | {torch.mean(batch.ep_rewards):>15.5g} |\n")
            sys.stdout.write(f"| {'Mean Eplen':>15} | {torch.mean(batch.ep_lens):>15.5g} |\n")
            sys.stdout.write(f"| {'Actor loss':>15} | {np.mean(actor_losses):>15.3g} |\n")
            sys.stdout.write(f"| {'Critic loss':>15} | {np.mean(critic_losses):>15.3g} |\n")
            sys.stdout.write(f"| {'Mirror loss':>15} | {np.mean(mirror_losses):>15.3g} |\n")
            sys.stdout.write(f"| {'Imitation loss':>15} | {np.mean(imitation_losses):>15.3g} |\n")
            sys.stdout.write(f"| {'Mean KL Div':>15} | {np.mean(kls):>15.3g} |\n")
            sys.stdout.write(f"| {'Mean Entropy':>15} | {np.mean(entropies):>15.3g} |\n")
            sys.stdout.write(f"| {'Clip Fraction':>15} | {np.mean(clip_fractions):>15.3g} |\n")
            sys.stdout.write(f"| {'Mean noise std':>15} | {np.mean(action_noise):>15.3g} |\n")
            sys.stdout.write("-" * 37 + "\n")
            sys.stdout.flush()

            total_time = time.time() - train_start_time
            fps = self.total_steps / total_time
            iter_avg = total_time / (itr + 1)
            ETA = round((n_itr - itr) * iter_avg)
            print(
                f"Total time elapsed: {total_time:.2f}s. Total steps: {self.total_steps} "
                f"(fps={fps:.2f}. iter-avg={iter_avg:.2f}s. "
                f"ETA={datetime.timedelta(seconds=ETA)})"
            )

            # To save time, perform evaluation only after 100 iters
            if itr == 0 or (itr + 1) % self.eval_freq == 0:
                nets = {"actor": self.policy, "critic": self.critic}

                evaluate_start = time.time()
                eval_batches = self.evaluate(env_fn, nets, itr)
                eval_time = time.time() - evaluate_start

                eval_ep_lens = [float(i) for b in eval_batches for i in b.ep_lens]
                eval_ep_rewards = [float(i) for b in eval_batches for i in b.ep_rewards]
                avg_eval_ep_lens = np.mean(eval_ep_lens)
                avg_eval_ep_rewards = np.mean(eval_ep_rewards)
                print("====EVALUATE EPISODE====")
                print(
                    f"(Episode length:{avg_eval_ep_lens:.3f}. Reward:{avg_eval_ep_rewards:.3f}. "
                    f"Time taken:{eval_time:.2f}s)"
                )

                # tensorboard logging for evaluation
                self.logger.log_eval_metrics(avg_eval_ep_rewards, avg_eval_ep_lens, itr)

            # tensorboard logging for training
            self.logger.log_training_metrics(
                actor_loss=np.mean(actor_losses),
                critic_loss=np.mean(critic_losses),
                mirror_loss=np.mean(mirror_losses),
                imitation_loss=np.mean(imitation_losses),
                mean_reward=float(torch.mean(batch.ep_rewards)),
                mean_ep_len=float(torch.mean(batch.ep_lens)),
                mean_noise_std=np.mean(action_noise),
                step=itr,
            )

            # tensorboard logging for timing/performance metrics
            self.logger.log_timing_metrics(
                fps=fps,
                sample_time=sample_time,
                optimize_time=optimize_time,
                total_time=total_time,
                step=itr,
            )

            # Note: Running observation normalization is fixed after warmup
            # to maintain training stability. The normalization params were
            # initialized during the warmup phase before training started.
