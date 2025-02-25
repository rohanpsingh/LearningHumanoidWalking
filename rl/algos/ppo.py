"""Proximal Policy Optimization (clip objective)."""
from copy import deepcopy

import torch
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
import sys
import time
import numpy as np
import datetime

import ray

from rl.storage.rollout_storage import PPOBuffer
from rl.policies.actor import Gaussian_FF_Actor, Gaussian_LSTM_Actor
from rl.policies.critic import FF_V, LSTM_V
from rl.envs.normalize import get_normalization_params

class PPO:
    def __init__(self, env_fn, args):
        self.gamma          = args.gamma
        self.lam            = args.lam
        self.lr             = args.lr
        self.eps            = args.eps
        self.ent_coeff      = args.entropy_coeff
        self.clip           = args.clip
        self.minibatch_size = args.minibatch_size
        self.epochs         = args.epochs
        self.max_traj_len   = args.max_traj_len
        self.use_gae        = args.use_gae
        self.n_proc         = args.num_procs
        self.grad_clip      = args.max_grad_norm
        self.mirror_coeff   = args.mirror_coeff
        self.eval_freq      = args.eval_freq
        self.recurrent      = args.recurrent
        self.imitate_coeff  = args.imitate_coeff

        # batch_size depends on number of parallel envs
        self.batch_size = self.n_proc * self.max_traj_len

        self.total_steps = 0
        self.highest_reward = -np.inf

        # counter for training iterations
        self.iteration_count = 0

        # directory logging and saving weights
        self.save_path = Path(args.logdir)
        Path.mkdir(self.save_path, parents=True, exist_ok=True)

        # create the summarywriter
        self.writer = SummaryWriter(log_dir=self.save_path, flush_secs=10)

        # create networks or load up pretrained
        obs_dim = env_fn().observation_space.shape[0]
        action_dim = env_fn().action_space.shape[0]
        if args.continued:
            path_to_actor = args.continued
            path_to_critic = Path(args.continued.parent, "critic" + str(args.continued).split('actor')[1])
            policy = torch.load(path_to_actor, weights_only=False)
            critic = torch.load(path_to_critic, weights_only=False)
            # policy action noise parameters are initialized from scratch and not loaded
            if args.learn_std:
                policy.stds = torch.nn.Parameter(args.std_dev * torch.ones(action_dim))
            else:
                policy.stds = args.std_dev * torch.ones(action_dim)
            print("Loaded (pre-trained) actor from: ", path_to_actor)
            print("Loaded (pre-trained) critic from: ", path_to_critic)
        else:
            if args.recurrent:
                policy = Gaussian_LSTM_Actor(obs_dim, action_dim, init_std=args.std_dev,
                                             learn_std=args.learn_std)
                critic = LSTM_V(obs_dim)
            else:
                policy = Gaussian_FF_Actor(obs_dim, action_dim, init_std=args.std_dev,
                                           learn_std=args.learn_std, bounded=False)
                critic = FF_V(obs_dim)

            if hasattr(env_fn(), 'obs_mean') and hasattr(env_fn(), 'obs_std'):
                obs_mean, obs_std = env_fn().obs_mean, env_fn().obs_std
            else:
                obs_mean, obs_std = get_normalization_params(iter=args.input_norm_steps,
                                                             noise_std=1,
                                                             policy=policy,
                                                             env_fn=env_fn,
                                                             procs=args.num_procs)
            with torch.no_grad():
                policy.obs_mean, policy.obs_std = map(torch.Tensor, (obs_mean, obs_std))
                critic.obs_mean = policy.obs_mean
                critic.obs_std = policy.obs_std

        base_policy = None
        if args.imitate:
            base_policy = torch.load(args.imitate, weights_only=False)

        self.old_policy = deepcopy(policy)
        self.policy = policy
        self.critic = critic
        self.base_policy = base_policy

    @staticmethod
    def save(nets, save_path, suffix=""):
        filetype = ".pt"
        for name, net in nets.items():
            path = Path(save_path, name + suffix + filetype)
            torch.save(net, path)
            print("Saved {} at {}".format(name, path))
        return

    @ray.remote
    @torch.no_grad()
    @staticmethod
    def sample(env_fn, policy, critic, gamma, lam, iteration_count, max_steps, max_traj_len, deterministic):
        """
        Sample max_steps number of total timesteps, truncating
        trajectories if they exceed max_traj_len number of timesteps.
        """
        env = env_fn()
        env.robot.iteration_count = iteration_count

        memory = PPOBuffer(policy.state_dim, policy.action_dim, gamma, lam, size=max_traj_len*2)
        memory_full = False

        while not memory_full:
            state = torch.tensor(env.reset(), dtype=torch.float)
            done = False
            traj_len = 0

            if hasattr(policy, 'init_hidden_state'):
                policy.init_hidden_state()

            if hasattr(critic, 'init_hidden_state'):
                critic.init_hidden_state()

            while not done and traj_len < max_traj_len:
                action = policy(state, deterministic=deterministic)
                value = critic(state)

                next_state, reward, done, _ = env.step(action.numpy().copy())

                reward = torch.tensor(reward, dtype=torch.float)
                memory.store(state, action, reward, value, done)
                memory_full = (len(memory) >= max_steps)

                state = torch.tensor(next_state, dtype=torch.float)
                traj_len += 1

                #if memory_full:
                #   break

            value = critic(state)
            memory.finish_path(last_val=(not done) * value)

        return memory.get_data()

    def sample_parallel(self, *args, deterministic=False):

        max_steps = (self.batch_size // self.n_proc)
        worker_args = (self.gamma, self.lam, self.iteration_count, max_steps, self.max_traj_len, deterministic)
        args = args + worker_args

        # Create pool of workers, each getting data for min_steps
        worker = self.sample
        workers = [worker.remote(*args) for _ in range(self.n_proc)]
        result = ray.get(workers)

        # Aggregate results
        keys = result[0].keys()
        aggregated_data = {
            k: torch.cat([r[k] for r in result]) for k in keys
        }

        class Data:
            def __init__(self, data):
                for key, value in data.items():
                    setattr(self, key, value)
        data = Data(aggregated_data)
        return data

    def update_actor_critic(self, obs_batch, action_batch, return_batch, advantage_batch, mask, mirror_observation=None, mirror_action=None):

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
        deterministic_actions = self.policy(obs_batch)
        if mirror_observation is not None and mirror_action is not None:
            if self.recurrent:
                mir_obs = torch.stack([mirror_observation(obs_batch[i,:,:]) for i in range(obs_batch.shape[0])])
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

        self.actor_optimizer.zero_grad()
        (actor_loss + self.mirror_coeff*mirror_loss + self.imitate_coeff*imitation_loss + self.ent_coeff*entropy_penalty).backward(retain_graph=True)

        # Clip the gradient norm to prevent "unlucky" minibatches from
        # causing pathological updates
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)

        # Clip the gradient norm to prevent "unlucky" minibatches from
        # causing pathological updates
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
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

        # collect some batches of data
        eval_batches = []
        for _ in range(num_batches):
            batch = self.sample_parallel(env_fn, *nets.values(), deterministic=True)
            eval_batches.append(batch)

        # save all the networks
        self.save(nets, self.save_path, "_" + repr(itr))

        # save as actor.pt, if it is best
        eval_ep_rewards = [float(i) for i in batch.ep_rewards for batch in eval_batches]
        avg_eval_ep_rewards = np.mean(eval_ep_rewards)
        if self.highest_reward < avg_eval_ep_rewards:
            self.highest_reward = avg_eval_ep_rewards
            self.save(nets, self.save_path)

        return eval_batches

    def train(self, env_fn, n_itr):

        self.actor_optimizer = optim.Adam(self.policy.parameters(), lr=self.lr, eps=self.eps)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr, eps=self.eps)

        train_start_time = time.time()

        obs_mirr, act_mirr = None, None
        if hasattr(env_fn(), 'mirror_observation'):
            obs_mirr = env_fn().mirror_clock_observation

        if hasattr(env_fn(), 'mirror_action'):
            act_mirr = env_fn().mirror_action

        for itr in range(n_itr):
            print("********** Iteration {} ************".format(itr))

            self.policy.train()
            self.critic.train()

            # set iteration count (could be used for curriculum training)
            self.iteration_count = itr

            sample_start_time = time.time()
            policy_ref = ray.put(self.policy)
            critic_ref = ray.put(self.critic)
            batch = self.sample_parallel(env_fn, policy_ref, critic_ref)
            observations = batch.states.float()
            actions = batch.actions.float()
            returns = batch.returns.float()
            values = batch.values.float()

            num_samples = len(observations)
            elapsed = time.time() - sample_start_time
            print("Sampling took {:.2f}s for {} steps.".format(elapsed, num_samples))

            # Normalize advantage
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps)

            minibatch_size = self.minibatch_size or num_samples
            self.total_steps += num_samples

            self.old_policy.load_state_dict(self.policy.state_dict())

            optimizer_start_time = time.time()

            actor_losses = []
            entropies = []
            critic_losses = []
            kls = []
            mirror_losses = []
            imitation_losses = []
            clip_fractions = []
            for epoch in range(self.epochs):
                if self.recurrent:
                    random_indices = SubsetRandomSampler(range(len(batch.traj_idx)-1))
                    sampler = BatchSampler(random_indices, minibatch_size, drop_last=False)
                else:
                    random_indices = SubsetRandomSampler(range(num_samples))
                    sampler = BatchSampler(random_indices, minibatch_size, drop_last=True)

                for indices in sampler:
                    if self.recurrent:
                        obs_batch       = [observations[int(batch.traj_idx[i]):int(batch.traj_idx[i+1])] for i in indices]
                        action_batch    = [actions[int(batch.traj_idx[i]):int(batch.traj_idx[i+1])] for i in indices]
                        return_batch    = [returns[int(batch.traj_idx[i]):int(batch.traj_idx[i+1])] for i in indices]
                        advantage_batch = [advantages[int(batch.traj_idx[i]):int(batch.traj_idx[i+1])] for i in indices]
                        mask            = [torch.ones_like(r) for r in return_batch]

                        obs_batch       = pad_sequence(obs_batch, batch_first=False)
                        action_batch    = pad_sequence(action_batch, batch_first=False)
                        return_batch    = pad_sequence(return_batch, batch_first=False)
                        advantage_batch = pad_sequence(advantage_batch, batch_first=False)
                        mask            = pad_sequence(mask, batch_first=False)
                    else:
                        obs_batch       = observations[indices]
                        action_batch    = actions[indices]
                        return_batch    = returns[indices]
                        advantage_batch = advantages[indices]
                        mask            = 1

                    scalars = self.update_actor_critic(obs_batch, action_batch, return_batch, advantage_batch, mask, mirror_observation=obs_mirr, mirror_action=act_mirr)
                    actor_loss, entropy_penalty, critic_loss, approx_kl_div, mirror_loss, imitation_loss, clip_fraction = scalars

                    actor_losses.append(actor_loss.item())
                    entropies.append(entropy_penalty.item())
                    critic_losses.append(critic_loss.item())
                    kls.append(approx_kl_div.item())
                    mirror_losses.append(mirror_loss.item())
                    imitation_losses.append(imitation_loss.item())
                    clip_fractions.append(clip_fraction)

            elapsed = time.time() - optimizer_start_time
            print("Optimizer took: {:.2f}s".format(elapsed))

            action_noise = self.policy.stds.data.tolist()

            sys.stdout.write("-" * 37 + "\n")
            sys.stdout.write("| %15s | %15s |" % ('Mean Eprew', "%8.5g" % torch.mean(batch.ep_rewards)) + "\n")
            sys.stdout.write("| %15s | %15s |" % ('Mean Eplen', "%8.5g" % torch.mean(batch.ep_lens)) + "\n")
            sys.stdout.write("| %15s | %15s |" % ('Actor loss', "%8.3g" % np.mean(actor_losses)) + "\n")
            sys.stdout.write("| %15s | %15s |" % ('Critic loss', "%8.3g" % np.mean(critic_losses)) + "\n")
            sys.stdout.write("| %15s | %15s |" % ('Mirror loss', "%8.3g" % np.mean(mirror_losses)) + "\n")
            sys.stdout.write("| %15s | %15s |" % ('Imitation loss', "%8.3g" % np.mean(imitation_losses)) + "\n")
            sys.stdout.write("| %15s | %15s |" % ('Mean KL Div', "%8.3g" % np.mean(kls)) + "\n")
            sys.stdout.write("| %15s | %15s |" % ('Mean Entropy', "%8.3g" % np.mean(entropies)) + "\n")
            sys.stdout.write("| %15s | %15s |" % ('Clip Fraction', "%8.3g" % np.mean(clip_fractions)) + "\n")
            sys.stdout.write("| %15s | %15s |" % ('Mean noise std', "%8.3g" % np.mean(action_noise)) + "\n")
            sys.stdout.write("-" * 37 + "\n")
            sys.stdout.flush()

            elapsed = time.time() - train_start_time
            iter_avg = elapsed/(itr+1)
            ETA = round((n_itr - itr)*iter_avg)
            print("Total time elapsed: {:.2f}s. Total steps: {} (fps={:.2f}. iter-avg={:.2f}s. ETA={})".format(
                elapsed, self.total_steps, self.total_steps/elapsed, iter_avg, datetime.timedelta(seconds=ETA)))

            # To save time, perform evaluation only after 100 iters
            if itr==0 or (itr+1)%self.eval_freq==0:
                nets = {"actor": self.policy, "critic": self.critic}

                evaluate_start = time.time()
                eval_batches = self.evaluate(env_fn, nets, itr)
                eval_time = time.time() - evaluate_start

                eval_ep_lens = [float(i) for b in eval_batches for i in b.ep_lens]
                eval_ep_rewards = [float(i) for b in eval_batches for i in b.ep_rewards]
                avg_eval_ep_lens = np.mean(eval_ep_lens)
                avg_eval_ep_rewards = np.mean(eval_ep_rewards)
                print("====EVALUATE EPISODE====")
                print("(Episode length:{:.3f}. Reward:{:.3f}. Time taken:{:.2f}s)".format(
                    avg_eval_ep_lens, avg_eval_ep_rewards, eval_time))

                # tensorboard logging
                self.writer.add_scalar("Eval/mean_reward", avg_eval_ep_rewards, itr)
                self.writer.add_scalar("Eval/mean_episode_length", avg_eval_ep_lens, itr)

            # tensorboard logging
            self.writer.add_scalar("Loss/actor", np.mean(actor_losses), itr)
            self.writer.add_scalar("Loss/critic", np.mean(critic_losses), itr)
            self.writer.add_scalar("Loss/mirror", np.mean(mirror_losses), itr)
            self.writer.add_scalar("Loss/imitation", np.mean(imitation_losses), itr)
            self.writer.add_scalar("Train/mean_reward", torch.mean(batch.ep_rewards), itr)
            self.writer.add_scalar("Train/mean_episode_length", torch.mean(batch.ep_lens), itr)
            self.writer.add_scalar("Train/mean_noise_std", np.mean(action_noise), itr)
