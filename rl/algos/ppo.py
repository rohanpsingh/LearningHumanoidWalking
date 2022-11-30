"""Proximal Policy Optimization (clip objective)."""
from copy import deepcopy

import torch
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import kl_divergence
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

import ray
from rl.envs import WrapEnv


class PPOBuffer:
    """
    A buffer for storing trajectory data and calculating returns for the policy
    and critic updates.
    This container is intentionally not optimized w.r.t. to memory allocation
    speed because such allocation is almost never a bottleneck for policy
    gradient.

    On the other hand, experience buffers are a frequent source of
    off-by-one errors and other bugs in policy gradient implementations, so
    this code is optimized for clarity and readability, at the expense of being
    (very) marginally slower than some other implementations.
    (Premature optimization is the root of all evil).
    """
    def __init__(self, gamma=0.99, lam=0.95, use_gae=False):
        self.states  = []
        self.actions = []
        self.rewards = []
        self.values  = []
        self.returns = []

        self.ep_returns = [] # for logging
        self.ep_lens    = []

        self.gamma, self.lam = gamma, lam

        self.ptr = 0
        self.traj_idx = [0]

    def __len__(self):
        return len(self.states)

    def storage_size(self):
        return len(self.states)

    def store(self, state, action, reward, value):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        # TODO: make sure these dimensions really make sense
        self.states  += [state.squeeze(0)]
        self.actions += [action.squeeze(0)]
        self.rewards += [reward.squeeze(0)]
        self.values  += [value.squeeze(0)]

        self.ptr += 1

    def finish_path(self, last_val=None):
        self.traj_idx += [self.ptr]
        rewards = self.rewards[self.traj_idx[-2]:self.traj_idx[-1]]

        returns = []

        R = last_val.squeeze(0).copy()  # Avoid copy?
        for reward in reversed(rewards):
            R = self.gamma * R + reward
            returns.insert(0, R)  # TODO: self.returns.insert(self.path_idx, R) ?
                                  # also technically O(k^2), may be worth just reversing list
                                  # BUG? This is adding copies of R by reference (?)

        self.returns += returns

        self.ep_returns += [np.sum(rewards)]
        self.ep_lens    += [len(rewards)]

    def get(self):
        return(
            self.states,
            self.actions,
            self.returns,
            self.values
        )

class PPO:
    def __init__(self, args, save_path):
        self.gamma          = args['gamma']
        self.lam            = args['lam']
        self.lr             = args['lr']
        self.eps            = args['eps']
        self.ent_coeff      = args['entropy_coeff']
        self.clip           = args['clip']
        self.minibatch_size = args['minibatch_size']
        self.epochs         = args['epochs']
        self.max_traj_len   = args['max_traj_len']
        self.use_gae        = args['use_gae']
        self.n_proc         = args['num_procs']
        self.grad_clip      = args['max_grad_norm']
        self.mirror_coeff   = args['mirror_coeff']
        self.eval_freq      = args['eval_freq']
        self.recurrent      = False

        # batch_size depends on number of parallel envs
        self.batch_size = self.n_proc * self.max_traj_len

        self.vf_coeff = 0.5
        self.target_kl = None # By default, there is no limit on the kl div

        self.total_steps = 0
        self.highest_reward = -1
        self.limit_cores = 0

        # counter for training iterations
        self.iteration_count = 0

        self.save_path = save_path
        self.eval_fn = os.path.join(self.save_path, 'eval.txt')
        with open(self.eval_fn, 'w') as out:
            out.write("test_ep_returns,test_ep_lens\n")

        self.train_fn = os.path.join(self.save_path, 'train.txt')
        with open(self.train_fn, 'w') as out:
            out.write("ep_returns,ep_lens\n")

        # os.environ['OMP_NUM_THREA DS'] = '1'
        # if args['redis_address'] is not None:
        #     ray.init(num_cpos=self.n_proc, redis_address=args['redis_address'])
        # else:
        #     ray.init(num_cpus=self.n_proc)

    def save(self, policy, critic, suffix=""):

        try:
            os.makedirs(self.save_path)
        except OSError:
            pass
        filetype = ".pt" # pytorch model
        torch.save(policy, os.path.join(self.save_path, "actor" + suffix + filetype))
        torch.save(critic, os.path.join(self.save_path, "critic" + suffix + filetype))

    @ray.remote
    @torch.no_grad()
    def sample(self, env_fn, policy, critic, max_steps, max_traj_len, deterministic=False, anneal=1.0, term_thresh=0):
        """
        Sample max_steps number of total timesteps, truncating
        trajectories if they exceed max_traj_len number of timesteps.
        """
        torch.set_num_threads(1)    # By default, PyTorch will use multiple cores to speed up operations.
                                    # This can cause issues when Ray also uses multiple cores, especially on machines
                                    # with a lot of CPUs. I observed a significant speedup when limiting PyTorch 
                                    # to a single core - I think it basically stopped ray workers from stepping on each
                                    # other's toes.

        env = WrapEnv(env_fn)  # TODO
        env.robot.iteration_count = self.iteration_count

        memory = PPOBuffer(self.gamma, self.lam)
        memory_full = False

        while not memory_full:
            state = torch.Tensor(env.reset())
            done = False
            traj_len = 0

            if hasattr(policy, 'init_hidden_state'):
                policy.init_hidden_state()

            if hasattr(critic, 'init_hidden_state'):
                critic.init_hidden_state()

            while not done and traj_len < max_traj_len:
                action = policy(state, deterministic=deterministic, anneal=anneal)
                value = critic(state)

                next_state, reward, done, _ = env.step(action.numpy())

                memory.store(state.numpy(), action.numpy(), reward, value.numpy())
                memory_full = (len(memory) == max_steps)

                state = torch.Tensor(next_state)
                traj_len += 1

                if memory_full:
                    break

            value = critic(state)
            memory.finish_path(last_val=(not done) * value.numpy())

        return memory

    def sample_parallel(self, env_fn, policy, critic, min_steps, max_traj_len, deterministic=False, anneal=1.0, term_thresh=0):

        worker = self.sample
        args = (self, env_fn, policy, critic, min_steps // self.n_proc, max_traj_len, deterministic, anneal, term_thresh)

        # Create pool of workers, each getting data for min_steps
        workers = [worker.remote(*args) for _ in range(self.n_proc)]
        result = ray.get(workers)

        # O(n)
        def merge(buffers):
            merged = PPOBuffer(self.gamma, self.lam)
            for buf in buffers:
                offset = len(merged)
                merged.states  += buf.states
                merged.actions += buf.actions
                merged.rewards += buf.rewards
                merged.values  += buf.values
                merged.returns += buf.returns

                merged.ep_returns += buf.ep_returns
                merged.ep_lens    += buf.ep_lens

                merged.traj_idx += [offset + i for i in buf.traj_idx[1:]]
                merged.ptr += buf.ptr

            return merged

        total_buf = merge(result)

        return total_buf

    def update_policy(self, obs_batch, action_batch, return_batch, advantage_batch, mask, mirror_observation=None, mirror_action=None):
        policy = self.policy
        critic = self.critic
        old_policy = self.old_policy

        values = critic(obs_batch)
        pdf = policy.distribution(obs_batch)
        log_probs = pdf.log_prob(action_batch).sum(-1, keepdim=True)

        old_pdf = old_policy.distribution(obs_batch)
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
        critic_loss = self.vf_coeff * F.mse_loss(return_batch, values)

        # Entropy loss favor exploration
        entropy_penalty = -(pdf.entropy() * mask).mean()

        # Mirror Symmetry Loss
        if mirror_observation is not None and mirror_action is not None:
            deterministic_actions = policy(obs_batch)
            mir_obs = mirror_observation(obs_batch)
            mirror_actions = policy(mir_obs)
            mirror_actions = mirror_action(mirror_actions)
            mirror_loss = (deterministic_actions - mirror_actions).pow(2).mean()
        else:
            mirror_loss = torch.Tensor([0])

        # Calculate approximate form of reverse KL Divergence for early stopping
        # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
        # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
        # and Schulman blog: http://joschu.net/blog/kl-approx.html
        with torch.no_grad():
            log_ratio = log_probs - old_log_probs
            approx_kl_div = torch.mean((ratio - 1) - log_ratio)

        return (
            actor_loss,
            entropy_penalty,
            critic_loss,
            approx_kl_div,
            mirror_loss,
            clip_fraction,
        )

    def train(self,
              env_fn,
              policy,
              critic,
              n_itr,
              anneal_rate=1.0):

        self.old_policy = deepcopy(policy)
        self.policy = policy
        self.critic = critic

        self.actor_optimizer = optim.Adam(policy.parameters(), lr=self.lr, eps=self.eps)
        self.critic_optimizer = optim.Adam(critic.parameters(), lr=self.lr, eps=self.eps)

        train_start_time = time.time()

        obs_mirr, act_mirr = None, None
        if hasattr(env_fn(), 'mirror_observation'):
            obs_mirr = env_fn().mirror_clock_observation

        if hasattr(env_fn(), 'mirror_action'):
            act_mirr = env_fn().mirror_action

        curr_anneal = 1.0
        curr_thresh = 0
        start_itr = 0
        ep_counter = 0
        do_term = False

        test_ep_lens = []
        test_ep_returns = []

        for itr in range(n_itr):
            print("********** Iteration {} ************".format(itr))

            # set iteration count (could be used for curriculum training)
            self.iteration_count = itr

            sample_start_time = time.time()
            if self.highest_reward > (2/3)*self.max_traj_len and curr_anneal > 0.5:
                curr_anneal *= anneal_rate
            if do_term and curr_thresh < 0.35:
                curr_thresh = .1 * 1.0006**(itr-start_itr)
            batch = self.sample_parallel(env_fn, self.policy, self.critic, self.batch_size, self.max_traj_len, anneal=curr_anneal, term_thresh=curr_thresh)
            observations, actions, returns, values = map(torch.Tensor, batch.get())

            num_samples = batch.storage_size()
            elapsed = time.time() - sample_start_time
            print("Sampling took {:.2f}s for {} steps.".format(elapsed, num_samples))

            # Normalize advantage
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps)

            minibatch_size = self.minibatch_size or num_samples
            self.total_steps += num_samples

            self.old_policy.load_state_dict(policy.state_dict())

            # Is false when 1.5*self.target_kl is breached
            continue_training = True

            optimizer_start_time = time.time()
            for epoch in range(self.epochs):
                actor_losses = []
                entropies = []
                critic_losses = []
                kls = []
                mirror_losses = []
                clip_fractions = []
                if self.recurrent:
                    random_indices = SubsetRandomSampler(range(len(batch.traj_idx)-1))
                    sampler = BatchSampler(random_indices, minibatch_size, drop_last=False)
                else:
                    random_indices = SubsetRandomSampler(range(num_samples))
                    sampler = BatchSampler(random_indices, minibatch_size, drop_last=True)

                for indices in sampler:
                    if self.recurrent:
                        obs_batch       = [observations[batch.traj_idx[i]:batch.traj_idx[i+1]] for i in indices]
                        action_batch    = [actions[batch.traj_idx[i]:batch.traj_idx[i+1]] for i in indices]
                        return_batch    = [returns[batch.traj_idx[i]:batch.traj_idx[i+1]] for i in indices]
                        advantage_batch = [advantages[batch.traj_idx[i]:batch.traj_idx[i+1]] for i in indices]
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

                    scalars = self.update_policy(obs_batch, action_batch, return_batch, advantage_batch, mask, mirror_observation=obs_mirr, mirror_action=act_mirr)
                    actor_loss, entropy_penalty, critic_loss, approx_kl_div, mirror_loss, clip_fraction = scalars

                    actor_losses.append(actor_loss.item())
                    entropies.append(entropy_penalty.item())
                    critic_losses.append(critic_loss.item())
                    kls.append(approx_kl_div.item())
                    mirror_losses.append(mirror_loss.item())
                    clip_fractions.append(clip_fraction)

                    if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                        continue_training = False
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                        break

                    self.actor_optimizer.zero_grad()
                    (actor_loss + self.mirror_coeff*mirror_loss + self.ent_coeff*entropy_penalty).backward()

                    # Clip the gradient norm to prevent "unlucky" minibatches from
                    # causing pathological updates
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), self.grad_clip)
                    self.actor_optimizer.step()

                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()

                    # Clip the gradient norm to prevent "unlucky" minibatches from
                    # causing pathological updates
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), self.grad_clip)
                    self.critic_optimizer.step()

                # Early stopping
                if not continue_training:
                    break

            elapsed = time.time() - optimizer_start_time
            print("Optimizer took: {:.2f}s".format(elapsed))

            if np.mean(batch.ep_lens) >= self.max_traj_len * 0.75:
                ep_counter += 1
            if do_term == False and ep_counter > 50:
                do_term = True
                start_itr = itr

            sys.stdout.write("-" * 37 + "\n")
            sys.stdout.write("| %15s | %15s |" % ('Return (batch)', "%8.5g" % np.mean(batch.ep_returns)) + "\n")
            sys.stdout.write("| %15s | %15s |" % ('Mean Eplen', "%8.5g" % np.mean(batch.ep_lens)) + "\n")
            sys.stdout.write("| %15s | %15s |" % ('Actor loss', "%8.3g" % np.mean(actor_losses)) + "\n")
            sys.stdout.write("| %15s | %15s |" % ('Critic loss', "%8.3g" % np.mean(critic_losses)) + "\n")
            sys.stdout.write("| %15s | %15s |" % ('Mirror loss', "%8.3g" % np.mean(mirror_losses)) + "\n")
            sys.stdout.write("| %15s | %15s |" % ('Mean KL Div', "%8.3g" % np.mean(kls)) + "\n")
            sys.stdout.write("| %15s | %15s |" % ('Mean Entropy', "%8.3g" % np.mean(entropies)) + "\n")
            sys.stdout.write("| %15s | %15s |" % ('Clip Fraction', "%8.3g" % np.mean(clip_fractions)) + "\n")
            sys.stdout.write("-" * 37 + "\n")
            sys.stdout.flush()

            elapsed = time.time() - train_start_time
            print("Total time elapsed: {:.2f}s. Total steps: {} (fps={:.2f})".format(elapsed, self.total_steps, self.total_steps/elapsed))

            # save metrics
            with open(self.train_fn, 'a') as out:
                out.write("{},{}\n".format(np.mean(batch.ep_returns), np.mean(batch.ep_lens)))

            # To save time, perform evaluation only after 100 iters
            if (itr+1)%self.eval_freq==0:
                # logger
                evaluate_start = time.time()
                test = self.sample_parallel(env_fn, self.policy, self.critic, self.batch_size, self.max_traj_len, deterministic=True)
                eval_time = time.time() - evaluate_start
                print("evaluate time elapsed: {:.2f} s".format(eval_time))

                avg_eval_reward = np.mean(test.ep_returns)
                print("====EVALUATE EPISODE====  (Return = {})".format(avg_eval_reward))

                # save metrics
                with open(self.eval_fn, 'a') as out:
                    out.write("{},{}\n".format(np.mean(test.ep_returns), np.mean(test.ep_lens)))
                test_ep_lens.append(np.mean(test.ep_lens))
                test_ep_returns.append(np.mean(test.ep_returns))
                plt.clf()
                xlabel = [i*self.eval_freq for i in range(len(test_ep_lens))]
                plt.plot(xlabel, test_ep_lens, color='blue', marker='o', label='Ep lens')
                plt.plot(xlabel, test_ep_returns, color='green', marker='o', label='Returns')
                plt.xticks(np.arange(0, itr+1, step=self.eval_freq))
                plt.xlabel('Iterations')
                plt.ylabel('Returns/Episode lengths')
                plt.legend()
                plt.grid()
                plt.savefig(os.path.join(self.save_path, 'eval.svg'), bbox_inches='tight')

                # save policy
                self.save(policy, critic, "_" + repr(itr))

                # save as actor.pt, if it is best
                if self.highest_reward < avg_eval_reward:
                    self.highest_reward = avg_eval_reward
                    self.save(policy, critic)
