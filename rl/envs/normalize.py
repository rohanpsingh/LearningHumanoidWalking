# Modified from https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_normalize.py
# Thanks to the authors + OpenAI for the code

import numpy as np
import functools
import torch
import ray

from .wrappers import WrapEnv

@ray.remote
def _run_random_actions(iter, policy, env_fn, noise_std):

    env = WrapEnv(env_fn)
    states = np.zeros((iter, env.observation_space.shape[0]))

    state = env.reset()
    for t in range(iter):
        states[t, :] = state

        state = torch.Tensor(state)

        action = policy(state)

        # add gaussian noise to deterministic action
        action = action + torch.randn(action.size()) * noise_std

        state, _, done, _ = env.step(action.data.numpy())

        if done:
            state = env.reset()
    
    return states

def get_normalization_params(iter, policy, env_fn, noise_std, procs=4):
    print("Gathering input normalization data using {0} steps, noise = {1}...".format(iter, noise_std))

    states_ids = [_run_random_actions.remote(iter // procs, policy, env_fn, noise_std) for _ in range(procs)]

    states = []
    for _ in range(procs):
        ready_ids, _ = ray.wait(states_ids, num_returns=1)
        states.extend(ray.get(ready_ids[0]))
        states_ids.remove(ready_ids[0])

    print("Done gathering input normalization data.")

    return np.mean(states, axis=0), np.sqrt(np.var(states, axis=0) + 1e-8)


# returns a function that creates a normalized environment, then pre-normalizes it 
# using states sampled from a deterministic policy with some added noise
def PreNormalizer(iter, noise_std, policy, *args, **kwargs):

    # noise is gaussian noise
    @torch.no_grad()
    def pre_normalize(env, policy, num_iter, noise_std):
        # save whether or not the environment is configured to do online normalization
        online_val = env.online
        env.online = True

        state = env.reset()

        for t in range(num_iter):
            state = torch.Tensor(state)

            _, action = policy(state)

            # add gaussian noise to deterministic action
            action = action + torch.randn(action.size()) * noise_std

            state, _, done, _ = env.step(action.data.numpy())

            if done:
                state = env.reset()

        env.online = online_val
    
    def _Normalizer(venv):
        venv = Normalize(venv, *args, **kwargs)

        print("Gathering input normalization data using {0} steps, noise = {1}...".format(iter, noise_std))
        pre_normalize(venv, policy, iter, noise_std)
        print("Done gathering input normalization data.")

        return venv

    return _Normalizer

# returns a function that creates a normalized environment
def Normalizer(*args, **kwargs):
    def _Normalizer(venv):
        return Normalize(venv, *args, **kwargs)

    return _Normalizer

class Normalize:
    """
    Vectorized environment base class
    """
    def __init__(self, 
                 venv,
                 ob_rms=None, 
                 ob=True, 
                 ret=False, 
                 clipob=10., 
                 cliprew=10., 
                 online=True,
                 gamma=1.0, 
                 epsilon=1e-8):

        self.venv = venv
        self._observation_space = venv.observation_space
        self._action_space = venv.action_space

        if ob_rms is not None:
            self.ob_rms = ob_rms
        else:
            self.ob_rms = RunningMeanStd(shape=self._observation_space.shape) if ob else None

        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

        self.online = online

    def step(self, vac):
        obs, rews, news, infos = self.venv.step(vac)

        #self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)

        # NOTE: shifting mean of reward seems bad; qualitatively changes MDP
        if self.ret_rms: 
            if self.online:
                self.ret_rms.update(self.ret)
            
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)

        return obs, rews, news, infos

    def _obfilt(self, obs):
        if self.ob_rms: 
            if self.online:
                self.ob_rms.update(obs)
            
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        """
        Reset all environments
        """
        obs = self.venv.reset()
        return self._obfilt(obs)

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def close(self):
        self.venv.close()
    
    def render(self):
        self.venv.render()

    @property
    def num_envs(self):
        return self.venv.num_envs



class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.zeros(shape, 'float64')
        self.count = epsilon


    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count        
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count        

def test_runningmeanstd():
    for (x1, x2, x3) in [
        (np.random.randn(3), np.random.randn(4), np.random.randn(5)),
        (np.random.randn(3,2), np.random.randn(4,2), np.random.randn(5,2)),
        ]:

        rms = RunningMeanStd(epsilon=0.0, shape=x1.shape[1:])

        x = np.concatenate([x1, x2, x3], axis=0)
        ms1 = [x.mean(axis=0), x.var(axis=0)]
        rms.update(x1)
        rms.update(x2)
        rms.update(x3)
        ms2 = [rms.mean, rms.var]

        assert np.allclose(ms1, ms2)
