# Modified from https://github.com/openai/baselines/blob/master/baselines/common/vec_env/dummy_vec_env.py
# Thanks to the authors + OpenAI for the code

import numpy as np

class Vectorize:
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]

        self._observation_space = env.observation_space
        self._action_space = env.action_space
       
        self.ts = np.zeros(len(self.envs), dtype='int')  
      
    def step(self, action_n):
        results = [env.step(a) for (a,env) in zip(action_n, self.envs)]
        obs, rews, dones, infos = map(np.array, zip(*results))

        # TODO: decide whether to uncomment this
        self.ts += 1
        # for (i, done) in enumerate(dones):
        #     if done:
        #         obs[i] = self.envs[i].reset()
        #         self.ts[i] = 0

        return np.array(obs), np.array(rews), np.array(dones), infos

    def reset(self):        
        results = [env.reset() for env in self.envs]
        return np.array(results)

    def render(self):
        self.envs[0].render()

    @property
    def num_envs(self):
        return len(self.envs)

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    