import torch
import os
import sys, time
from functools import partial

import tty
import termios
import select
import numpy as np

class EvalProcessClass():
    def __init__(self, env):
        self.env = env
        return

    def eval_policy(self, policy, run_args):

        def print_input_update(e):
            print(f"\n\nstance dur.: {e.stance_duration:.2f}\t swing dur.: {e.swing_duration:.2f}\t stance mode: {e.stance_mode}\n")

        def isData():
            return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])
        
        max_traj_len = run_args.max_traj_len
        visualize = True
        env = partial(self.env)()
        
        if hasattr(policy, 'init_hidden_state'):
            policy.init_hidden_state()

        old_settings = termios.tcgetattr(sys.stdin)

        orient_add = 0

        slowmo = False

        if visualize:
            env.render()
            if hasattr(env, "viewer"):
                env.viewer._paused = True
        render_state = True
        try:
            tty.setcbreak(sys.stdin.fileno())

            state = env.reset()
            done = False
            timesteps = 0
            eval_reward = 0
            speed = 0.0
            side_speed = 0.0

            while render_state:
                if isData():
                    c = sys.stdin.read(1)
                    if c == 'w':
                        speed += 0.1
                    elif c == 's':
                        speed -= 0.1
                    elif c == 'd':
                        side_speed += 0.02
                    elif c == 'a':
                        side_speed -= 0.
                    elif c == 'j':
                        env.phase_add += .1
                        # print("Increasing frequency to: {:.1f}".format(env.phase_add))
                    elif c == 'h':
                        env.phase_add -= .1
                        # print("Decreasing frequency to: {:.1f}".format(env.phase_add))
                    elif c == 'l':
                        orient_add -= .1
                        # print("Increasing orient_add to: ", orient_add)
                    elif c == 'k':
                        orient_add += .1
                        # print("Decreasing orient_add to: ", orient_add)
                    elif c == 'x':
                        env.swing_duration += .01
                        print_input_update(env)
                    elif c == 'z':
                        env.swing_duration -= .01
                        print_input_update(env)
                    elif c == 'v':
                        env.stance_duration += .01
                        print_input_update(env)
                    elif c == 'c':
                        env.stance_duration -= .01
                        print_input_update(env)

                    elif c == '1':
                        env.stance_mode = "zero"
                        print_input_update(env)
                    elif c == '2':
                        env.stance_mode = "grounded"
                        print_input_update(env)
                    elif c == '3':
                        env.stance_mode = "aerial"
                    elif c == 'r':
                        state = env.reset()
                        speed = env.speed
                        if hasattr(policy, 'init_hidden_state'):
                            policy.init_hidden_state()
                        print("Resetting environment via env.reset()")
                    elif c == 'p':
                        push = 100
                        push_dir = 2
                        force_arr = np.zeros(6)
                        force_arr[push_dir] = push
                        env.sim.apply_force(force_arr)
                    elif c == 't':
                        slowmo = not slowmo
                        print("Slowmo : \n", slowmo)

                if hasattr(env, 'frame_skip'):
                    start = time.time()

                if (not env.viewer_is_paused()):
                    # Update Orientation
                    env.orient_add = orient_add

                    action = policy.forward(torch.Tensor(state), deterministic=True).detach().numpy()
                    #action = torch.randn(env.action_space.shape[0]).detach().numpy()

                    state, reward, done, _ = env.step(action)

                    eval_reward += reward
                    timesteps += 1

                if visualize:
                    env.render()
                if hasattr(env, 'frame_skip'):
                    end = time.time()
                    delaytime = max(0, env.frame_skip / 1000 - (end-start))
                    if slowmo:
                        while(time.time() - end < delaytime*10):
                            env.render()
                            time.sleep(delaytime)
                    else:
                        time.sleep(delaytime)
            print("Eval reward: ", eval_reward)

        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
