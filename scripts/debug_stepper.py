import os
import time
import argparse
import torch
import pickle
import mujoco
import numpy as np
import transforms3d as tf3
from run_experiment import import_env

def print_reward(ep_rewards):
    mean_rewards = {k:[] for k in ep_rewards[-1].keys()}
    print('*********************************')
    for key in mean_rewards.keys():
        l = [step[key] for step in ep_rewards]
        mean_rewards[key] = sum(l)/len(l)
        print(key, ': ', mean_rewards[key])
        #total_rewards = [r for step in ep_rewards for r in step.values()]
    print('*********************************')
    print("mean per step reward: ", sum(mean_rewards.values()))

def draw_stuff(task, viewer):
    # render
    arrow_size = [0.02, 0.02, 0.5]
    sphere = mujoco.mjtGeom.mjGEOM_SPHERE
    arrow = mujoco.mjtGeom.mjGEOM_ARROW
    if hasattr(task, 'sequence'):
        for idx, step in enumerate(task.sequence):
            step_pos = [step[0], step[1], step[2]]
            step_theta = step[3]
            if step_pos not in [task.sequence[task.t1][0:3].tolist(), task.sequence[task.t2][0:3].tolist()]:
                viewer.add_marker(pos=step_pos, size=np.ones(3)*0.05, rgba=np.array([0, 1, 1, 1]), type=sphere, label="")
                viewer.add_marker(pos=step_pos, mat=tf3.euler.euler2mat(0, np.pi/2, step_theta), size=arrow_size, rgba=np.array([0, 1, 1, 1]), type=arrow, label="")

        target_radius = task.target_radius
        step_pos = task.sequence[task.t1][0:3].tolist()
        step_theta = task.sequence[task.t1][3]
        viewer.add_marker(pos=step_pos, size=np.ones(3)*0.05, rgba=np.array([1, 0, 0, 1]), type=sphere, label="t1")
        viewer.add_marker(pos=step_pos, mat=tf3.euler.euler2mat(0, np.pi/2, step_theta), size=arrow_size, rgba=np.array([1, 0, 0, 1]), type=arrow, label="")
        viewer.add_marker(pos=step_pos, size=np.ones(3)*target_radius, rgba=np.array([1, 0, 0, 0.1]), type=sphere, label="")
        step_pos = task.sequence[task.t2][0:3].tolist()
        step_theta = task.sequence[task.t2][3]
        viewer.add_marker(pos=step_pos, size=np.ones(3)*0.05, rgba=np.array([0, 0, 1, 1]), type=sphere, label="t2")
        viewer.add_marker(pos=step_pos, mat=tf3.euler.euler2mat(0, np.pi/2, step_theta), size=arrow_size, rgba=np.array([0, 0, 1, 1]), type=arrow, label="")
        viewer.add_marker(pos=step_pos, size=np.ones(3)*target_radius, rgba=np.array([0, 0, 1, 0.1]), type=sphere, label="")

    goalx = task._goal_steps_x
    goaly = task._goal_steps_y
    goaltheta = task._goal_steps_theta
    viewer.add_marker(pos=[goalx[0], goaly[0], 0], size=np.ones(3)*0.05, rgba=np.array([0, 1, 1, 1]), type=sphere, label="G1")
    viewer.add_marker(pos=[goalx[0], goaly[0], 0], mat=tf3.euler.euler2mat(0, np.pi/2, goaltheta[0]), size=arrow_size, rgba=np.array([0, 1, 1, 1]), type=arrow, label="")
    viewer.add_marker(pos=[goalx[1], goaly[1], 0], size=np.ones(3)*0.05, rgba=np.array([0, 1, 1, 1]), type=sphere, label="G2")
    viewer.add_marker(pos=[goalx[1], goaly[1], 0], mat=tf3.euler.euler2mat(0, np.pi/2, goaltheta[1]), size=arrow_size, rgba=np.array([0, 1, 1, 1]), type=arrow, label="")

    # draw feet pose
    lfoot_orient = (tf3.quaternions.quat2mat(task.l_foot_quat)).dot(tf3.euler.euler2mat(0, np.pi/2, 0))
    rfoot_orient = (tf3.quaternions.quat2mat(task.r_foot_quat)).dot(tf3.euler.euler2mat(0, np.pi/2, 0))
    viewer.add_marker(pos=task.l_foot_pos, size=np.ones(3)*0.05, rgba=[0.5, 0.5, 0.5, 1], type=sphere, label="")
    viewer.add_marker(pos=task.l_foot_pos, mat=lfoot_orient, size=arrow_size, rgba=[0.5, 0.5, 0.5, 1], type=arrow, label="")
    viewer.add_marker(pos=task.r_foot_pos, size=np.ones(3)*0.05, rgba=[0.5, 0.5, 0.5, 1], type=sphere, label="")
    viewer.add_marker(pos=task.r_foot_pos, mat=rfoot_orient, size=arrow_size, rgba=[0.5, 0.5, 0.5, 1], type=arrow, label="")

    # draw origin
    viewer.add_marker(pos=[0, 0, 0], size=np.ones(3)*0.05, rgba=np.array([1, 1, 1, 1]), type=sphere, label="")
    viewer.add_marker(pos=[0, 0, 0], mat=tf3.euler.euler2mat(0, 0, 0), size=[0.01, 0.01, 2], rgba=np.array([0, 0, 1, 0.2]), type=arrow, label="")
    viewer.add_marker(pos=[0, 0, 0], mat=tf3.euler.euler2mat(0, np.pi/2, 0), size=[0.01, 0.01, 2], rgba=np.array([1, 0, 0, 0.2]), type=arrow, label="")
    viewer.add_marker(pos=[0, 0, 0], mat=tf3.euler.euler2mat(-np.pi/2, np.pi/2, 0), size=[0.01, 0.01, 2], rgba=np.array([0, 1, 0, 0.2]), type=arrow, label="")
    return

def run(env, policy):
    observation = env.reset()

    env.render()
    viewer = env.viewer
    viewer._paused = True
    done = False
    ts, end_ts = 0, 2000
    ep_rewards = []

    while (ts < end_ts) and (done == False):
        if hasattr(env, 'frame_skip'):
            start = time.time()

        with torch.no_grad():
            action = policy.forward(torch.Tensor(observation), deterministic=True).detach().numpy()

        observation, _, done, info = env.step(action.copy())
        ep_rewards.append(info)

        if env.__class__.__name__ == 'JvrcStepEnv':
            draw_stuff(env.task, viewer)
        env.render()

        if hasattr(env, 'frame_skip'):
            end = time.time()
            sim_dt = env.robot.client.sim_dt()
            delaytime = max(0, env.frame_skip / (1/sim_dt) - (end-start))
            time.sleep(delaytime)
        ts+=1

    print("Episode finished after {} timesteps".format(ts))
    print_reward(ep_rewards)
    env.close()

def main():
    # get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",
                        required=True,
                        type=str,
                        help="path to trained model dir",
    )
    args = parser.parse_args()

    path_to_actor = ""
    path_to_pkl = ""
    if os.path.isfile(args.path) and args.path.endswith(".pt"):
        path_to_actor = args.path
        path_to_pkl = os.path.join(os.path.dirname(args.path), "experiment.pkl")
    if os.path.isdir(args.path):
        path_to_actor = os.path.join(args.path, "actor.pt")
        path_to_pkl = os.path.join(args.path, "experiment.pkl")

    # load experiment args
    run_args = pickle.load(open(path_to_pkl, "rb"))
    # load trained policy
    policy = torch.load(path_to_actor)
    policy.eval()
    # import the correct environment
    env = import_env(run_args.env)()

    run(env, policy)
    print("-----------------------------------------")

if __name__=='__main__':
    main()
