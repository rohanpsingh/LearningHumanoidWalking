import torch
import time
from pathlib import Path

import mujoco

import imageio
from datetime import datetime


class EvaluateEnv:
    def __init__(self, env, policy, args):
        self.env = env
        self.policy = policy
        self.ep_len = args.ep_len

        if args.out_dir is None:
            args.out_dir = Path(args.path.parent, "videos")

        video_outdir = Path(args.out_dir)
        try:
            Path.mkdir(video_outdir, exist_ok=True)
            now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            video_fn = Path(video_outdir, args.path.stem + "-" + now + ".mp4")
            self.writer = imageio.get_writer(video_fn, fps=60)
        except Exception as e:
            print("Could not create video writer:", e)
            exit(-1)

    @torch.no_grad()
    def run(self):
        height = 480
        width = 640
        renderer = mujoco.Renderer(self.env.model, height, width)
        frames = []

        # Initialize viewer via env.render()
        observation = self.env.reset()
        self.env.render()
        viewer = self.env.viewer

        # Configure camera
        mujoco.mjv_defaultCamera(viewer.cam)
        viewer.cam.elevation = -20
        viewer.cam.distance = 4

        reset_counter = 0
        while viewer.is_running() and self.env.data.time < self.ep_len:
            step_start = time.time()

            # forward pass and step
            raw = self.policy.forward(torch.tensor(observation, dtype=torch.float32), deterministic=True).detach().numpy()
            observation, _, done, _ = self.env.step(raw.copy())

            # render scene for video recording
            viewer.cam.lookat = self.env.data.body(1).xpos.copy()
            renderer.update_scene(self.env.data, viewer.cam)
            if not self.env.viewer_is_paused():
                frames.append(renderer.render())

            # render viewer (handles pause internally)
            self.env.render()

            if done and reset_counter < 3:
                observation = self.env.reset()
                reset_counter += 1

            time_until_next_step = max(
                0, self.env.frame_skip * self.env.model.opt.timestep - (time.time() - step_start))
            time.sleep(time_until_next_step)

        for frame in frames:
            self.writer.append_data(frame)
        self.writer.close()
        self.env.close()
