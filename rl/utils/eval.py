import time
from datetime import datetime
from pathlib import Path

import imageio
import mujoco
import torch


class EvaluateEnv:
    def __init__(self, env, policy, args):
        self.env = env
        # Move policy to CPU for evaluation
        self.policy = policy.cpu()
        if hasattr(self.policy, "obs_mean") and torch.is_tensor(self.policy.obs_mean):
            self.policy.obs_mean = self.policy.obs_mean.cpu()
        if hasattr(self.policy, "obs_std") and torch.is_tensor(self.policy.obs_std):
            self.policy.obs_std = self.policy.obs_std.cpu()
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
            obs_tensor = torch.tensor(observation, dtype=torch.float32)
            raw = self.policy.forward(obs_tensor, deterministic=True).numpy()
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
                0, self.env.frame_skip * self.env.model.opt.timestep - (time.time() - step_start)
            )
            time.sleep(time_until_next_step)

        for frame in frames:
            self.writer.append_data(frame)
        self.writer.close()
        self.env.close()
