import os
import numpy as np
import mujoco
import mujoco_viewer

DEFAULT_SIZE = 500

class MujocoEnv():
    """Superclass for all MuJoCo environments.
    """

    def __init__(self, model_path, sim_dt, control_dt):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            raise Exception("Provide full path to robot description package.")
        if not os.path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.model = mujoco.MjModel.from_xml_path(fullpath)
        self.data = mujoco.MjData(self.model)
        self.viewer = None

        # set frame skip and sim dt
        self.frame_skip = (control_dt/sim_dt)
        self.model.opt.timestep = sim_dt

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.5
        self.viewer.cam.lookat[2] = 1.5
        self.viewer.cam.lookat[0] = 2.0
        self.viewer.cam.elevation = -20
        self.viewer.vopt.geomgroup[0] = 1
        self.viewer._render_every_frame = True

    def viewer_is_paused(self):
        return self.viewer._paused

    # -----------------------------

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        ob = self.reset_model()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
            self.viewer_setup()
        self.viewer.render()

    def uploadGPU(self, hfieldid=None, meshid=None, texid=None):
        # hfield
        if hfieldid is not None:
            mujoco.mjr_uploadHField(self.model, self.viewer.ctx, hfieldid)
        # mesh
        if meshid is not None:
            mujoco.mjr_uploadMesh(self.model, self.viewer.ctx, meshid)
        # texture
        if texid is not None:
            mujoco.mjr_uploadTexture(self.model, self.viewer.ctx, texid)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
