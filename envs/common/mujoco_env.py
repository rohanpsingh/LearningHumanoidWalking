import contextlib
import os

import mujoco
import mujoco.viewer
import numpy as np

from envs.common.marker_drawer import MarkerDrawer

DEFAULT_SIZE = 500


class MujocoEnv:
    """Superclass for all MuJoCo environments."""

    def __init__(self, model_path, sim_dt, control_dt):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            raise Exception("Provide full path to robot description package.")
        if not os.path.exists(fullpath):
            raise OSError(f"File {fullpath} does not exist")

        self.spec = mujoco.MjSpec.from_file(fullpath)
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self._viewer_paused = False
        self._marker_drawer = None

        # set frame skip and sim dt
        self.frame_skip = control_dt / sim_dt
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

    def _key_callback(self, keycode):
        """Handle keyboard events from the viewer."""
        # Space key (ASCII/GLFW code 32)
        if keycode == 32:
            self._viewer_paused = not self._viewer_paused

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        with self.viewer.lock():
            self.viewer.cam.trackbodyid = 1
            self.viewer.cam.distance = self.model.stat.extent * 1.5
            self.viewer.cam.lookat[2] = 1.5
            self.viewer.cam.lookat[0] = 2.0
            self.viewer.cam.elevation = -20
            self.viewer.opt.geomgroup[2] = 0

    def viewer_is_paused(self):
        return self._viewer_paused

    # -----------------------------
    # (some methods are taken directly from dm_control)

    @contextlib.contextmanager
    def disable(self, *flags):
        """Context manager for temporarily disabling MuJoCo flags.

        Args:
          *flags: Positional arguments specifying flags to disable. Can be either
            lowercase strings (e.g. 'gravity', 'contact') or `mjtDisableBit` enum
            values.

        Yields:
          None

        Raises:
          ValueError: If any item in `flags` is neither a valid name nor a value
            from `mujoco.mjtDisableBit`.
        """
        old_bitmask = self.model.opt.disableflags
        new_bitmask = old_bitmask
        for flag in flags:
            if isinstance(flag, str):
                try:
                    field_name = "mjDSBL_" + flag.upper()
                    flag = getattr(mujoco.mjtDisableBit, field_name)
                except AttributeError:
                    valid_names = [
                        field_name.split("_")[1].lower() for field_name in list(mujoco.mjtDisableBit.__members__)[:-1]
                    ]
                    raise ValueError(
                        "'{}' is not a valid flag name. Valid names: {}".format(flag, ", ".join(valid_names))
                    ) from None
            elif isinstance(flag, int):
                flag = mujoco.mjtDisableBit(flag)
            new_bitmask |= flag.value
        self.model.opt.disableflags = new_bitmask
        try:
            yield
        finally:
            self.model.opt.disableflags = old_bitmask

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        ob = self.reset_model()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,), f"qpos shape {qpos.shape} is expected to be {(self.model.nq,)}"
        assert qvel.shape == (self.model.nv,), f"qvel shape {qvel.shape} is expected to be {(self.model.nv,)}"
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.act = []
        self.data.plugin_state = []
        # Disable actuation since we don't yet have meaningful control inputs.
        with self.disable("actuation"):
            mujoco.mj_forward(self.model, self.data)

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def draw_markers(self, marker_drawer):
        """Draw task-specific markers in the viewer.

        Override this method in subclasses to draw custom visualizations
        (e.g., step targets, goal positions, debug info).

        Args:
            marker_drawer: MarkerDrawer instance for adding geometries
        """
        pass

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=self._key_callback)
            self._marker_drawer = MarkerDrawer(self.viewer)
            self.viewer_setup()

        # Draw markers if we have a marker drawer
        if self._marker_drawer is not None:
            self._marker_drawer.reset()
            self.draw_markers(self._marker_drawer)
            self._marker_drawer.finalize()

        # Block while paused, but keep viewer responsive
        while self._viewer_paused and self.viewer.is_running():
            self.viewer.sync()
        self.viewer.sync()

    def uploadGPU(self, hfieldid=None, meshid=None, texid=None):
        raise NotImplementedError(
            "uploadGPU is not supported with mujoco.viewer.launch_passive. "
            "GPU uploads happen automatically when modifying model data."
        )

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    # Contact diagnostics for MuJoCo 3.4.0 upgrade verification
    def check_contact_truncation(self):
        """Check if contacts are being truncated due to nconmax limit.

        Returns:
            tuple: (is_truncated, ncon, nconmax)
        """
        ncon = self.data.ncon
        nconmax = self.model.nconmax
        # In MuJoCo 3.x, -1 means auto-sized (no explicit limit)
        is_truncated = nconmax > 0 and ncon >= nconmax
        return (is_truncated, ncon, nconmax)

    def print_contact_diagnostics(self):
        """Print contact diagnostics for debugging collision issues."""
        is_truncated, ncon, nconmax = self.check_contact_truncation()
        print(f"Contacts: {ncon}/{nconmax}" + (" [TRUNCATED!]" if is_truncated else ""))
        print(f"Constraints (njmax): {self.data.nefc}/{self.model.njmax}")

        for i in range(ncon):
            con = self.data.contact[i]
            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or f"geom{con.geom1}"
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or f"geom{con.geom2}"
            print(f"  Contact {i}: {geom1_name} <-> {geom2_name}, dist={con.dist:.6f}")

    def get_contact_summary(self):
        """Get summary of current contacts for verification.

        Returns:
            dict: Contact statistics and details including forces
        """
        is_truncated, ncon, nconmax = self.check_contact_truncation()
        contacts = []
        for i in range(ncon):
            con = self.data.contact[i]
            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1)
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2)

            # Get contact force
            c_array = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, c_array)
            force = np.linalg.norm(c_array)

            contacts.append(
                {
                    "geom1": geom1_name or f"geom{con.geom1}",
                    "geom2": geom2_name or f"geom{con.geom2}",
                    "dist": con.dist,
                    "force": force,
                    "pos": con.pos.copy(),
                }
            )
        return {
            "ncon": ncon,
            "nconmax": nconmax,
            "truncated": is_truncated,
            "nefc": self.data.nefc,
            "njmax": self.model.njmax,
            "contacts": contacts,
        }
