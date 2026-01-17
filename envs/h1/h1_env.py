import os

import numpy as np

from tasks.standing_task import StandingTask

from .gen_xml import ARM_JOINTS, WAIST_JOINTS, builder
from .h1_base import H1BaseEnv


class H1Env(H1BaseEnv):
    """Unitree H1 humanoid standing environment."""

    def _build_xml(self) -> str:
        export_dir = self._get_xml_export_dir("h1")
        path_to_xml = os.path.join(export_dir, "h1.xml")
        if not os.path.exists(path_to_xml):
            builder(
                export_dir,
                config={
                    "unused_joints": [WAIST_JOINTS, ARM_JOINTS],
                    "rangefinder": False,
                    "raisedplatform": False,
                    "ctrllimited": self.cfg.ctrllimited,
                    "jointlimited": self.cfg.jointlimited,
                    "minimal": self.cfg.reduced_xml,
                },
            )
        return path_to_xml

    def _setup_task(self, control_dt: float) -> None:
        self.task = StandingTask(self.interface, self.half_sitting_pose)

    def _get_external_state(self) -> np.ndarray:
        return np.array([])

    def _setup_obs_normalization(self) -> None:
        self.obs_mean = np.concatenate(
            (
                np.zeros(5),
                self.half_sitting_pose,
                np.zeros(10),
                np.zeros(10),
            )
        )
        self.obs_std = np.concatenate(
            (
                [0.2, 0.2, 1, 1, 1],
                0.5 * np.ones(10),
                4 * np.ones(10),
                100 * np.ones(10),
            )
        )
        self.obs_mean = np.tile(self.obs_mean, self.history_len)
        self.obs_std = np.tile(self.obs_std, self.history_len)
