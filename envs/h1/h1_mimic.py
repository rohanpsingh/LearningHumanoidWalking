"""H1 humanoid mocap-imitation environment.

Subclasses H1BaseEnv but overrides:
  * `_build_xml`      — adds movable box, mocap-vis bodies, optional hfield
  * `_setup_robot`    — actuates all 19 joints (legs + waist + arms) and adds box DOFs
  * `_setup_spaces`   — 19-D action, observation = robot state (62) + clocks (4) + box rel pose (9)
  * `step`/`reset_model` — phase-driven episode + reference state initialization
  * `render`/`viewer_setup` — mocap-body visualization of the reference clip
"""

from __future__ import annotations

import collections
import contextlib
import os

import numpy as np
import torch
import transforms3d as tf3

from envs.common import robot_interface
from envs.common.utils import get_project_root
from rl.algos.imitation import ImitationQuery
from robots.robot_base import RobotBase
from tasks.mimic_task import MimicTask

from .gen_xml import ARM_JOINTS, LEG_JOINTS, WAIST_JOINTS, builder
from .h1_base import H1BaseEnv
from .mimic_reference import Mode, load_reference


class H1MimicEnv(H1BaseEnv):
    """H1 humanoid imitating a mocap clip while interacting with a box."""

    # Bodies the env visualizes against the reference clip.
    _MOCAP_BODIES = [
        "box",
        "pelvis",
        "torso_link",
        "left_elbow_link",
        "right_elbow_link",
        "left_knee_link",
        "left_ankle_link",
        "right_knee_link",
        "right_ankle_link",
    ]

    def _get_default_config_path(self) -> str:
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs/mimic.yaml")

    def _build_xml(self) -> str:
        export_dir = self._get_xml_export_dir("h1_mimic")
        path_to_xml = os.path.join(export_dir, "h1.xml")
        if not os.path.exists(path_to_xml):
            builder(
                export_dir,
                config={
                    "unused_joints": [],
                    "rangefinder": False,
                    "raisedplatform": False,
                    "ctrllimited": self.cfg.ctrllimited,
                    "jointlimited": self.cfg.jointlimited,
                    "minimal": self.cfg.reduced_xml,
                    "movable_box": "box",
                    "mocap_bodies": self._MOCAP_BODIES,
                    "hand_sites": True,
                    "hfield": getattr(self.cfg, "uneven_terrain", None) is not None and self.cfg.uneven_terrain.enable,
                },
            )
        return path_to_xml

    def _setup_robot(self) -> None:
        control_dt = self.cfg.control_dt

        # Actual robot is ~7kg heavier
        self.model.body("pelvis").mass = 8.89
        self.model.body("torso_link").mass = 21.289

        # Actuate all 19 joints
        self.leg_names = LEG_JOINTS
        self.waist_names = WAIST_JOINTS
        self.arm_names = ARM_JOINTS
        self.actuators = LEG_JOINTS + WAIST_JOINTS + ARM_JOINTS

        gains_dict = self.cfg.pdgains.to_dict()
        kp, kd = zip(*[gains_dict[jn] for jn in self.actuators], strict=True)
        pdgains = np.array([kp, kd])

        # Half-sitting pose (19 entries)
        self.half_sitting_pose = list(self.cfg.half_sitting_pose)
        assert len(self.half_sitting_pose) == len(self.actuators), (
            f"half_sitting_pose has {len(self.half_sitting_pose)} entries but "
            f"{len(self.actuators)} actuated joints are configured"
        )

        # Nominal pose: root (7) + joints (19) + box (7)
        base_position = [0, 0, 0.98]
        base_orientation = [1, 0, 0, 0]
        box_position = [1.5, -0.8, 0.81]
        box_orientation = list(tf3.euler.euler2quat(0, 0, 1.57))
        self.nominal_pose = base_position + base_orientation + self.half_sitting_pose + box_position + box_orientation

        self.interface = robot_interface.RobotInterface(self.model, self.data, self.RFOOT_BODY, self.LFOOT_BODY, None)

        # Load mocap clip
        cfg_mimic = self.cfg.mimic
        motion_pkl = get_project_root() / cfg_mimic.motion_pkl
        object_csv = get_project_root() / cfg_mimic.object_csv
        mode = Mode[cfg_mimic.mode]
        self.reference = load_reference(motion_pkl, object_csv, mode)

        # Setup task
        self.task = MimicTask(
            client=self.interface,
            dt=control_dt,
            root_body="pelvis",
            rhand_body="right_elbow_link",
            lhand_body="left_elbow_link",
            rfoot_body=self.RFOOT_BODY,
            lfoot_body=self.LFOOT_BODY,
            object_body="box",
            track_joint_names=self.actuators,
            track_body_names=[
                "torso_link",
                "left_elbow_link",
                "right_elbow_link",
                "left_ankle_link",
                "right_ankle_link",
            ],
            reference=self.reference,
        )
        # Initial periods so calc_reward() is callable before reset
        self.task.reset(iter_count=0)

        self.robot = RobotBase(pdgains, control_dt, self.interface, self.task)

    def _setup_spaces(self) -> None:
        action_space_size = len(self.actuators)
        self.action_space = np.zeros(action_space_size)
        self.prev_prediction = np.zeros(action_space_size)

        self.base_obs_len = self._get_robot_state_len() + self._get_num_external_obs()
        self.observation_space = np.zeros(self.base_obs_len * self.history_len)

        self._setup_obs_normalization()

    def _get_robot_state_len(self) -> int:
        # root_r(1) + root_p(1) + root_ang_vel(3) + motor_pos(19) + motor_vel(19) + motor_tau(19)
        return 1 + 1 + 3 + 3 * len(self.actuators)

    def _get_num_external_obs(self) -> int:
        # mimic clock (2) + gait clock (2) + rel obj pos (3) + rel obj rot top-2-rows (6)
        return 2 + 2 + 3 + 6

    def _setup_obs_normalization(self) -> None:
        n_jnt = len(self.actuators)
        obs_mean = np.concatenate(
            (
                np.zeros(5),
                np.array(self.half_sitting_pose),
                np.zeros(n_jnt),
                np.zeros(n_jnt),
                np.zeros(2),
                np.zeros(2),
                np.zeros(9),
            )
        )
        obs_std = np.concatenate(
            (
                np.array([0.2, 0.2, 1, 1, 1]),
                0.5 * np.ones(n_jnt),
                4 * np.ones(n_jnt),
                100 * np.ones(n_jnt),
                np.ones(2),
                np.ones(2),
                np.ones(9),
            )
        )
        self.obs_mean = np.tile(obs_mean, self.history_len)
        self.obs_std = np.tile(obs_std, self.history_len)

    def _get_external_state(self) -> np.ndarray:
        # Clocks
        phi_mimic = 2 * np.pi * self.task.mimic_phase / self.task.mimic_period
        phi_gait = 2 * np.pi * self.task.gait_phase / self.task.gait_period
        clocks = np.array([np.sin(phi_mimic), np.cos(phi_mimic), np.sin(phi_gait), np.cos(phi_gait)])

        # Relative pose of the observation target (box for pickup, table2 otherwise)
        cfg_mimic = self.cfg.mimic
        obs_obj_name = "box" if cfg_mimic.mode == "BOX_PICKUP" else "table2"
        root_pose = self.interface.get_object_affine_by_name("pelvis", "OBJ_BODY")
        obj_pose = self.interface.get_object_affine_by_name(obs_obj_name, "OBJ_SITE")
        rel = np.linalg.inv(root_pose).dot(obj_pose)
        rel_obj_pos = rel[:3, 3].flatten()
        rel_obj_rot = rel[:2, :3].flatten()

        ext = np.concatenate([clocks, rel_obj_pos, rel_obj_rot])

        # Optional observation noise on relative object pose
        if hasattr(self.cfg, "observation_noise") and self.cfg.observation_noise.enabled:
            level = self.cfg.observation_noise.multiplier
            noise_type = self.cfg.observation_noise.type
            noise = (
                (lambda x, n: np.random.uniform(-x, x, n))
                if noise_type == "uniform"
                else (lambda x, n: np.random.randn(n) * x)
            )
            ext[4:7] += noise(0.01 * level, 3)
            ext[7:13] += noise(0.001 * level, 6)
        return ext

    def step(self, action: np.ndarray):
        targets = self._action_smoothing * action + (1 - self._action_smoothing) * self.prev_prediction
        offsets = np.asarray(self._get_action_offsets())

        rewards, done = self.robot.step(targets, offsets)

        self.task.mimic_phase += 1
        self.task.gait_phase += 1
        if self.task.mimic_phase >= self.task.mimic_period:
            obs = self.reset_model(phase=0)
            return obs, sum(rewards.values()), done, rewards
        if self.task.gait_phase >= self.task.gait_period:
            self.task.gait_phase = 0

        obs = self.get_obs()
        self.prev_prediction = action

        # Domain randomization (matching the BaseHumanoidEnv schedule)
        if self.dynrand_interval > 0 and np.random.randint(self.dynrand_interval) == 0:
            self._randomize_dynamics()
        if self.perturb_interval > 0 and np.random.randint(self.perturb_interval) == 0:
            self._apply_perturbation()

        return obs, sum(rewards.values()), done, rewards

    def reset_model(self, phase: int | None = None) -> np.ndarray:
        # Heightfield height jitter (only applied if hfield exists)
        if getattr(self.cfg, "uneven_terrain", None) is not None and self.cfg.uneven_terrain.enable:
            with contextlib.suppress(KeyError):
                self.model.body("hfield").pos[2] = np.random.uniform(-0.05, -0.08)

        self.task.reset(iter_count=self.robot.iteration_count)
        if phase is not None:
            self.task.mimic_phase = phase

        idx = self._reference_idx()
        self._put_robot_in_scene(idx)

        self.prev_prediction = np.zeros_like(self.prev_prediction)
        self.observation_history = collections.deque(maxlen=self.history_len)
        return self.get_obs()

    def _reference_idx(self) -> int:
        n = self.reference["root_pose"].shape[0]
        return int((self.task.mimic_phase / self.task.mimic_period) * n)

    def _put_robot_in_scene(self, idx: int) -> None:
        init_qpos = np.array(self.nominal_pose, dtype=float)
        init_qvel = np.zeros(self.interface.nv())

        # Robot joints from reference clip
        for jn in self.actuators:
            qp = self.interface.get_jnt_qposadr_by_name(jn)[0]
            qv = self.interface.get_jnt_qveladr_by_name(jn)[0]
            init_qpos[qp] = self.reference["joint_position"][jn][idx]
            init_qvel[qv] = self.reference["joint_velocity"][jn][idx]

        # Robot root from reference (mocap quat is [x, y, z, w]; mujoco wants [w, x, y, z])
        ref_root_pose = self.reference["root_pose"][idx]
        root_jnt_adr = self.model.body("pelvis").jntadr[0]
        qpos_adr = self.model.joint(root_jnt_adr).qposadr[0]
        init_qpos[qpos_adr : qpos_adr + 7] = ref_root_pose[[0, 1, 2, 6, 3, 4, 5]]

        # Object root from reference
        ref_obj_pose = self.reference["object_pose"][idx]
        box_jnt_adr = self.model.body("box").jntadr[0]
        box_qpos_adr = self.model.joint(box_jnt_adr).qposadr[0]
        init_qpos[box_qpos_adr : box_qpos_adr + 7] = ref_obj_pose[[0, 1, 2, 6, 3, 4, 5]]

        self.set_state(init_qpos, init_qvel)
        for _ in range(3):
            self.interface.step()

    def render(self):
        if self.viewer is None:
            super().render()
            self.viewer.opt.geomgroup[2] = 1  # show collision
            self.viewer.opt.geomgroup[3] = 1  # show mocap
            return

        idx = self._reference_idx()
        ref_root = self.reference["root_pose"][idx][[0, 1, 2, 6, 3, 4, 5]]
        ref_root_mat = tf3.affines.compose(ref_root[:3], tf3.quaternions.quat2mat(ref_root[3:]), np.ones(3))
        bodies: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for b, v in self.reference["relative_link_pose"].items():
            rel_pose = v[idx][[0, 1, 2, 6, 3, 4, 5]]
            rel_pose_mat = tf3.affines.compose(rel_pose[:3], tf3.quaternions.quat2mat(rel_pose[3:]), np.ones(3))
            global_pose_mat = ref_root_mat.dot(rel_pose_mat)
            bodies[b] = (global_pose_mat[:3, 3], tf3.quaternions.mat2quat(global_pose_mat[:3, :3]))
        bodies["pelvis"] = (ref_root[:3], ref_root[3:])
        obj_root = self.reference["object_pose"][idx][[0, 1, 2, 6, 3, 4, 5]]
        bodies["box"] = (obj_root[:3], obj_root[3:])

        for i in range(self.model.nbody):
            body = self.model.body(i)
            mocapid = body.mocapid[0] if hasattr(body.mocapid, "__len__") else body.mocapid
            if mocapid != -1:
                name = body.name[len("mocap_") :]
                if name in bodies:
                    p, q = bodies[name]
                    self.data.mocap_pos[mocapid] = p
                    self.data.mocap_quat[mocapid] = q
        super().render()

    def viewer_setup(self):
        super().viewer_setup()
        self.viewer.cam.distance = 8
        self.viewer.cam.lookat[2] = 1.5
        self.viewer.cam.lookat[0] = 1.0

    def imitation_projector(self):
        """Build the leg-imitation projector for the H1 walking expert.

        Slices the 19-joint mimic obs down to the 10-leg walker layout,
        appends a synthetic walker command (INPLACE + zero velocity ref),
        and masks samples to those whose gait phase falls within one full
        leg swing+stance window.
        """
        n_legs = len(self.leg_names)
        n_act = len(self.actuators)

        # Per-timestep mimic obs layout (must match _get_robot_state +
        # _get_external_state above):
        #   [0..5)   root_r, root_p, root_ang_vel(3)
        #   [5..5+n_act)            motor_pos
        #   [5+n_act..5+2*n_act)    motor_vel
        #   [5+2*n_act..5+3*n_act)  motor_tau
        #   [robot_state_len..)     mimic_clock(2), gait_clock(2), obj_pos(3), obj_rot(6)
        robot_state_len = self._get_robot_state_len()
        motor_pos_start = 5
        motor_vel_start = motor_pos_start + n_act
        motor_tau_start = motor_pos_start + 2 * n_act
        gait_clock_start = robot_state_len + 2  # past mimic_clock

        # Indices into one mimic timestep that map to one walker timestep.
        # Walker order: root(5), legs_pos(10), legs_vel(10), legs_tau(10), gait_clock(2).
        per_ts_indices = (
            list(range(0, 5))
            + list(range(motor_pos_start, motor_pos_start + n_legs))
            + list(range(motor_vel_start, motor_vel_start + n_legs))
            + list(range(motor_tau_start, motor_tau_start + n_legs))
            + list(range(gait_clock_start, gait_clock_start + 2))
        )

        # INPLACE mode + zero velocity reference: tells the walker to "walk in
        # place" for the leg-imitation signal during the box-manip task.
        inplace_command = np.concatenate([np.array([0.0, 1.0, 0.0]), np.zeros(3)])

        # Imitation phase window: one full leg swing + half-stance buffer on
        # either side, derived from the task's gait timing.
        s = self.task._stance_duration
        w = self.task._swing_duration
        t = self.task._total_duration
        phi_lo = (s / 2) / t
        phi_hi = (s / 2 + w + s / 2) / t

        return _MimicLegImitationProjector(
            per_ts_indices=per_ts_indices,
            per_ts_mimic_dim=self.base_obs_len,
            history_len=self.history_len,
            gait_clock_offset=gait_clock_start,
            synthetic_command=inplace_command,
            action_indices=np.arange(n_legs, dtype=np.int64),
            phi_window=(phi_lo, phi_hi),
        )


class _MimicLegImitationProjector:
    """Projects mimic obs into the walker-expert obs space.

    The projector is layout-stateless: it captures all index/shape constants
    at construction and operates purely on the obs batch tensor. Both envs
    share the same per-step gait clock layout (sin/cos of `2*pi*phi`), so we
    read the current gait phase out of the most-recent history chunk
    (chunk 0 — `appendleft` puts the latest state at the front).
    """

    def __init__(
        self,
        per_ts_indices: list[int],
        per_ts_mimic_dim: int,
        history_len: int,
        gait_clock_offset: int,
        synthetic_command: np.ndarray,
        action_indices: np.ndarray,
        phi_window: tuple[float, float],
    ):
        self._per_ts_mimic_dim = per_ts_mimic_dim
        self._history_len = history_len
        self._phi_lo, self._phi_hi = phi_window

        # Absolute index of the gait_clock sin/cos in the latest chunk.
        self._gait_sin_idx = gait_clock_offset
        self._gait_cos_idx = gait_clock_offset + 1

        # Pre-build the projection index list for every chunk in the history.
        per_ts = torch.tensor(per_ts_indices, dtype=torch.long)
        offsets = torch.arange(history_len, dtype=torch.long) * per_ts_mimic_dim
        self._select_idx = (offsets[:, None] + per_ts[None, :]).reshape(-1)

        cmd = torch.tensor(synthetic_command, dtype=torch.float32)
        self._synthetic_command = cmd.repeat(history_len)

        self._action_indices = torch.tensor(action_indices, dtype=torch.long)

    def __call__(self, obs_batch: torch.Tensor) -> ImitationQuery:
        device = obs_batch.device

        # Mask: gait phase of the most recent timestep within the imitation window.
        sin_phi = obs_batch[:, self._gait_sin_idx]
        cos_phi = obs_batch[:, self._gait_cos_idx]
        phi = torch.atan2(sin_phi, cos_phi) / (2 * torch.pi)
        phi = phi % 1.0
        sample_mask = (phi >= self._phi_lo) & (phi <= self._phi_hi)

        select_idx = self._select_idx.to(device)
        active = obs_batch[sample_mask]
        sliced = active.index_select(dim=1, index=select_idx)
        # Interleave per-timestep walker chunks with appended command per chunk.
        n_active = sliced.shape[0]
        per_ts_walker_minus_cmd = sliced.shape[1] // self._history_len
        cmd_per_ts = self._synthetic_command.shape[0] // self._history_len
        sliced = sliced.reshape(n_active, self._history_len, per_ts_walker_minus_cmd)
        cmd_chunks = self._synthetic_command.to(device).reshape(self._history_len, cmd_per_ts)
        cmd_chunks = cmd_chunks.unsqueeze(0).expand(n_active, -1, -1)
        expert_obs = torch.cat([sliced, cmd_chunks], dim=2).reshape(n_active, -1)

        return ImitationQuery(
            expert_obs=expert_obs,
            sample_mask=sample_mask,
            action_indices=self._action_indices.to(device),
        )
