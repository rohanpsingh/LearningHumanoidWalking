"""Mocap-imitation task for the H1 humanoid.

Tracks reference root pose, joint positions/velocities, body poses, and
contact forces against a preloaded mocap clip. All joint names used here
are XML-style (no `_joint` suffix); the reference loader rewrites mocap
keys accordingly.
"""

from __future__ import annotations

import numpy as np
import transforms3d as tf3
from scipy.spatial.transform import Rotation

from tasks.base_task import BaseTask

LEG_JOINTS = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]
WAIST_JOINTS = ["torso"]
ARM_JOINTS = [
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
]


class MimicTask(BaseTask):
    """Reference-tracking task for an H1 carrying a box."""

    def __init__(
        self,
        client=None,
        dt: float = 0.025,
        root_body: str = "pelvis",
        rhand_body: str = "right_elbow_link",
        lhand_body: str = "left_elbow_link",
        rfoot_body: str = "right_ankle_link",
        lfoot_body: str = "left_ankle_link",
        object_body: str = "box",
        track_joint_names: list[str] | None = None,
        track_body_names: list[str] | None = None,
        reference: dict | None = None,
    ):
        self._client = client
        self._control_dt = dt

        self.reference_motion = reference or {}

        self.root_body_name = root_body
        self.rhand_body_name = rhand_body
        self.lhand_body_name = lhand_body
        self.rfoot_body_name = rfoot_body
        self.lfoot_body_name = lfoot_body
        self.object_body_name = object_body

        self.track_joint_names = list(track_joint_names) if track_joint_names else []
        self.track_body_names = list(track_body_names) if track_body_names else []

        # Phase / period are set in reset()
        self.mimic_phase = 0
        self.gait_phase = 0
        self.mimic_period = 1
        self.gait_period = 1

        # Tracking errors (filled in calc_reward; consumed by done())
        self.root_track_error = 0.0
        self.body_track_error = 0.0
        self.object_track_error = 0.0
        self.joint_track_error = 0.0
        self.hand_track_error = 0.0
        self.force_track_error = 0.0

    def step(self) -> None:
        # Phase advancement is handled at the env level so the env can detect
        # clip-end and trigger a reset; nothing to do here.
        return

    def calc_reward(self, prev_torque, prev_action, action) -> dict[str, float]:
        idx = self._current_idx()
        dt = self.reference_motion["dt"]
        ref_xy_vel = np.linalg.norm(
            (self.reference_motion["root_pose"][idx] - self.reference_motion["root_pose"][idx - 1])[:2] / dt
        )
        mimic_flag = ref_xy_vel < 0.15

        if mimic_flag:
            self.track_joint_names = LEG_JOINTS + WAIST_JOINTS + ARM_JOINTS
            self.track_body_names = [
                "torso_link",
                "left_elbow_link",
                "right_elbow_link",
                "left_ankle_link",
                "right_ankle_link",
            ]
        else:
            self.track_joint_names = WAIST_JOINTS + ARM_JOINTS
            self.track_body_names = ["torso_link", "left_elbow_link", "right_elbow_link"]

        # Current motor positions/velocities
        motor_pos = [self._client.get_act_joint_position(jn + "_motor") for jn in self.track_joint_names]
        motor_vel = [self._client.get_act_joint_velocity(jn + "_motor") for jn in self.track_joint_names]

        # Current relative body poses (in root frame)
        root_pose = self._client.get_object_affine_by_name(self.root_body_name, "OBJ_BODY")
        rel_link_poses = []
        for body_name in self.track_body_names:
            body_pose = self._client.get_object_affine_by_name(body_name, "OBJ_BODY")
            rel_link_poses.append(np.linalg.inv(root_pose).dot(body_pose))

        thresh = 10.0
        lhand_frc = int(self._client.get_interaction_force(self.lhand_body_name, self.object_body_name) > thresh)
        rhand_frc = int(self._client.get_interaction_force(self.rhand_body_name, self.object_body_name) > thresh)
        box_frc = int(
            self._client.get_interaction_force(self.object_body_name, "table1")
            + self._client.get_interaction_force(self.object_body_name, "table2")
            > thresh / 2
        )

        # Reference at current phase
        ref_root_pose = self.reference_motion["root_pose"][idx][[0, 1, 2, 6, 3, 4, 5]]
        ref_root_pose = tf3.affines.compose(ref_root_pose[:3], tf3.quaternions.quat2mat(ref_root_pose[3:]), np.ones(3))
        ref_motor_pos = np.array([self.reference_motion["joint_position"][jn][idx] for jn in self.track_joint_names])
        ref_motor_vel = np.array([self.reference_motion["joint_velocity"][jn][idx] for jn in self.track_joint_names])
        ref_rel_link_poses = []
        for body_name in self.track_body_names:
            ref_link_pose = self.reference_motion["relative_link_pose"][body_name][idx][[0, 1, 2, 6, 3, 4, 5]]
            ref_rel_link_poses.append(
                tf3.affines.compose(ref_link_pose[:3], tf3.quaternions.quat2mat(ref_link_pose[3:]), np.ones(3))
            )

        ref_lhand_frc = self.reference_motion["force_lhand"][idx]
        ref_rhand_frc = self.reference_motion["force_rhand"][idx]
        ref_box_frc = self.reference_motion["force_box"][idx]

        # Joint tracking error
        motor_vel = 0.1 * np.array(motor_vel)
        ref_motor_vel = 0.1 * ref_motor_vel
        self.joint_track_error = np.linalg.norm(
            np.concatenate((ref_motor_pos, ref_motor_vel)) - np.concatenate((motor_pos, motor_vel))
        )

        # Body tracking error
        link_positions = np.array(rel_link_poses)[:, :3, 3].flatten()
        link_rotations = np.array(rel_link_poses)[:, :3, :3].flatten()
        ref_link_positions = np.array(ref_rel_link_poses)[:, :3, 3].flatten()
        ref_link_rotations = np.array(ref_rel_link_poses)[:, :3, :3].flatten()
        body_pos_track_error = np.linalg.norm(ref_link_positions - link_positions)
        body_rot_track_error = np.linalg.norm(ref_link_rotations - link_rotations)
        self.body_track_error = body_pos_track_error + 0.1 * body_rot_track_error

        # Root tracking error
        root_position = root_pose[:3, 3].flatten()
        root_rotation = root_pose[:3, :3].flatten()
        ref_root_position = ref_root_pose[:3, 3].flatten()
        ref_root_rotation = ref_root_pose[:3, :3].flatten()
        self.root_track_error = np.linalg.norm(root_position - ref_root_position) + 0.1 * np.linalg.norm(
            root_rotation - ref_root_rotation
        )

        # Object tracking error (in root frame)
        obj_pose = self._client.get_object_affine_by_name(self.object_body_name, "OBJ_BODY")
        rel_obj_pose = np.linalg.inv(root_pose).dot(obj_pose)
        ref_obj_pose_v = self.reference_motion["object_pose"][idx][[0, 1, 2, 6, 3, 4, 5]]
        ref_obj_pose = tf3.affines.compose(ref_obj_pose_v[:3], tf3.quaternions.quat2mat(ref_obj_pose_v[3:]), np.ones(3))
        ref_rel_obj_pose = np.linalg.inv(ref_root_pose).dot(ref_obj_pose)
        self.object_track_error = np.linalg.norm(rel_obj_pose[:3, 3] - ref_rel_obj_pose[:3, 3]) + 0.1 * np.linalg.norm(
            rel_obj_pose[:2, :3].flatten() - ref_rel_obj_pose[:2, :3].flatten()
        )

        # Hand tracking error (target expressed in box frame)
        target_positions = {
            "right_hand": np.array([0.2, 0, 0]),
            "left_hand": np.array([-0.2, 0, 0]),
        }
        box_p = self._client.get_object_xpos_by_name("box", "OBJ_BODY")
        box_q = self._client.get_object_xquat_by_name("box", "OBJ_BODY")
        # MuJoCo quat is [w, x, y, z]; scipy expects [x, y, z, w]
        box_rot = Rotation.from_quat([box_q[1], box_q[2], box_q[3], box_q[0]]).as_matrix()
        hands_in_box = []
        for hand in ["right_hand", "left_hand"]:
            p = self._client.get_object_xpos_by_name(hand, "OBJ_SITE")
            hand_in_box_frame = box_rot.T.dot(p - np.array(box_p))
            hands_in_box.append(np.linalg.norm(hand_in_box_frame - target_positions[hand]))

        ref_rhand_dist = self.reference_motion["distance_right_hand_to_target"][idx]
        ref_lhand_dist = self.reference_motion["distance_left_hand_to_target"][idx]
        self.hand_track_error = np.linalg.norm(np.array([ref_rhand_dist, ref_lhand_dist]) - np.array(hands_in_box))

        # Force tracking error
        self.force_track_error = np.linalg.norm(
            np.array([lhand_frc, rhand_frc, box_frc]) - np.array([ref_lhand_frc, ref_rhand_frc, ref_box_frc])
        )

        # Joint range / torque penalties
        current_pose = np.array(self._client.get_act_joint_positions())
        l_oo, u_oo = self.get_out_of_limit_joints(current_pose, reduction=10)
        joint_range_error = float(np.sum(l_oo) + np.sum(u_oo))
        tau_error = np.linalg.norm(self._client.get_act_joint_torques())

        return {
            "root_track_score": 0.2 * np.exp(-0.5 * self.root_track_error**2),
            "body_track_score": 0.15 * np.exp(-10.0 * self.body_track_error**2),
            "joint_track_score": 0.15 * np.exp(-0.5 * self.joint_track_error**2),
            "object_track_score": 0.1 * np.exp(-4.0 * self.object_track_error**2),
            "hand_track_score": 0.2 * np.exp(-10.0 * self.hand_track_error**2),
            "force_track_score": 0.1 * np.exp(-100.0 * self.force_track_error**2),
            "joint_range_score": 0.05 * np.exp(-20.0 * joint_range_error**2),
            "joint_torque_score": 0.05 * np.exp(-5e-5 * tau_error**2),
        }

    def done(self) -> bool:
        current_pose = np.array(self._client.get_act_joint_positions())
        l_oo, u_oo = self.get_out_of_limit_joints(current_pose, reduction=5)
        joint_near_limit = bool(np.any(l_oo > 0)) or bool(np.any(u_oo > 0))

        root_jnt_adr = self._client.model.body(self.root_body_name).jntadr[0]
        root_qpos_adr = self._client.model.joint(root_jnt_adr).qposadr[0]
        qpos = self._client.get_qpos()[root_qpos_adr : root_qpos_adr + 7]
        bad_contact = self._check_bad_collisions(self._client.model, self._client.data)

        terminate = {
            "qpos[2]_ll": qpos[2] < 0.6,
            "qpos[2]_ul": qpos[2] > 1.4,
            "faraway_root": self.root_track_error > 0.4,
            "faraway_bodies": self.body_track_error > 0.8,
            "bad_contact": bad_contact,
            "joint_near_limit": joint_near_limit,
        }
        return any(terminate.values())

    def reset(self, iter_count: int = 0) -> None:
        dt = self.reference_motion["dt"]
        clip_frames = len(self.reference_motion["root_pose"])
        clip_duration_s = round(clip_frames * dt, 5)
        self.mimic_period = clip_duration_s * (1 / self._control_dt)

        if iter_count < 5000:
            p = [0.2, 0.8]
        else:
            p = [0.8, 0.2]
        self.mimic_phase = int(
            np.random.choice(
                [0, np.random.randint(int(0.8 * self.mimic_period), int(self.mimic_period))],
                p=p,
            )
        )

        # Bipedal gait clock (independent of mocap clock)
        swing_duration = 0.4
        stance_duration = 0.1
        total_duration = 2 * (swing_duration + stance_duration)
        self.gait_period = float(np.floor(total_duration * (1 / self._control_dt)))
        self.gait_phase = int(np.random.randint(0, int(self.gait_period)))

    def _current_idx(self) -> int:
        n = self.reference_motion["root_pose"].shape[0]
        return int((self.mimic_phase / self.mimic_period) * n)

    def get_out_of_limit_joints(self, current_pose, reduction: int = 10):
        llimit, ulimit = self._client.get_act_joint_ranges()[0], self._client.get_act_joint_ranges()[1]
        llimit = llimit + np.deg2rad(reduction)
        ulimit = ulimit - np.deg2rad(reduction)
        out_of_llimit = -np.clip(current_pose - llimit, a_min=-np.inf, a_max=0)
        out_of_ulimit = np.clip(current_pose - ulimit, a_min=0, a_max=np.inf)
        return out_of_llimit, out_of_ulimit

    @staticmethod
    def _check_bad_collisions(model, data) -> bool:
        allowed = {
            "world",
            "right_elbow_link",
            "left_elbow_link",
            "right_ankle_link",
            "left_ankle_link",
            "box",
            "table1",
            "table2",
            "hfield",
        }
        for i in range(data.ncon):
            c = data.contact[i]
            geom1_body = model.body(model.geom_bodyid[c.geom1]).name
            geom2_body = model.body(model.geom_bodyid[c.geom2]).name
            if geom1_body not in allowed or geom2_body not in allowed:
                return True
            if geom1_body.endswith("elbow_link") and geom2_body != "box":
                return True
        return False
