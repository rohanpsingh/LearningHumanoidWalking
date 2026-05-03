"""Reference motion loader for H1 mimic environment.

Loads a pickled mocap clip plus an object-motion CSV and applies the same set
of fixes used by the internal mocap rig (manual fix-ups for bad mocap frames,
synthesized return-to-origin segment, derived hand poses, and force schedule).
"""

from __future__ import annotations

import pickle
from enum import Enum, auto
from pathlib import Path

import numpy as np
import transforms3d as tf3
from scipy.spatial.transform import Rotation, Slerp

from envs.common.utils import parse_mocap_csv

# Joint name suffix used by the mocap dictionary; not present in the menagerie XML.
_MOCAP_JOINT_SUFFIX = "_joint"


class Mode(Enum):
    BOX_PICKUP = auto()
    BOX_DROPOFF = auto()
    RETURN_TO_ORIGIN = auto()

    def get_bounds(self):
        if self.name == "BOX_PICKUP":
            return [1000, 2400]
        if self.name == "BOX_DROPOFF":
            return [2400, 3700]
        if self.name == "RETURN_TO_ORIGIN":
            return [3700, 5500]
        raise ValueError(self.name)


def _strip_joint_suffix(d: dict) -> dict:
    """Rewrite mocap dict keys to drop the `_joint` suffix to match XML names."""
    return {(k[: -len(_MOCAP_JOINT_SUFFIX)] if k.endswith(_MOCAP_JOINT_SUFFIX) else k): v for k, v in d.items()}


def load_reference(motion_pkl: str | Path, object_csv: str | Path, mode: Mode) -> dict:
    """Load and preprocess a mocap clip + object motion CSV.

    Returns a dict keyed by:
        - "root_pose"        : (N, 7) [x, y, z, qx, qy, qz, qw]
        - "joint_position"   : dict[xml_joint_name, (N,)]
        - "joint_velocity"   : dict[xml_joint_name, (N,)]
        - "relative_link_pose": dict[body_name, (N, 7)]
        - "object_pose"      : (N, 7) [x, y, z, qx, qy, qz, qw]
        - "force_lhand", "force_rhand", "force_box", "force_lfoot", "force_rfoot": (N,)
        - "distance_left_hand_to_target", "distance_right_hand_to_target": (N,)
        - "dt": float
    """
    with open(motion_pkl, "rb") as f:
        pkl_data = pickle.load(f)

    # Manually fix arm/return frames (caused by bad mocap data)
    jnts = [
        "_shoulder_pitch_joint",
        "_shoulder_roll_joint",
        "_shoulder_yaw_joint",
        "_elbow_joint",
    ]
    for i, j in zip(range(3300, 3700), range(1950, 2350), strict=False):
        q1 = pkl_data["root_pose"][i][3:]
        q2 = pkl_data["root_pose"][j][3:]
        new_q = Rotation.from_euler(
            "xyz",
            [
                *Rotation.from_quat(q2).as_euler("xyz")[:2],
                Rotation.from_quat(q1).as_euler("xyz")[2],
            ],
        ).as_quat()
        pkl_data["root_pose"][i][3:] = new_q
        pkl_data["root_pose"][i][:2] = pkl_data["root_pose"][3300][:2]
        pkl_data["root_pose"][i][2] = pkl_data["root_pose"][j][2]
        for attr in ["joint_position", "joint_velocity", "relative_link_pose"]:
            for key in pkl_data[attr]:
                if attr == "relative_link_pose" and key.endswith("_elbow_link"):
                    continue
                if attr == "joint_position":
                    if key in ["right" + jn for jn in jnts]:
                        continue
                    if key in ["left" + jn for jn in jnts]:
                        continue
                pkl_data[attr][key][i] = pkl_data[attr][key][j]

    # Mirror left elbow pose into right elbow for the same range
    for i in range(3300, 3700):
        left_pose = pkl_data["relative_link_pose"]["left_elbow_link"][i]
        p, q = left_pose[:3], left_pose[3:]
        pkl_data["relative_link_pose"]["right_elbow_link"][i][0] = p[0]
        pkl_data["relative_link_pose"]["right_elbow_link"][i][1] = -p[1]
        pkl_data["relative_link_pose"]["right_elbow_link"][i][2] = p[2]
        rot = Rotation.from_quat(q)
        R_reflect = np.diag([1, -1, 1])
        R_mirrored = R_reflect @ rot.as_matrix() @ R_reflect
        pkl_data["relative_link_pose"]["right_elbow_link"][i][3:] = Rotation.from_matrix(R_mirrored).as_quat()
        for jn in jnts:
            pkl_data["joint_position"]["right" + jn][i] = pkl_data["joint_position"]["left" + jn][i]
            pkl_data["joint_velocity"]["right" + jn][i] = pkl_data["joint_velocity"]["left" + jn][i]
            if jn in ("_shoulder_roll_joint", "_shoulder_yaw_joint"):
                pkl_data["joint_position"]["right" + jn][i] = -pkl_data["joint_position"]["left" + jn][i]
                pkl_data["joint_velocity"]["right" + jn][i] = -pkl_data["joint_velocity"]["left" + jn][i]

    # Synthesize a 'return to origin' segment from frame 3700 to 5500
    start_idx, end_idx = 3700, 5000
    s = pkl_data["root_pose"][start_idx, :2]
    e = pkl_data["root_pose"][0, :2]
    pkl_data["root_pose"] = pkl_data["root_pose"][:start_idx]
    slerp = Slerp(
        [0, 1],
        Rotation.from_quat(
            [
                pkl_data["root_pose"][start_idx - 1, 3:],
                pkl_data["root_pose"][1000, 3:],
            ]
        ),
    )
    for i in range(start_idx, end_idx):
        root_xy = s + (e - s) * (i - start_idx) / (end_idx - start_idx)
        root_z = [1.05]
        root_q = pkl_data["root_pose"][start_idx - 1, 3:]
        if i > 4700:
            j = (i - 4700) / (end_idx - 4700)
            root_q = slerp(j).as_quat()
        v = np.concatenate((root_xy, root_z, root_q))
        pkl_data["root_pose"] = np.vstack([pkl_data["root_pose"], v])
    pkl_data["root_pose"] = np.vstack([pkl_data["root_pose"], np.tile(pkl_data["root_pose"][-1], (500, 1))])
    ref_idx = 1000
    for f in ["joint_position", "joint_velocity", "relative_link_pose"]:
        for key in pkl_data[f]:
            pkl_data[f][key] = pkl_data[f][key][:start_idx]
            len_diff = len(pkl_data["root_pose"]) - len(pkl_data[f][key])
            if f == "relative_link_pose":
                reps = np.tile(pkl_data[f][key][ref_idx], (len_diff, 1))
            else:
                reps = np.tile(pkl_data[f][key][ref_idx], len_diff)
            pkl_data[f][key] = np.concatenate((pkl_data[f][key], reps))

    # Object motion (CSV in mm; rotated -90 deg around z to match world frame)
    df = parse_mocap_csv(object_csv, range(9))
    pose_array = df[["PosX", "PosY", "PosZ", "RotX", "RotY", "RotZ", "RotW"]].to_numpy()
    pose_array = np.vstack([pose_array, np.tile(pose_array[-1], (len(pkl_data["root_pose"]) - len(pose_array), 1))])
    rot = Rotation.from_euler("z", -90, degrees=True)
    rotated_positions = rot.apply(pose_array[:, :3])
    rotated_quaternions = (
        rot * Rotation.from_quat(pose_array[:, 3:]) * Rotation.from_euler("z", 90, degrees=True)
    ).as_quat()
    pkl_data["object_pose"] = np.zeros_like(pose_array)
    pkl_data["object_pose"][:, :3] = rotated_positions / 1000
    pkl_data["object_pose"][:, 3:] = rotated_quaternions
    pkl_data["object_pose"][:, 0] -= 0.1
    pkl_data["object_pose"][:, 2] -= 0.1

    # Lock the object to the midpoint of both hands during carry
    hands_offset = tf3.affines.compose([0.2605, 0, -0.0185], np.eye(3), np.ones(3))
    for idx in range(2100, 3400):
        ref_root = pkl_data["root_pose"][idx][[0, 1, 2, 6, 3, 4, 5]]
        ref_root_mat = tf3.affines.compose(ref_root[:3], tf3.quaternions.quat2mat(ref_root[3:]), np.ones(3))
        rel_pose = pkl_data["relative_link_pose"]["right_elbow_link"][idx][[0, 1, 2, 6, 3, 4, 5]]
        rel_pose_mat = tf3.affines.compose(rel_pose[:3], tf3.quaternions.quat2mat(rel_pose[3:]), np.ones(3))
        p1 = ref_root_mat.dot(rel_pose_mat.dot(hands_offset))[:3, 3]
        rel_pose = pkl_data["relative_link_pose"]["left_elbow_link"][idx][[0, 1, 2, 6, 3, 4, 5]]
        rel_pose_mat = tf3.affines.compose(rel_pose[:3], tf3.quaternions.quat2mat(rel_pose[3:]), np.ones(3))
        p2 = ref_root_mat.dot(rel_pose_mat.dot(hands_offset))[:3, 3]
        pkl_data["object_pose"][idx, :3] = (p1 + p2) / 2
    pkl_data["object_pose"][3400:] = pkl_data["object_pose"][3400 - 1]

    # Synthetic force schedule
    n = len(pkl_data["root_pose"])
    pkl_data["force_rfoot"] = np.zeros(n)
    pkl_data["force_lfoot"] = np.zeros(n)
    pkl_data["force_rhand"] = np.zeros(n)
    pkl_data["force_lhand"] = np.zeros(n)
    pkl_data["force_box"] = np.ones(n)
    for frc in ["lhand", "rhand"]:
        pkl_data["force_" + frc][2100:3400] = 1
    pkl_data["force_box"][2100:3400] = 0

    # Synthesize hand-link poses from elbow-link poses + a fixed offset
    hand_in_elbow = np.array([0.2605, 0, -0.0185, 0, 0, 0, 1])
    for idx in range(n):
        for s_side in ["right", "left"]:
            elbow_in_body = pkl_data["relative_link_pose"][s_side + "_elbow_link"][idx]
            R_elbow = Rotation.from_quat(elbow_in_body[3:]).as_matrix()
            hand_in_body_t = R_elbow.dot(hand_in_elbow[:3]) + elbow_in_body[:3]
            pkl_data["relative_link_pose"][s_side + "_hand_link"][idx] = np.concatenate(
                (hand_in_body_t, elbow_in_body[3:])
            )

    # Hand-to-target distances (target expressed in box frame)
    pkl_data["distance_right_hand_to_target"] = np.zeros(n)
    pkl_data["distance_left_hand_to_target"] = np.zeros(n)
    targets = {
        "right_hand": np.array([0.2, 0, 0]),
        "left_hand": np.array([-0.2, 0, 0]),
    }
    for idx in range(n):
        R_root = Rotation.from_quat(pkl_data["root_pose"][idx][3:])
        t_root = pkl_data["root_pose"][idx][:3]
        R_obj = Rotation.from_quat(pkl_data["object_pose"][idx][3:])
        t_obj = pkl_data["object_pose"][idx][:3]
        for s_side in ["right", "left"]:
            t_hand_local = pkl_data["relative_link_pose"][s_side + "_hand_link"][idx][:3]
            hand_in_world = R_root.apply(t_hand_local) + t_root
            target_in_world = R_obj.apply(targets[s_side + "_hand"]) + t_obj
            pkl_data["distance_" + s_side + "_hand_to_target"][idx] = np.linalg.norm(hand_in_world - target_in_world)
    pkl_data["distance_right_hand_to_target"][2000:3400] = 0
    pkl_data["distance_left_hand_to_target"][2000:3400] = 0

    # Zero out roll/pitch on feet rotations (keep yaw only)
    for bn in ["right_ankle_link", "left_ankle_link"]:
        ref_q = Rotation.from_quat(pkl_data["relative_link_pose"][bn][:, 3:])
        root_q = Rotation.from_quat(pkl_data["root_pose"][: len(ref_q.as_quat()), 3:])
        rotw = (root_q * ref_q).as_euler("xyz")
        rotw[:, 0] = 0
        rotw[:, 1] = 0
        rot_new = Rotation.from_euler("xyz", rotw)
        pkl_data["relative_link_pose"][bn][:, 3:] = (root_q.inv() * rot_new).as_quat()

    # Lift the root height slightly to avoid floor penetration
    pkl_data["root_pose"][:, 2] += 0.05

    # Slice out the requested mode segment
    start_idx, end_idx = mode.get_bounds()
    for attr in pkl_data:
        if attr == "dt":
            continue
        v = pkl_data[attr]
        if isinstance(v, np.ndarray):
            pkl_data[attr] = v[start_idx:end_idx]
        elif isinstance(v, dict):
            for key in v:
                v[key] = v[key][start_idx:end_idx]

    # Rewrite joint dict keys to drop the "_joint" suffix to match XML
    pkl_data["joint_position"] = _strip_joint_suffix(pkl_data["joint_position"])
    pkl_data["joint_velocity"] = _strip_joint_suffix(pkl_data["joint_velocity"])

    return pkl_data
