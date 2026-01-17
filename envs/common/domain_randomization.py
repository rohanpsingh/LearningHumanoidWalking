"""Domain randomization utilities for sim-to-real transfer.

This module provides functions to randomize simulation parameters
during training to improve policy robustness for real robot deployment.
"""

import numpy as np


def apply_perturbation(data, cfg):
    """Apply random perturbation forces to robot bodies.

    Args:
        data: MuJoCo data object.
        cfg: Perturbation config with fields:
            - bodies: List of body names to perturb.
            - force_magnitude: Max force magnitude.
            - torque_magnitude: Max torque magnitude.
    """
    frc_mag = cfg.force_magnitude
    tau_mag = cfg.torque_magnitude
    for body in cfg.bodies:
        data.body(body).xfrc_applied[:3] = np.random.uniform(-frc_mag, frc_mag, 3)
        data.body(body).xfrc_applied[3:] = np.random.uniform(-tau_mag, tau_mag, 3)
        if np.random.randint(2) == 0:
            data.xfrc_applied = np.zeros_like(data.xfrc_applied)


def randomize_dynamics(model, default_model, interface, joint_names, cfg):
    """Randomize dynamics parameters (friction, damping, mass, CoM).

    Args:
        model: MuJoCo model object.
        default_model: Copy of the original model for reference.
        interface: RobotInterface instance.
        joint_names: List of joint names to randomize.
        cfg: Dynamics randomization config (currently unused, for future extension).
    """
    # Randomize joint friction and damping
    dofadr = [interface.get_jnt_qveladr_by_name(jn) for jn in joint_names]
    for jnt in dofadr:
        model.dof_frictionloss[jnt] = np.random.uniform(0, 2)
        model.dof_damping[jnt] = np.random.uniform(0.02, 2)

    # Randomize center of mass and mass
    bodies = ["pelvis"]
    for legjoint in joint_names:
        bodyid = model.joint(legjoint).bodyid
        bodyname = model.body(bodyid).name
        bodies.append(bodyname)

    for body in bodies:
        default_mass = default_model.body(body).mass[0]
        default_ipos = default_model.body(body).ipos
        model.body(body).mass[0] = default_mass * np.random.uniform(0.95, 1.05)
        model.body(body).ipos = default_ipos + np.random.uniform(-0.01, 0.01, 3)
