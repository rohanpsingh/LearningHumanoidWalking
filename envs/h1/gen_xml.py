import os

from dm_control import mjcf

import models

H1_DESCRIPTION_PATH = os.path.join(os.path.dirname(models.__file__), "mujoco_menagerie/unitree_h1/scene.xml")
TERRAINS_PATH = os.path.join(os.path.dirname(models.__file__), "terrains/hfield.png")

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


def create_rangefinder_array(mjcf_model, num_rows=4, num_cols=4, spacing=0.4):
    for i in range(num_rows * num_cols):
        name = "rf" + repr(i)
        u = i % num_cols
        v = i // num_rows
        x = (v - (num_cols - 1) / 2) * spacing
        y = ((num_rows - 1) / 2 - u) * (-spacing)
        # add sites
        mjcf_model.find("body", "pelvis").add("site", name=name, pos=[x, y, 0], quat="0 1 0 0")

        # add sensors
        mjcf_model.sensor.add("rangefinder", name=name, site=name)

    return mjcf_model


def create_hfield(mjcf_model):
    mjcf_model.asset.add(
        "hfield",
        name="hf1",
        size="2.5 2.5 0.08 0.01",
        file=TERRAINS_PATH,
    )
    mjcf_model.worldbody.add("body", name="hfield", pos=[0, 0, -0.1])
    mjcf_model.find("body", "hfield").add(
        "geom",
        name="hfield",
        type="hfield",
        dclass="collision",
        condim="3",
        conaffinity="15",
        hfield="hf1",
        friction=".8 .1 .1",
    )
    return mjcf_model


def create_mocap_bodies(mjcf_model, body_names):
    """Add mocap-target bodies for visualizing the reference motion.

    Each target is a single sphere marker (not a geometry-faithful clone of
    the source body). Geometry-faithful cloning is brittle when source geoms
    inherit type/size from default classes (e.g. the H1 menagerie ankle).
    """
    for bn in body_names:
        b = mjcf_model.find("body", bn)
        if b is None:
            continue
        attr = b.get_attributes()
        attr["name"] = "mocap_" + b.name
        attr["mocap"] = "true"
        mocap_b = mjcf_model.worldbody.add("body")
        mocap_b.set_attributes(**attr)
        mocap_b.add("inertial", pos="0 0 0", mass="0")
        mocap_b.add("geom", dclass="mocap", type="sphere", size="0.04")
    return mjcf_model


def add_movable_box(mjcf_model, name):
    body = mjcf_model.worldbody.add("body", name=name, pos=[1.5, -0.8, 0.81], euler=[0, 0, 1.57])
    body.add("inertial", pos="0 0 0", mass="1.36", diaginertia="0.01655043 0.01566667 0.00995043")
    body.add("joint", type="free")
    for s in ["visual", "collision"]:
        geom0 = body.add("geom", name=name + "-geom0-" + s, size=[0.20, 0.1, 0.01], pos=[0, 0, 0.1])
        geom1 = body.add("geom", name=name + "-geom1-" + s, size=[0.18, 0.1, 0.1], pos=[0, 0, 0])
        for geom in [geom0, geom1]:
            geom.dclass = mjcf_model.default.default["h1"].default[s]
            geom.type = "box"
            geom.rgba = [1, 0, 0, 1]
    body.add("site", name="box", size="0.01", pos=[0, 0, 0])

    # Two fixed tables for pickup / dropoff
    body = mjcf_model.worldbody.add("body", name="table1", pos=[1.5, -0.8, 0.65])
    for s in ["visual", "collision"]:
        body.add(
            "geom",
            name="table1-" + s,
            dclass=s,
            size="0.25 0.25 0.05",
            type="box",
            pos=[0, 0, 0],
            rgba="0.18 0. 0.30 1",
            density="0.001",
        )
    body.add("site", name="table1", size="0.01", pos=[0, 0, 0.05])

    body = mjcf_model.worldbody.add("body", name="table2", pos=[0.47, -2.7, 0.65])
    for s in ["visual", "collision"]:
        body.add(
            "geom",
            name="table2-" + s,
            dclass=s,
            size="0.25 0.25 0.05",
            type="box",
            pos=[0, 0, 0],
            rgba="0.18 0. 0.30 1",
            density="0.001",
        )
    body.add("site", name="table2", size="0.01", pos=[0, 0, 0.05])
    return mjcf_model


def add_hand_sites(mjcf_model):
    """Add hand sites at the end-effector of each elbow link.

    The H1 menagerie model has no hand bodies; we approximate hand position
    with sites offset from the elbow link, mirroring the internal mocap rig.
    """
    mjcf_model.find("body", "left_elbow_link").add("site", name="left_hand", pos="0.2605 0 -0.0185", size="0.01")
    mjcf_model.find("body", "right_elbow_link").add("site", name="right_hand", pos="0.2605 0 -0.0185", size="0.01")
    return mjcf_model


def remove_joints_and_actuators(mjcf_model, config):
    # remove joints
    for limb in config["unused_joints"]:
        for joint in limb:
            mjcf_model.find("joint", joint).remove()

    # remove all actuators with no corresponding joint
    for mot in mjcf_model.actuator.motor:
        mot.user = None
        if mot.joint is None:
            mot.remove()
    return mjcf_model


def builder(export_path, config):
    print("Modifying XML model...")
    mjcf_model = mjcf.from_path(H1_DESCRIPTION_PATH)

    mjcf_model.model = "h1"

    # modify model
    mjcf_model = remove_joints_and_actuators(mjcf_model, config)
    mjcf_model.find("default", "visual").geom.group = 1
    mjcf_model.find("default", "collision").geom.group = 2
    if "ctrllimited" in config:
        mjcf_model.find("default", "h1").motor.ctrllimited = config["ctrllimited"]
    if "jointlimited" in config:
        mjcf_model.find("default", "h1").joint.limited = config["jointlimited"]

    # rename all motors to fit assumed convention
    for mot in mjcf_model.actuator.motor:
        mot.name = mot.name + "_motor"

    # remove keyframe for now
    mjcf_model.keyframe.remove()

    # remove vis geoms and assets, if needed
    if "minimal" in config:
        if config["minimal"]:
            mjcf_model.find("default", "collision").geom.group = 1
            meshes = mjcf_model.asset.mesh
            for mesh in meshes:
                mesh.remove()
            for geom in mjcf_model.find_all("geom"):
                if geom.dclass:
                    if geom.dclass.dclass == "visual":
                        geom.remove()

    # set name of freejoint
    mjcf_model.find("body", "pelvis").freejoint.name = "root"

    # add hand sites (for tasks that need end-effector targets)
    if config.get("hand_sites", False):
        mjcf_model = add_hand_sites(mjcf_model)

    # add a movable box and surrounding tables
    if "movable_box" in config and config["movable_box"]:
        mjcf_model = add_movable_box(mjcf_model, config["movable_box"])

    # add mocap bodies (visualization of reference motion)
    if config.get("mocap_bodies"):
        mocap_default = mjcf_model.default.default["h1"].add("default", dclass="mocap")
        mocap_default.geom.contype = "0"
        mocap_default.geom.conaffinity = "0"
        mocap_default.geom.group = "3"
        mocap_default.geom.rgba = "0 1 0 0.5"
        mocap_default.geom.density = "0"
        mjcf_model = create_mocap_bodies(mjcf_model, config["mocap_bodies"])

    # add an uneven heightfield underneath the robot
    if config.get("hfield", False):
        mjcf_model = create_hfield(mjcf_model)

    # add rangefinder
    if "rangefinder" in config:
        if config["rangefinder"]:
            mjcf_model = create_rangefinder_array(mjcf_model)

    # create a raised platform
    if "raisedplatform" in config:
        if config["raisedplatform"]:
            block_pos = [2.5, 0, 0]
            block_size = [3, 3, 1]
            name = "raised-platform"
            mjcf_model.worldbody.add("body", name=name, pos=block_pos)
            mjcf_model.find("body", name).add(
                "geom", name=name, group="3", condim="3", friction=".8 .1 .1", size=block_size, type="box", material=""
            )

    # set some size options
    mjcf_model.size.njmax = "-1"
    mjcf_model.size.nconmax = "-1"
    mjcf_model.size.nuser_actuator = "-1"

    # export model
    mjcf.export_with_assets(mjcf_model, out_dir=export_path, precision=5)
    path_to_xml = os.path.join(export_path, mjcf_model.model + ".xml")
    print("Exporting XML model to ", path_to_xml)
    return
