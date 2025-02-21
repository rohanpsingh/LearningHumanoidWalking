import os
import numpy as np
import models
from dm_control import mjcf
import transforms3d as tf3

H1_DESCRIPTION_PATH=os.path.join(os.path.dirname(models.__file__), "mujoco_menagerie/unitree_h1/scene.xml")

LEG_JOINTS = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
              "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]
WAIST_JOINTS = ["torso"]
ARM_JOINTS = ["left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
              "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow"]

def create_rangefinder_array(mjcf_model, num_rows=4, num_cols=4, spacing=0.4):
    for i in range(num_rows*num_cols):
        name = 'rf' + repr(i)
        u = (i % num_cols)
        v = (i // num_rows)
        x = (v - (num_cols - 1) / 2) * spacing
        y = ((num_rows - 1) / 2 - u) * (-spacing)
        # add sites
        mjcf_model.find('body', 'pelvis').add('site',
                                              name=name,
                                              pos=[x, y, 0],
                                              quat='0 1 0 0')

        # add sensors
        mjcf_model.sensor.add('rangefinder',
                              name=name,
                              site=name)

    return mjcf_model

def remove_joints_and_actuators(mjcf_model, config):
    # remove joints
    for limb in config['unused_joints']:
        for joint in limb:
            mjcf_model.find('joint', joint).remove()

    # remove all actuators with no corresponding joint
    for mot in mjcf_model.actuator.motor:
        mot.user = None
        if mot.joint==None:
            mot.remove()
    return mjcf_model

def builder(export_path, config):
    print("Modifying XML model...")
    mjcf_model = mjcf.from_path(H1_DESCRIPTION_PATH)

    mjcf_model.model = 'h1'

    # modify model
    mjcf_model = remove_joints_and_actuators(mjcf_model, config)
    mjcf_model.find('default', 'visual').geom.group = 1
    mjcf_model.find('default', 'collision').geom.group = 2
    if 'ctrllimited' in config:
        mjcf_model.find('default', 'h1').motor.ctrllimited = config['ctrllimited']
    if 'jointlimited' in config:
        mjcf_model.find('default', 'h1').joint.limited = config['jointlimited']

    # rename all motors to fit assumed convention
    for mot in mjcf_model.actuator.motor:
        mot.name = mot.name + "_motor"

    # remove keyframe for now
    mjcf_model.keyframe.remove()

    # remove vis geoms and assets, if needed
    if 'minimal' in config:
        if config['minimal']:
            mjcf_model.find('default', 'collision').geom.group = 1
            meshes = mjcf_model.asset.mesh
            for mesh in meshes:
                mesh.remove()
            for geom in mjcf_model.find_all('geom'):
                if geom.dclass:
                    if geom.dclass.dclass=="visual":
                        geom.remove()

    # set name of freejoint
    mjcf_model.find('body', 'pelvis').freejoint.name = 'root'

    # add rangefinder
    if 'rangefinder' in config:
        if config['rangefinder']:
            mjcf_model = create_rangefinder_array(mjcf_model)

    # create a raised platform
    if 'raisedplatform' in config:
        if config['raisedplatform']:
            block_pos = [2.5, 0, 0]
            block_size = [3, 3, 1]
            name = 'raised-platform'
            mjcf_model.worldbody.add('body', name=name, pos=block_pos)
            mjcf_model.find('body', name).add('geom', name=name, group='3',
                                              condim='3', friction='.8 .1 .1', size=block_size,
                                              type='box', material='')

    # set some size options
    mjcf_model.size.njmax = "-1"
    mjcf_model.size.nconmax = "-1"
    mjcf_model.size.nuser_actuator = "-1"

    # export model
    mjcf.export_with_assets(mjcf_model, out_dir=export_path, precision=5)
    path_to_xml = os.path.join(export_path, mjcf_model.model + '.xml')
    print("Exporting XML model to ", path_to_xml)
    return
