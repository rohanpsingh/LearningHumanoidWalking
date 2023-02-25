import sys
import os
from dm_control import mjcf
import random
import string

JVRC_DESCRIPTION_PATH="models/jvrc_mj_description/xml/scene.xml"

def builder(export_path):

    print("Modifying XML model...")
    mjcf_model = mjcf.from_path(JVRC_DESCRIPTION_PATH)

    # set njmax and nconmax
    mjcf_model.size.njmax = 1200
    mjcf_model.size.nconmax = 400

    # remove all collisions
    mjcf_model.contact.remove()

    waist_joints = ['WAIST_Y', 'WAIST_P', 'WAIST_R']
    head_joints = ['NECK_Y', 'NECK_R', 'NECK_P']
    hand_joints = ['R_UTHUMB', 'R_LTHUMB', 'R_UINDEX', 'R_LINDEX', 'R_ULITTLE', 'R_LLITTLE',
                   'L_UTHUMB', 'L_LTHUMB', 'L_UINDEX', 'L_LINDEX', 'L_ULITTLE', 'L_LLITTLE']
    arm_joints = ['R_SHOULDER_Y', 'R_ELBOW_Y', 'R_WRIST_R', 'R_WRIST_Y',
                  'L_SHOULDER_Y', 'L_ELBOW_Y', 'L_WRIST_R', 'L_WRIST_Y']
    leg_joints = ['R_HIP_P', 'R_HIP_R', 'R_HIP_Y', 'R_KNEE', 'R_ANKLE_R', 'R_ANKLE_P',
                  'L_HIP_P', 'L_HIP_R', 'L_HIP_Y', 'L_KNEE', 'L_ANKLE_R', 'L_ANKLE_P']

    # remove actuators except for leg joints
    for mot in mjcf_model.actuator.motor:
        if mot.joint.name not in leg_joints:
            mot.remove()

    # remove unused joints
    for joint in waist_joints + head_joints + hand_joints + arm_joints:
        mjcf_model.find('joint', joint).remove()

    # remove existing equality
    mjcf_model.equality.remove()

    # add equality for arm joints
    arm_joints = ['R_SHOULDER_P', 'R_SHOULDER_R', 'R_ELBOW_P',
                  'L_SHOULDER_P', 'L_SHOULDER_R', 'L_ELBOW_P']
    mjcf_model.equality.add('joint', joint1=arm_joints[0], polycoef='-0.052 0 0 0 0')
    mjcf_model.equality.add('joint', joint1=arm_joints[1], polycoef='-0.169 0 0 0 0')
    mjcf_model.equality.add('joint', joint1=arm_joints[2], polycoef='-0.523 0 0 0 0')
    mjcf_model.equality.add('joint', joint1=arm_joints[3], polycoef='-0.052 0 0 0 0')
    mjcf_model.equality.add('joint', joint1=arm_joints[4], polycoef='0.169 0 0 0 0')
    mjcf_model.equality.add('joint', joint1=arm_joints[5], polycoef='-0.523 0 0 0 0')

    # collision geoms
    collision_geoms = [
        'R_HIP_R_S', 'R_HIP_Y_S', 'R_KNEE_S',
        'L_HIP_R_S', 'L_HIP_Y_S', 'L_KNEE_S',
    ]

    # remove unused collision geoms
    for body in mjcf_model.worldbody.find_all('body'):
        for idx, geom in enumerate(body.geom):
            geom.name = body.name + '-geom-' + repr(idx)
            if (geom.dclass.dclass=="collision"):
                if body.name not in collision_geoms:
                    geom.remove()

    # manually create collision geom for feet
    mjcf_model.worldbody.find('body', 'R_ANKLE_P_S').add('geom', dclass='collision', size='0.1 0.05 0.01', pos='0.029 0 -0.09778', type='box')
    mjcf_model.worldbody.find('body', 'L_ANKLE_P_S').add('geom', dclass='collision', size='0.1 0.05 0.01', pos='0.029 0 -0.09778', type='box')

    # ignore collision
    mjcf_model.contact.add('exclude', body1='R_KNEE_S', body2='R_ANKLE_P_S')
    mjcf_model.contact.add('exclude', body1='L_KNEE_S', body2='L_ANKLE_P_S')

    # remove unused meshes
    meshes = [g.mesh.name for g in mjcf_model.find_all('geom') if g.type=='mesh' or g.type==None]
    for mesh in mjcf_model.find_all('mesh'):
        if mesh.name not in meshes:
            mesh.remove()

    # fix site pos
    mjcf_model.worldbody.find('site', 'rf_force').pos = '0.03 0.0 -0.1'
    mjcf_model.worldbody.find('site', 'lf_force').pos = '0.03 0.0 -0.1'

    # add box geoms
    for idx in range(20):
        mjcf_model.worldbody.add('geom',
                                 name='box' + repr(idx+1).zfill(2),
                                 pos='0 0 -0.2',
                                 dclass='collision',
                                 group='0',
                                 size='1 1 0.1',
                                 type='box',
                                 material='')

    # export model
    mjcf.export_with_assets(mjcf_model, out_dir=os.path.dirname(export_path), out_file_name=export_path, precision=5)
    print("Exporting XML model to ", export_path)
    return

if __name__=='__main__':
    builder(sys.argv[1])
