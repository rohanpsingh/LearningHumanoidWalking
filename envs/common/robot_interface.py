import numpy as np
import transforms3d as tf3
import mujoco

class RobotInterface(object):
    def __init__(self, model, data, rfoot_body_name=None, lfoot_body_name=None):
        self.model = model
        self.data = data

        self.rfoot_body_name = rfoot_body_name
        self.lfoot_body_name = lfoot_body_name
        self.floor_body_name = 'world'
        self.robot_root_name = 'PELVIS_S'

        self.stepCounter = 0

    def nq(self):
        return self.model.nq

    def nu(self):
        return self.model.nu

    def nv(self):
        return self.model.nv

    def sim_dt(self):
        return self.model.opt.timestep

    def get_robot_mass(self):
        return mujoco.mj_getTotalmass(self.model)

    def get_qpos(self):
        return self.data.qpos.copy()

    def get_qvel(self):
        return self.data.qvel.copy()

    def get_qacc(self):
        return self.data.qacc.copy()

    def get_cvel(self):
        return self.data.cvel.copy()

    def get_jnt_id_by_name(self, name):
        return self.model.joint(name)

    def get_jnt_qposadr_by_name(self, name):
        return self.model.joint(name).qposadr

    def get_jnt_qveladr_by_name(self, name):
        return self.model.joint(name).dofadr

    def get_body_ext_force(self):
        return self.data.cfrc_ext.copy()

    def get_motor_speed_limits(self):
        """
        Returns speed limits of the *actuator* in radians per sec.
        This assumes the actuator 'user' element defines speed limits
        at the actuator level in revolutions per minute.
        """
        rpm_limits = self.model.actuator_user[:,0] # RPM
        return ((rpm_limits)*(2*np.pi/60)) # radians per sec

    def get_act_joint_speed_limits(self):
        """
        Returns speed limits of the *joint* in radians per sec.
        This assumes the actuator 'user' element defines speed limits
        at the actuator level in revolutions per minute.
        """
        gear_ratios = self.model.actuator_gear[:,0]
        mot_lims = self.get_motor_speed_limits()
        return [float(i/j) for i,j in zip(mot_lims, gear_ratios)]

    def get_gear_ratios(self):
        """
        Returns transmission ratios.
        """
        return self.model.actuator_gear[:,0]

    def get_motor_names(self):
        actuator_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self.model.nu)]
        return actuator_names

    def get_actuated_joint_inds(self):
        """
        Returns list of joint indices to which actuators are attached.
        """
        joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(self.model.njnt)]
        actuator_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self.model.nu)]
        return [idx for idx, jnt in enumerate(joint_names) if jnt+'_motor' in actuator_names]

    def get_actuated_joint_names(self):
        """
        Returns list of joint names to which actuators are attached.
        """
        joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(self.model.njnt)]
        actuator_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self.model.nu)]
        return [jnt for idx, jnt in enumerate(joint_names) if jnt+'_motor' in actuator_names]

    def get_motor_qposadr(self):
        """
        Returns the list of qpos indices of all actuated joints.
        """
        indices = self.get_actuated_joint_inds()
        return [self.model.jnt_qposadr[i] for i in indices]

    def get_motor_positions(self):
        """
        Returns position of actuators.
        """
        return self.data.actuator_length

    def get_motor_velocities(self):
        """
        Returns velocities of actuators.
        """
        return self.data.actuator_velocity

    def get_act_joint_torques(self):
        """
        Returns actuator force in joint space.
        """
        gear_ratios = self.model.actuator_gear[:,0]
        motor_torques = self.data.actuator_force.tolist()
        return [float(i*j) for i,j in zip(motor_torques, gear_ratios)]

    def get_act_joint_positions(self):
        """
        Returns position of actuators at joint level.
        """
        gear_ratios = self.model.actuator_gear[:,0]
        motor_positions = self.get_motor_positions()
        return (motor_positions/gear_ratios)

    def get_act_joint_velocities(self):
        """
        Returns velocities of actuators at joint level.
        """
        gear_ratios = self.model.actuator_gear[:,0]
        motor_velocities = self.get_motor_velocities()
        return (motor_velocities/gear_ratios)

    def get_act_joint_position(self, act_name):
        """
        Returns position of actuator at joint level.
        """
        assert len(self.data.actuator(act_name).length)==1
        return self.data.actuator(act_name).length[0]/self.model.actuator(act_name).gear[0]

    def get_act_joint_velocity(self, act_name):
        """
        Returns velocity of actuator at joint level.
        """
        assert len(self.data.actuator(act_name).velocity)==1
        return self.data.actuator(act_name).velocity[0]/self.model.actuator(act_name).gear[0]

    def get_act_joint_ranges(self):
        """
        Returns the lower and upper limits of all actuated joints.
        """
        indices = self.get_actuated_joint_inds()
        low, high = self.model.jnt_range[indices, :].T
        return low, high

    def get_act_joint_range(self, act_name):
        """
        Returns the lower and upper limits of given joint.
        """
        low, high = self.model.joint(act_name).range
        return low, high

    def get_actuator_ctrl_range(self):
        """
        Returns the acutator ctrlrange defined in model xml.
        """
        low, high = self.model.actuator_ctrlrange.copy().T
        return low, high

    def get_actuator_user_data(self):
        """
        Returns the user data (if any) attached to each actuator.
        """
        return self.model.actuator_user.copy()

    def get_root_body_pos(self):
        return self.data.xpos[1].copy()

    def get_root_body_vel(self):
        qveladr = self.get_jnt_qveladr_by_name("root")
        return self.data.qvel[qveladr:qveladr+6].copy()

    def get_sensordata(self, sensor_name):
        sensor = self.model.sensor(sensor_name)
        sensor_adr = sensor.adr[0]
        data_dim = sensor.dim[0]
        return self.data.sensordata[sensor_adr:sensor_adr+data_dim]

    def get_rfoot_body_pos(self):
        if isinstance(self.rfoot_body_name, list):
            return [self.data.body(i).xpos.copy() for i in self.rfoot_body_name]
        return self.data.body(self.rfoot_body_name).xpos.copy()

    def get_lfoot_body_pos(self):
        if isinstance(self.lfoot_body_name, list):
            return [self.data.body(i).xpos.copy() for i in self.lfoot_body_name]
        return self.data.body(self.lfoot_body_name).xpos.copy()

    def get_rfoot_floor_contacts(self):
        """
        Returns list of right foot and floor contacts.
        """
        contacts = [self.data.contact[i] for i in range(self.data.ncon)]
        rcontacts = []

        rfeet = [self.rfoot_body_name] if isinstance(self.rfoot_body_name, str) else self.rfoot_body_name
        rfeet_ids = [self.model.body(bn).id for bn in rfeet]
        for i,c in enumerate(contacts):
            geom1_body = self.model.body(self.model.geom_bodyid[c.geom1])
            geom2_body = self.model.body(self.model.geom_bodyid[c.geom2])
            geom1_is_floor = (self.model.body(geom1_body.rootid).name!=self.robot_root_name)
            geom2_is_rfoot = (self.model.geom_bodyid[c.geom2] in rfeet_ids)
            if (geom1_is_floor and geom2_is_rfoot):
                rcontacts.append((i,c))
        return rcontacts

    def get_lfoot_floor_contacts(self):
        """
        Returns list of left foot and floor contacts.
        """
        contacts = [self.data.contact[i] for i in range(self.data.ncon)]
        lcontacts = []

        lfeet = [self.lfoot_body_name] if isinstance(self.lfoot_body_name, str) else self.lfoot_body_name
        lfeet_ids = [self.model.body(bn).id for bn in lfeet]
        for i,c in enumerate(contacts):
            geom1_body = self.model.body(self.model.geom_bodyid[c.geom1])
            geom2_body = self.model.body(self.model.geom_bodyid[c.geom2])
            geom1_is_floor = (self.model.body(geom1_body.rootid).name!=self.robot_root_name)
            geom2_is_lfoot = (self.model.geom_bodyid[c.geom2] in lfeet_ids)
            if (geom1_is_floor and geom2_is_lfoot):
                lcontacts.append((i,c))
        return lcontacts

    def get_rfoot_grf(self):
        """
        Returns total Ground Reaction Force on right foot.
        """
        right_contacts = self.get_rfoot_floor_contacts()
        rfoot_grf = 0
        for i, con in right_contacts:
            c_array = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, c_array)
            rfoot_grf += np.linalg.norm(c_array)
        return rfoot_grf

    def get_lfoot_grf(self):
        """
        Returns total Ground Reaction Force on left foot.
        """
        left_contacts = self.get_lfoot_floor_contacts()
        lfoot_grf = 0
        for i, con in left_contacts:
            c_array = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, c_array)
            lfoot_grf += (np.linalg.norm(c_array))
        return lfoot_grf

    def get_body_vel(self, body_name, frame=0):
        """
        Returns translational and rotational velocity of a body in body-centered frame, world/local orientation.
        """
        body_vel = np.zeros(6)
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY,
                                 body_id, body_vel, frame)
        return [body_vel[3:6], body_vel[0:3]]

    def get_rfoot_body_vel(self, frame=0):
        """
        Returns translational and rotational velocity of right foot.
        """
        if isinstance(self.rfoot_body_name, list):
            return [self.get_body_vel(i, frame=frame) for i in self.rfoot_body_name]
        return self.get_body_vel(self.rfoot_body_name, frame=frame)

    def get_lfoot_body_vel(self, frame=0):
        """
        Returns translational and rotational velocity of left foot.
        """
        if isinstance(self.lfoot_body_name, list):
            return [self.get_body_vel(i, frame=frame) for i in self.lfoot_body_name]
        return self.get_body_vel(self.lfoot_body_name, frame=frame)

    def get_object_xpos_by_name(self, obj_name, object_type):
        if object_type=="OBJ_BODY":
            return self.data.body(obj_name).xpos
        elif object_type=="OBJ_GEOM":
            return self.data.geom(obj_name).xpos
        elif object_type=="OBJ_SITE":
            return self.data.site(obj_name).xpos
        else:
            raise Exception("object type should either be OBJ_BODY/OBJ_GEOM/OBJ_SITE.")

    def get_object_xquat_by_name(self, obj_name, object_type):
        if object_type=="OBJ_BODY":
            return self.data.body(obj_name).xquat
        if object_type=="OBJ_SITE":
            xmat = self.data.site(obj_name).xmat
            return tf3.quaternions.mat2quat(xmat)
        else:
            raise Exception("object type should be OBJ_BODY/OBJ_SITE.")

    def get_robot_com(self):
        """
        Returns the center of mass of subtree originating at root body
        i.e. the CoM of the entire robot body in world coordinates.
        """
        sensor_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(self.model.nsensor)]
        if 'subtreecom' not in sensor_names:
            raise Exception("subtree_com sensor not attached.")
        return self.data.subtree_com[1].copy()

    def get_robot_linmom(self):
        """
        Returns linear momentum of robot in world coordinates.
        """
        sensor_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(self.model.nsensor)]
        if 'subtreelinvel' not in sensor_names:
            raise Exception("subtree_linvel sensor not attached.")
        linvel = self.data.subtree_linvel[1].copy()
        total_mass = self.get_robot_mass()
        return linvel*total_mass

    def get_robot_angmom(self):
        """
        Return angular momentum of robot's CoM about the world origin.
        """
        sensor_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(self.model.nsensor)]
        if 'subtreeangmom' not in sensor_names:
            raise Exception("subtree_angmom sensor not attached.")
        com_pos = self.get_robot_com()
        lin_mom = self.get_robot_linmom()
        return self.data.subtree_angmom[1] + np.cross(com_pos, lin_mom)


    def check_rfoot_floor_collision(self):
        """
        Returns True if there is a collision between right foot and floor.
        """
        return (len(self.get_rfoot_floor_contacts())>0)

    def check_lfoot_floor_collision(self):
        """
        Returns True if there is a collision between left foot and floor.
        """
        return (len(self.get_lfoot_floor_contacts())>0)

    def check_bad_collisions(self):
        """
        Returns True if there are collisions other than feet-floor.
        """
        num_rcons = len(self.get_rfoot_floor_contacts())
        num_lcons = len(self.get_lfoot_floor_contacts())
        return (num_rcons + num_lcons) != self.data.ncon

    def check_self_collisions(self):
        """
        Returns True if there are collisions other than any-geom-floor.
        """
        contacts = [self.data.contact[i] for i in range(self.data.ncon)]
        for i,c in enumerate(contacts):
            geom1_body = self.model.body(self.model.geom_bodyid[c.geom1])
            geom2_body = self.model.body(self.model.geom_bodyid[c.geom2])
            geom1_is_robot = self.model.body(geom1_body.rootid).name==self.robot_root_name
            geom2_is_robot = self.model.body(geom2_body.rootid).name==self.robot_root_name
            if geom1_is_robot and geom2_is_robot:
                return True
        return False

    def set_pd_gains(self, kp, kv):
        assert kp.size==self.model.nu
        assert kv.size==self.model.nu
        self.kp = kp.copy()
        self.kv = kv.copy()
        return

    def step_pd(self, p, v):
        target_angles = p
        target_speeds = v

        assert type(target_angles)==np.ndarray
        assert type(target_speeds)==np.ndarray

        curr_angles = self.get_act_joint_positions()
        curr_speeds = self.get_act_joint_velocities()

        perror = target_angles - curr_angles
        verror = target_speeds - curr_speeds

        assert self.kp.size==perror.size
        assert self.kv.size==verror.size
        return self.kp * perror + self.kv * verror

    def set_motor_torque(self, torque):
        """
        Apply torques to motors.
        """
        if isinstance(torque, np.ndarray):
            assert torque.shape==(self.nu(), )
            ctrl = torque
        elif isinstance(torque, list):
            assert len(torque)==self.nu()
            ctrl = np.copy(torque)
        else:
            raise Exception("motor torque should be list of ndarray.")
        try:
            np.copyto(self.data.ctrl, ctrl)
        except Exception as e:
            print("Could not apply motor torque.")
            print(e)
        return

    def step(self, mj_step=True, nstep=1):
        """
        (Adapted from dm_control/mujoco/engine.py)

        Advances physics with up-to-date position and velocity dependent fields.
        Args:
          nstep: Optional integer, number of steps to take.
        """
        if mj_step:
            mujoco.mj_step(self.model, self.data, nstep)
            self.stepCounter += nstep
            return

        # In the case of Euler integration we assume mj_step1 has already been
        # called for this state, finish the step with mj_step2 and then update all
        # position and velocity related fields with mj_step1. This ensures that
        # (most of) mjData is in sync with qpos and qvel. In the case of non-Euler
        # integrators (e.g. RK4) an additional mj_step1 must be called after the
        # last mj_step to ensure mjData syncing.
        if self.model.opt.integrator != mujoco.mjtIntegrator.mjINT_RK4.value:
          mujoco.mj_step2(self.model, self.data)
          if nstep > 1:
            mujoco.mj_step(self.model, self.data, nstep-1)
        else:
          mujoco.mj_step(self.model, self.data, nstep)

        mujoco.mj_step1(self.model, self.data)

        self.stepCounter += nstep
