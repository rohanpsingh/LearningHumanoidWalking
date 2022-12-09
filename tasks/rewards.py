import numpy as np

##############################
##############################
# Define reward functions here
##############################
##############################

def _calc_fwd_vel_reward(self):
    # forward vel reward
    root_vel = self._client.get_qvel()[0]
    error = np.linalg.norm(root_vel - self._goal_speed_ref)
    return np.exp(-error)

def _calc_action_reward(self, prev_action):
    # action reward
    action = self._client.get_pd_target()[0]
    penalty = 5 * sum(np.abs(prev_action - action)) / len(action)
    return np.exp(-penalty)

def _calc_torque_reward(self, prev_torque):
    # actuator torque reward
    torque = np.asarray(self._client.get_act_joint_torques())
    penalty = 0.25 * (sum(np.abs(prev_torque - torque)) / len(torque))
    return np.exp(-penalty)

def _calc_height_reward(self):
    # height reward
    if self._client.check_rfoot_floor_collision() or self._client.check_lfoot_floor_collision():
        contact_point = min([c.pos[2] for _,c in (self._client.get_rfoot_floor_contacts() +
                                                  self._client.get_lfoot_floor_contacts())])
    else:
        contact_point = 0
    current_height = self._client.get_object_xpos_by_name(self._root_body_name, 'OBJ_BODY')[2]
    relative_height = current_height - contact_point
    error = np.abs(relative_height - self._goal_height_ref)
    deadzone_size = 0.01 + 0.05 * self._goal_speed_ref
    if error < deadzone_size:
        error = 0
    return np.exp(-40*np.square(error))

def _calc_heading_reward(self):
    # heading reward
    cur_heading = self._client.get_qvel()[:3]
    cur_heading /= np.linalg.norm(cur_heading)
    error = np.linalg.norm(cur_heading - np.array([1, 0, 0]))
    return np.exp(-error)

def _calc_root_accel_reward(self):
    qvel = self._client.get_qvel()
    qacc = self._client.get_qacc()
    error = 0.25 * (np.abs(qvel[3:6]).sum() + np.abs(qacc[0:3]).sum())
    return np.exp(-error)

def _calc_feet_separation_reward(self):
    # feet y-separation cost
    rfoot_pos = self._client.get_rfoot_body_pos()[1]
    lfoot_pos = self._client.get_lfoot_body_pos()[1]
    foot_dist = np.abs(rfoot_pos-lfoot_pos)
    error = 5*np.square(foot_dist-0.35)
    if foot_dist < 0.40 and foot_dist > 0.30:
        error = 0
    return np.exp(-error)

def _calc_foot_frc_clock_reward(self, left_frc_fn, right_frc_fn):
    # constraints of foot forces based on clock
    desired_max_foot_frc = self._client.get_robot_mass()*9.8*0.5
    #desired_max_foot_frc = self._client.get_robot_mass()*10*1.2
    normed_left_frc = min(self.l_foot_frc, desired_max_foot_frc) / desired_max_foot_frc
    normed_right_frc = min(self.r_foot_frc, desired_max_foot_frc) / desired_max_foot_frc
    normed_left_frc*=2
    normed_left_frc-=1
    normed_right_frc*=2
    normed_right_frc-=1

    left_frc_clock = left_frc_fn(self._phase)
    right_frc_clock = right_frc_fn(self._phase)

    left_frc_score = np.tan(np.pi/4 * left_frc_clock * normed_left_frc)
    right_frc_score = np.tan(np.pi/4 * right_frc_clock * normed_right_frc)

    foot_frc_score = (left_frc_score + right_frc_score)/2
    return foot_frc_score

def _calc_foot_vel_clock_reward(self, left_vel_fn, right_vel_fn):
    # constraints of foot velocities based on clock
    desired_max_foot_vel = 0.2
    normed_left_vel = min(np.linalg.norm(self.l_foot_vel), desired_max_foot_vel) / desired_max_foot_vel
    normed_right_vel = min(np.linalg.norm(self.r_foot_vel), desired_max_foot_vel) / desired_max_foot_vel
    normed_left_vel*=2
    normed_left_vel-=1
    normed_right_vel*=2
    normed_right_vel-=1

    left_vel_clock = left_vel_fn(self._phase)
    right_vel_clock = right_vel_fn(self._phase)

    left_vel_score = np.tan(np.pi/4 * left_vel_clock * normed_left_vel)
    right_vel_score = np.tan(np.pi/4 * right_vel_clock * normed_right_vel)

    foot_vel_score = (left_vel_score + right_vel_score)/2
    return foot_vel_score

def _calc_foot_pos_clock_reward(self):
    # constraints of foot height based on clock
    desired_max_foot_height = 0.05
    l_foot_pos = self._client.get_object_xpos_by_name('lf_force', 'OBJ_SITE')[2]
    r_foot_pos = self._client.get_object_xpos_by_name('rf_force', 'OBJ_SITE')[2]
    normed_left_pos = min(np.linalg.norm(l_foot_pos), desired_max_foot_height) / desired_max_foot_height
    normed_right_pos = min(np.linalg.norm(r_foot_pos), desired_max_foot_height) / desired_max_foot_height

    left_pos_clock = self.left_clock[1](self._phase)
    right_pos_clock = self.right_clock[1](self._phase)

    left_pos_score = np.tan(np.pi/4 * left_pos_clock * normed_left_pos)
    right_pos_score = np.tan(np.pi/4 * right_pos_clock * normed_right_pos)

    foot_pos_score = left_pos_score + right_pos_score
    return foot_pos_score

def _calc_body_orient_reward(self, body_name, quat_ref=[1, 0, 0, 0]):
    # body orientation reward
    body_quat = self._client.get_object_xquat_by_name(body_name, "OBJ_BODY")
    target_quat = np.array(quat_ref)
    error = 10 * (1 - np.inner(target_quat, body_quat) ** 2)
    return np.exp(-error)

def _calc_joint_vel_reward(self, enabled, cutoff=0.5):
    # joint velocity reward
    motor_speeds = self._client.get_motor_velocities()
    motor_limits = self._client.get_motor_speed_limits()
    motor_speeds = [motor_speeds[i] for i in enabled]
    motor_limits = [motor_limits[i] for i in enabled]
    error = 5e-6*sum([np.square(q)
                      for q, qmax in zip(motor_speeds, motor_limits)
                      if np.abs(q)>np.abs(cutoff*qmax)])
    return np.exp(-error)


def _calc_joint_acc_reward(self):
    # joint accelaration reward
    joint_acc_cost = np.sum(np.square(self._client.get_qacc()[-self._num_joints:]))
    return self.wp.joint_acc_weight*joint_acc_cost

def _calc_ang_vel_reward(self):
    # angular vel reward
    ang_vel = self._client.get_qvel()[3:6]
    ang_vel_cost = np.square(np.linalg.norm(ang_vel))
    return self.wp.ang_vel_weight*ang_vel_cost

def _calc_impact_reward(self):
    # contact reward
    ncon = len(self._client.get_rfoot_floor_contacts()) + \
           len(self._client.get_lfoot_floor_contactts())
    if ncon==0:
        return 0
    quad_impact_cost = np.sum(np.square(self._client.get_body_ext_force()))/ncon
    return self.wp.impact_weight*quad_impact_cost

def _calc_zmp_reward(self):
    # zmp reward
    self.current_zmp = estimate_zmp(self)
    if np.linalg.norm(self.current_zmp - self._prev_zmp) > 1:
        self.current_zmp = self._prev_zmp
    zmp_cost = np.square(np.linalg.norm(self.current_zmp - self.desired_zmp))
    self._prev_zmp = self.current_zmp
    return self.wp.zmp_weight*zmp_cost

def _calc_foot_contact_reward(self):
    right_contacts = self._client.get_rfoot_floor_collisions()
    left_contacts = self._client.get_lfoot_floor_collisions()

    radius_thresh = 0.3
    f_base = self._client.get_qpos()[0:2]
    c_dist_r = [(np.linalg.norm(c.pos[0:2] - f_base)) for _, c in right_contacts]
    c_dist_l = [(np.linalg.norm(c.pos[0:2] - f_base)) for _, c in left_contacts]
    d = sum([r for r in c_dist_r if r > radius_thresh] +
            [r for r in c_dist_l if r > radius_thresh])
    return self.wp.foot_contact_weight*d

def _calc_gait_reward(self):
    if self._period<=0:
        raise Exception("Cycle period should be greater than zero.")

    # get foot-ground contact force
    rfoot_grf = self._client.get_rfoot_grf()
    lfoot_grf = self._client.get_lfoot_grf()

    # get foot speed
    rfoot_speed = self._client.get_rfoot_body_speed()
    lfoot_speed = self._client.get_lfoot_body_speed()

    # get foot position
    rfoot_pos = self._client.get_rfoot_body_pos()
    lfoot_pos = self._client.get_lfoot_body_pos()
    swing_height = 0.3
    stance_height = 0.1

    r = 0.5
    if self._phase < r:
        # right foot is in contact
        # left foot is swinging
        cost = (0.01*lfoot_grf)# \
               #+ np.square(lfoot_pos[2]-swing_height)
               #+ (10*np.square(rfoot_pos[2]-stance_height))
    else:
        # left foot is in contact
        # right foot is swinging
        cost = (0.01*rfoot_grf)
               #+ np.square(rfoot_pos[2]-swing_height)
               #+ (10*np.square(lfoot_pos[2]-stance_height))
    return self.wp.gait_weight*cost

def _calc_reference(self):
    if self.ref_poses is None:
        raise Exception("Reference trajectory not provided.")

    # get reference pose
    phase = self._phase
    traj_length = self.traj_len
    indx = int(phase*(traj_length-1))
    reference_pose = self.ref_poses[indx,:]

    # get current pose
    current_pose = np.array(self._client.get_act_joint_positions())

    cost = np.square(np.linalg.norm(reference_pose-current_pose))
    return self.wp.ref_traj_weight*cost

##############################
##############################
# Define utility functions
##############################
##############################

def estimate_zmp(self):
    Gv = 9.80665
    Mg = self._mass * Gv

    com_pos = self._sim.data.subtree_com[1].copy()
    lin_mom = self._sim.data.subtree_linvel[1].copy()*self._mass
    ang_mom = self._sim.data.subtree_angmom[1].copy() + np.cross(com_pos, lin_mom)

    d_lin_mom = (lin_mom - self._prev_lin_mom)/self._control_dt
    d_ang_mom = (ang_mom - self._prev_ang_mom)/self._control_dt

    Fgz = d_lin_mom[2] + Mg

    # check contact with floor
    contacts = [self._sim.data.contact[i] for i in range(self._sim.data.ncon)]
    contact_flag = [(c.geom1==0 or c.geom2==0) for c in contacts]

    if (True in contact_flag) and Fgz > 20:
        zmp_x = (Mg*com_pos[0] - d_ang_mom[1])/Fgz
        zmp_y = (Mg*com_pos[1] + d_ang_mom[0])/Fgz
    else:
        zmp_x = com_pos[0]
        zmp_y = com_pos[1]

    self._prev_lin_mom = lin_mom
    self._prev_ang_mom = ang_mom
    return np.array([zmp_x, zmp_y])

##############################
##############################
# Based on apex
##############################
##############################

def create_phase_reward(swing_duration, stance_duration, strict_relaxer, stance_mode, FREQ=40):

    from scipy.interpolate import PchipInterpolator

    # NOTE: these times are being converted from time in seconds to phaselength
    right_swing = np.array([0.0, swing_duration]) * FREQ
    first_dblstance = np.array([swing_duration, swing_duration + stance_duration]) * FREQ
    left_swing = np.array([swing_duration + stance_duration, 2 * swing_duration + stance_duration]) * FREQ
    second_dblstance = np.array([2 * swing_duration + stance_duration, 2 * (swing_duration + stance_duration)]) * FREQ

    r_frc_phase_points = np.zeros((2, 8))
    r_vel_phase_points = np.zeros((2, 8))
    l_frc_phase_points = np.zeros((2, 8))
    l_vel_phase_points = np.zeros((2, 8))

    right_swing_relax_offset = (right_swing[1] - right_swing[0]) * strict_relaxer
    l_frc_phase_points[0,0] = r_frc_phase_points[0,0] = right_swing[0] + right_swing_relax_offset
    l_frc_phase_points[0,1] = r_frc_phase_points[0,1] = right_swing[1] - right_swing_relax_offset
    l_vel_phase_points[0,0] = r_vel_phase_points[0,0] = right_swing[0] + right_swing_relax_offset
    l_vel_phase_points[0,1] = r_vel_phase_points[0,1] = right_swing[1] - right_swing_relax_offset
    # During right swing we want foot velocities and don't want foot forces
    l_vel_phase_points[1,:2] = r_frc_phase_points[1,:2] = np.negative(np.ones(2))  # penalize l vel and r force
    l_frc_phase_points[1,:2] = r_vel_phase_points[1,:2] = np.ones(2)  # incentivize l force and r vel

    dbl_stance_relax_offset = (first_dblstance[1] - first_dblstance[0]) * strict_relaxer
    l_frc_phase_points[0,2] = r_frc_phase_points[0,2] = first_dblstance[0] + dbl_stance_relax_offset
    l_frc_phase_points[0,3] = r_frc_phase_points[0,3] = first_dblstance[1] - dbl_stance_relax_offset
    l_vel_phase_points[0,2] = r_vel_phase_points[0,2] = first_dblstance[0] + dbl_stance_relax_offset
    l_vel_phase_points[0,3] = r_vel_phase_points[0,3] = first_dblstance[1] - dbl_stance_relax_offset
    if stance_mode == "aerial":
        # During aerial we want foot velocities and don't want foot forces
        # During grounded walking we want foot forces and don't want velocities
        l_frc_phase_points[1,2:4] = r_frc_phase_points[1,2:4] = np.negative(np.ones(2))  # penalize l and r foot force
        l_vel_phase_points[1,2:4] = r_vel_phase_points[1,2:4] = np.ones(2)  # incentivize l and r foot velocity
    elif stance_mode == "zero":
        l_frc_phase_points[1,2:4] = r_frc_phase_points[1,2:4] = np.zeros(2)
        l_vel_phase_points[1,2:4] = r_vel_phase_points[1,2:4] = np.zeros(2)
    else:
        # During grounded walking we want foot forces and don't want velocities
        l_frc_phase_points[1,2:4] = r_frc_phase_points[1,2:4] = np.ones(2)  # incentivize l and r foot force
        l_vel_phase_points[1,2:4] = r_vel_phase_points[1,2:4] = np.negative(np.ones(2))  # penalize l and r foot velocity

    left_swing_relax_offset = (left_swing[1] - left_swing[0]) * strict_relaxer
    l_frc_phase_points[0,4] = r_frc_phase_points[0,4] = left_swing[0] + left_swing_relax_offset
    l_frc_phase_points[0,5] = r_frc_phase_points[0,5] = left_swing[1] - left_swing_relax_offset
    l_vel_phase_points[0,4] = r_vel_phase_points[0,4] = left_swing[0] + left_swing_relax_offset
    l_vel_phase_points[0,5] = r_vel_phase_points[0,5] = left_swing[1] - left_swing_relax_offset
    # During left swing we want foot forces and don't want foot velocities (from perspective of right foot)
    l_vel_phase_points[1,4:6] = r_frc_phase_points[1,4:6] = np.ones(2)  # incentivize l vel and r force
    l_frc_phase_points[1,4:6] = r_vel_phase_points[1,4:6] = np.negative(np.ones(2))  # penalize l force and r vel

    dbl_stance_relax_offset = (second_dblstance[1] - second_dblstance[0]) * strict_relaxer
    l_frc_phase_points[0,6] = r_frc_phase_points[0,6] = second_dblstance[0] + dbl_stance_relax_offset
    l_frc_phase_points[0,7] = r_frc_phase_points[0,7] = second_dblstance[1] - dbl_stance_relax_offset
    l_vel_phase_points[0,6] = r_vel_phase_points[0,6] = second_dblstance[0] + dbl_stance_relax_offset
    l_vel_phase_points[0,7] = r_vel_phase_points[0,7] = second_dblstance[1] - dbl_stance_relax_offset
    if stance_mode == "aerial":
        # During aerial we want foot velocities and don't want foot forces
        # During grounded walking we want foot forces and don't want velocities
        l_frc_phase_points[1,6:] = r_frc_phase_points[1,6:] = np.negative(np.ones(2))  # penalize l and r foot force
        l_vel_phase_points[1,6:] = r_vel_phase_points[1,6:] = np.ones(2)  # incentivize l and r foot velocity
    elif stance_mode == "zero":
        l_frc_phase_points[1,6:] = r_frc_phase_points[1,6:] = np.zeros(2)
        l_vel_phase_points[1,6:] = r_vel_phase_points[1,6:] = np.zeros(2)
    else:
        # During grounded walking we want foot forces and don't want velocities
        l_frc_phase_points[1,6:] = r_frc_phase_points[1,6:] = np.ones(2)  # incentivize l and r foot force
        l_vel_phase_points[1,6:] = r_vel_phase_points[1,6:] = np.negative(np.ones(2))  # penalize l and r foot velocity

    ## extend the data to three cycles : one before and one after : this ensures continuity

    r_frc_prev_cycle = np.copy(r_frc_phase_points)
    r_vel_prev_cycle = np.copy(r_vel_phase_points)
    l_frc_prev_cycle = np.copy(l_frc_phase_points)
    l_vel_prev_cycle = np.copy(l_vel_phase_points)
    l_frc_prev_cycle[0] = r_frc_prev_cycle[0] = r_frc_phase_points[0] - r_frc_phase_points[0, -1] - dbl_stance_relax_offset
    l_vel_prev_cycle[0] = r_vel_prev_cycle[0] = r_vel_phase_points[0] - r_vel_phase_points[0, -1] - dbl_stance_relax_offset

    r_frc_second_cycle = np.copy(r_frc_phase_points)
    r_vel_second_cycle = np.copy(r_vel_phase_points)
    l_frc_second_cycle = np.copy(l_frc_phase_points)
    l_vel_second_cycle = np.copy(l_vel_phase_points)
    l_frc_second_cycle[0] = r_frc_second_cycle[0] = r_frc_phase_points[0] + r_frc_phase_points[0, -1] + dbl_stance_relax_offset
    l_vel_second_cycle[0] = r_vel_second_cycle[0] = r_vel_phase_points[0] + r_vel_phase_points[0, -1] + dbl_stance_relax_offset

    r_frc_phase_points_repeated = np.hstack((r_frc_prev_cycle, r_frc_phase_points, r_frc_second_cycle))
    r_vel_phase_points_repeated = np.hstack((r_vel_prev_cycle, r_vel_phase_points, r_vel_second_cycle))
    l_frc_phase_points_repeated = np.hstack((l_frc_prev_cycle, l_frc_phase_points, l_frc_second_cycle))
    l_vel_phase_points_repeated = np.hstack((l_vel_prev_cycle, l_vel_phase_points, l_vel_second_cycle))

    ## Create the smoothing function with cubic spline and cutoff at limits -1 and 1
    r_frc_phase_spline = PchipInterpolator(r_frc_phase_points_repeated[0], r_frc_phase_points_repeated[1])
    r_vel_phase_spline = PchipInterpolator(r_vel_phase_points_repeated[0], r_vel_phase_points_repeated[1])
    l_frc_phase_spline = PchipInterpolator(l_frc_phase_points_repeated[0], l_frc_phase_points_repeated[1])
    l_vel_phase_spline = PchipInterpolator(l_vel_phase_points_repeated[0], l_vel_phase_points_repeated[1])

    return [r_frc_phase_spline, r_vel_phase_spline], [l_frc_phase_spline, l_vel_phase_spline]
