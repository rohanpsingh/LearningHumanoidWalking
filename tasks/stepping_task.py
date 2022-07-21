import numpy as np
import random
import transforms3d as tf3
from tasks import rewards
from enum import Enum, auto

class WalkModes(Enum):
    STANDING = auto()
    CURVED = auto()
    FORWARD = auto()
    BACKWARD = auto()
    INPLACE = auto()
    LATERAL = auto()

class SteppingTask(object):
    """Bipedal locomotion by stepping on targets."""

    def __init__(self,
                 client=None,
                 dt=0.025,
                 neutral_foot_orient=[],
                 root_body='pelvis',
                 lfoot_body='lfoot',
                 rfoot_body='rfoot',
                 head_body='head',
    ):

        self._client = client
        self._control_dt = dt

        self._mass = self._client.get_robot_mass()

        self._goal_speed_ref = 0
        self._goal_height_ref = []
        self._swing_duration = []
        self._stance_duration = []
        self._total_duration = []

        self._head_body_name = head_body
        self._root_body_name = root_body
        self._lfoot_body_name = lfoot_body
        self._rfoot_body_name = rfoot_body

        # read previously generated footstep plans
        with open('utils/footstep_plans.txt', 'r') as fn:
            lines = [l.strip() for l in fn.readlines()]
        self.plans = []
        sequence = []
        for line in lines:
            if line=='---':
                if len(sequence):
                    self.plans.append(sequence)
                sequence=[]
                continue
            else:
                sequence.append(np.array([float(l) for l in line.split(',')]))

    def step_reward(self):
        target_pos = self.sequence[self.t1][0:3]
        foot_dist_to_target = min([np.linalg.norm(ft-target_pos) for ft in [self.l_foot_pos,
                                                                            self.r_foot_pos]])
        hit_reward = 0
        if self.target_reached:
            hit_reward = np.exp(-foot_dist_to_target/0.25)

        target_mp = (self.sequence[self.t1][0:2] + self.sequence[self.t2][0:2])/2
        root_xy_pos = self._client.get_object_xpos_by_name(self._root_body_name, 'OBJ_BODY')[0:2]
        root_dist_to_target = np.linalg.norm(root_xy_pos-target_mp)
        progress_reward = np.exp(-root_dist_to_target/2)
        return (0.8*hit_reward + 0.2*progress_reward)

    def calc_reward(self, prev_torque, prev_action, action):
        orient = tf3.euler.euler2quat(0, 0, self.sequence[self.t1][3])
        r_frc = self.right_clock[0]
        l_frc = self.left_clock[0]
        r_vel = self.right_clock[1]
        l_vel = self.left_clock[1]
        if self.mode == WalkModes.STANDING:
            r_frc = (lambda _:1)
            l_frc = (lambda _:1)
            r_vel = (lambda _:-1)
            l_vel = (lambda _:-1)
        head_pos = self._client.get_object_xpos_by_name(self._head_body_name, 'OBJ_BODY')[0:2]
        root_pos = self._client.get_object_xpos_by_name(self._root_body_name, 'OBJ_BODY')[0:2]
        reward = dict(foot_frc_score=0.150 * rewards._calc_foot_frc_clock_reward(self, l_frc, r_frc),
                      foot_vel_score=0.150 * rewards._calc_foot_vel_clock_reward(self, l_vel, r_vel),
                      orient_cost=0.050 * rewards._calc_body_orient_reward(self,
                                                                           self._root_body_name,
                                                                           quat_ref=orient),
                      height_error=0.050 * rewards._calc_height_reward(self),
                      #torque_penalty=0.050 * rewards._calc_torque_reward(self, prev_torque),
                      #action_penalty=0.050 * rewards._calc_action_reward(self, prev_action),
                      step_reward=0.450 * self.step_reward(),
                      upper_body_reward=0.050 * np.exp(-10*np.square(np.linalg.norm(head_pos-root_pos)))
        )
        return reward

    def transform_sequence(self, sequence):
        lfoot_pos = self._client.get_lfoot_body_pos()
        rfoot_pos = self._client.get_rfoot_body_pos()
        root_yaw = tf3.euler.quat2euler(self._client.get_object_xquat_by_name(self._root_body_name, 'OBJ_BODY'))[2]
        mid_pt = (lfoot_pos + rfoot_pos)/2
        sequence_rel = []
        for x, y, z, theta in sequence:
            x_ = mid_pt[0] + x*np.cos(root_yaw) - y*np.sin(root_yaw)
            y_ = mid_pt[1] + x*np.sin(root_yaw) + y*np.cos(root_yaw)
            theta_ = root_yaw + theta
            step = np.array([x_, y_, z, theta_])
            sequence_rel.append(step)
        return sequence_rel

    def generate_step_sequence(self, **kwargs):
        step_size, step_gap, step_height, num_steps, curved, lateral = kwargs.values()
        if curved:
            # set 0 height for curved sequences
            plan = random.choice(self.plans)
            sequence = [[s[0], s[1], 0, s[2]] for s in plan]
            return np.array(sequence)

        if lateral:
            sequence = []
            y = 0
            c = np.random.choice([-1, 1])
            for i in range(1, num_steps):
                if i%2:
                    y += step_size
                else:
                    y -= (2/3)*step_size
                step = np.array([0, c*y, 0, 0])
                sequence.append(step)
            return sequence

        sequence = []
        if self._phase==(0.5*self._period):
            first_step = np.array([0, -1*np.random.uniform(0.095, 0.105), 0, 0])
            y = -step_gap
        else:
            first_step = np.array([0, 1*np.random.uniform(0.095, 0.105), 0, 0])
            y = step_gap
        sequence.append(first_step)
        x, z = 0, 0
        c = np.random.randint(2, 4)
        for i in range(1, num_steps):
            x += step_size
            y *= -1
            if i > c: # let height of first few steps equal to 0
                z += step_height
            step = np.array([x, y, z, 0])
            sequence.append(step)
        return sequence

    def update_goal_steps(self):
        self._goal_steps_x[:] = np.zeros(2)
        self._goal_steps_y[:] = np.zeros(2)
        self._goal_steps_z[:] = np.zeros(2)
        self._goal_steps_theta[:] = np.zeros(2)
        root_pos = self._client.get_object_xpos_by_name(self._root_body_name, 'OBJ_BODY')
        root_quat = self._client.get_object_xquat_by_name(self._root_body_name, 'OBJ_BODY')
        for idx, t in enumerate([self.t1, self.t2]):
            ref_frame = tf3.affines.compose(root_pos, tf3.quaternions.quat2mat(root_quat), np.ones(3))
            abs_goal_pos = self.sequence[t][0:3]
            abs_goal_rot = tf3.euler.euler2mat(0, 0, self.sequence[t][3])
            absolute_target = tf3.affines.compose(abs_goal_pos, abs_goal_rot, np.ones(3))
            relative_target = np.linalg.inv(ref_frame).dot(absolute_target)
            if self.mode != WalkModes.STANDING:
                self._goal_steps_x[idx] = relative_target[0, 3]
                self._goal_steps_y[idx] = relative_target[1, 3]
                self._goal_steps_z[idx] = relative_target[2, 3]
                self._goal_steps_theta[idx] = tf3.euler.mat2euler(relative_target[:3, :3])[2]
        return

    def update_target_steps(self):
        assert len(self.sequence)>0
        self.t1 = self.t2
        self.t2+=1
        if self.t2==len(self.sequence):
            self.t2 = len(self.sequence)-1
        return

    def step(self):
        # increment phase
        self._phase+=1
        if self._phase>=self._period:
            self._phase=0

        self.l_foot_quat = self._client.get_object_xquat_by_name('lf_force', 'OBJ_SITE')
        self.r_foot_quat = self._client.get_object_xquat_by_name('rf_force', 'OBJ_SITE')
        self.l_foot_pos = self._client.get_object_xpos_by_name('lf_force', 'OBJ_SITE')
        self.r_foot_pos = self._client.get_object_xpos_by_name('rf_force', 'OBJ_SITE')
        self.l_foot_vel = self._client.get_lfoot_body_vel()[0]
        self.r_foot_vel = self._client.get_rfoot_body_vel()[0]
        self.l_foot_frc = self._client.get_lfoot_grf()
        self.r_foot_frc = self._client.get_rfoot_grf()

        # check if target reached
        target_pos = self.sequence[self.t1][0:3]
        foot_dist_to_target = min([np.linalg.norm(ft-target_pos) for ft in [self.l_foot_pos,
                                                                            self.r_foot_pos]])


        lfoot_in_target = (np.linalg.norm(self.l_foot_pos-target_pos) < self.target_radius)
        rfoot_in_target = (np.linalg.norm(self.r_foot_pos-target_pos) < self.target_radius)
        if lfoot_in_target or rfoot_in_target:
            self.target_reached = True
            self.target_reached_frames+=1
        else:
            self.target_reached = False
            self.target_reached_frames=0

        # update target steps if needed
        if self.target_reached and (self.target_reached_frames>=self.delay_frames):
            self.update_target_steps()
            self.target_reached = False
            self.target_reached_frames = 0

        # update goal
        self.update_goal_steps()
        return

    def substep(self):
        pass

    def done(self):
        contact_flag = self._client.check_bad_collisions()

        qpos = self._client.get_object_xpos_by_name(self._root_body_name, 'OBJ_BODY')
        foot_pos = min([c[2] for c in (self.l_foot_pos, self.r_foot_pos)])
        root_rel_height = qpos[2] - foot_pos
        terminate_conditions = {"qpos[2]_ll":(root_rel_height < 0.6),
                                "contact_flag":contact_flag,
        }

        done = True in terminate_conditions.values()
        return done

    def reset(self, iter_count=0):
        # training iteration
        self.iteration_count = iter_count

        # for steps
        self._goal_steps_x = [0, 0]
        self._goal_steps_y = [0, 0]
        self._goal_steps_z = [0, 0]
        self._goal_steps_theta = [0, 0]

        self.target_radius = 0.20
        self.delay_frames = int(np.floor(self._swing_duration/self._control_dt))
        self.target_reached = False
        self.target_reached_frames = 0
        self.t1 = 0
        self.t2 = 0

        self.right_clock, self.left_clock = rewards.create_phase_reward(self._swing_duration,
                                                                        self._stance_duration,
                                                                        0.1,
                                                                        "grounded",
                                                                        1/self._control_dt)

        # number of control steps in one full cycle
        # (one full cycle includes left swing + right swing)
        self._period = np.floor(2*self._total_duration*(1/self._control_dt))
        # randomize phase during initialization
        self._phase = int(np.random.choice([0, self._period/2]))

        ## GENERATE STEP SEQUENCE
        # select a walking 'mode'
        self.mode = np.random.choice(
            [WalkModes.CURVED, WalkModes.STANDING, WalkModes.BACKWARD, WalkModes.LATERAL, WalkModes.FORWARD],
            p=[0.15, 0.05, 0.2, 0.3, 0.3])

        d = {'step_size':0.3, 'step_gap':0.15, 'step_height':0, 'num_steps':20, 'curved':False, 'lateral':False}
        # generate sequence according to mode
        if self.mode == WalkModes.CURVED:
            d['curved'] = True
        elif self.mode == WalkModes.STANDING:
            d['num_steps'] = 1
        elif self.mode == WalkModes.BACKWARD:
            d['step_size'] = -0.1
        elif self.mode == WalkModes.INPLACE:
            ss = np.random.uniform(-0.05, 0.05)
            d['step_size']=ss
        elif self.mode == WalkModes.LATERAL:
            d['step_size'] = 0.4
            d['lateral'] = True
        elif self.mode == WalkModes.FORWARD:
            h = np.clip((self.iteration_count-3000)/8000, 0, 1)*0.1
            d['step_height']=np.random.choice([-h, h])
        else:
            raise Exception("Invalid WalkModes")
        sequence = self.generate_step_sequence(**d)
        self.sequence = self.transform_sequence(sequence)
        self.update_target_steps()

        ## CREATE TERRAIN USING GEOMS
        nboxes = 20
        boxes = ["box"+repr(i+1).zfill(2) for i in range(nboxes)]
        sequence = [np.array([0, 0, -1, 0]) for i in range(nboxes)]
        sequence[:len(self.sequence)] = self.sequence
        for box, step in zip(boxes, sequence):
            box_h = self._client.model.geom(box).size[2]
            self._client.model.geom(box).pos[:] = step[0:3] - np.array([0, 0, box_h])
            self._client.model.geom(box).quat[:] = tf3.euler.euler2quat(0, 0, step[3])
            self._client.model.geom(box).size[:] = np.array([0.15, 1, box_h])
            self._client.model.geom(box).rgba[:] = np.array([0.8, 0.8, 0.8, 1])

        self._client.model.geom('floor').pos[:] = np.array([0, 0, 0])
        if self.mode == WalkModes.FORWARD:
            self._client.model.geom('floor').pos[:] = np.array([0, 0, -100])
