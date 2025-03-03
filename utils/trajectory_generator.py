"""
Reference: https://github.com/fdcl-gwu/uav_simulator/blob/main/scripts/trajectory.py
"""
import numpy as np
import datetime

import args_parse
from gym_rotor.envs.quad_utils import *

class TrajectoryGenerator:
    def __init__(self, env):
        # Hyperparameters:
        parser = args_parse.create_parser()
        args = parser.parse_args()

        """----------------------------------------------------------
            self.mode == 0  # manual mode (idle and warm-up)
        -------------------------------------------------------------
            self.mode == 1:  # hovering
        -------------------------------------------------------------
            self.mode == 2:  # take-off
        -------------------------------------------------------------
            self.mode == 3:  # landing
        -------------------------------------------------------------
            self.mode == 4:  # stay (maintain current position)
        -------------------------------------------------------------
            self.mode == 5:  # circle
        -------------------------------------------------------------
            self.mode == 6:  #  eight-shaped curve
        ----------------------------------------------------------"""
        self.mode = 0
        self.is_mode_changed = False
        self.is_landed = False
        self.e1 = np.array([1.,0.,0.])
        self.e2 = np.array([0.,1.,0.])
        self.e3 = np.array([0.,0.,1.])
        self.R2D = 180./np.pi # [rad] to [deg]
        self.D2R = np.pi/180. # [deg] to [rad]

        self.is_realtime = False # if False, it is sim_time
        self.t0 = datetime.datetime.now()
        self.t = 0.0
        self.t_traj = 0.0
        self.dt = env.dt

        self.x_lim, self.v_lim, self.W_lim = env.x_lim, env.v_lim, env.W_lim

        self.x_init, self.v_init = np.zeros(3), np.zeros(3)
        self.R_init, self.W_init = np.identity(3), np.zeros(3)
        self.b1_init = np.zeros(3)
        self.theta_init = 0.0

        self.x, self.v, self.W = np.zeros(3), np.zeros(3), np.zeros(3)
        self.R = np.identity(3)

        self.xd, self.vd, self.Wd = np.zeros(3), np.zeros(3), np.zeros(3)
        self.b1d = np.array([1.,0.,0.]) # desired heading direction

        # Integral terms:
        self.sat_sigma = 1.
        self.eIx = IntegralErrorVec3() # Position integral error
        self.eIb1 = IntegralError() # Attitude integral error
        self.eIx.set_zero() # Set all integrals to zero
        self.eIb1.set_zero()
        self.alpha, self.beta = args.alpha, args.beta # addressing noise or delay
        self.eIx_lim, self.eIb1_lim = env.eIx_lim, env.eIb1_lim

        # Geometric tracking controller:
        self.xd_2dot, self.xd_3dot, self.xd_4dot = np.zeros(3), np.zeros(3), np.zeros(3)
        self.b1d_dot, self.b1d_2dot = np.zeros(3), np.zeros(3)

        self.trajectory_started  = False
        self.trajectory_complete = False
        
        # Manual mode:
        self.manual_mode = False
        self.manual_mode_init = False
        self.init_b1d = True
        self.x_offset = np.zeros(3)
        self.yaw_offset = 0.0

        # Take-off:
        self.takeoff_end_height = -0.5  # [m]
        self.takeoff_velocity = -0.05  # [m/s]

        # Landing:
        self.landing_velocity = 1.  # [m/s]
        self.landing_motor_cutoff_height = -0.25  # [m]

        # Circle:
        self.num_circles = 2
        self.circle_radius = 0.7
        self.circle_linear_v = 0.4
        self.circle_W = 0.4
        
        # Eight-shaped curve:
        self.num_of_eights = 3
        self.eight_A1 = 1.5
        self.eight_A2 = 1.0
        self.eight_T = 9 # the period of the cycle [sec]
        self.eight_w1 = 2*np.pi/self.eight_T # w = 2*pi*t/self.t_origin 
        self.eight_w2 = 4*np.pi/self.eight_T
        self.eight_w_b1d = 0.349066 # [rad/sec] = 20 deg/sec
        
        # Exponential smoothing factor:
        epsilon = 0.01 # determines the degree of smoothing for the exponential term
        self.eight_exp_xy = -np.log(epsilon) / self.eight_T # smooth one period
        # self.eight_exp_xy = -np.log(epsilon) / (2*self.eight_T) # Two times of the period
        self.eight_alt_d = -0.6  # [m] desired altitude
        self.eight_exp_z = -np.log(epsilon) / (3*self.eight_T) #0.1

    
    def get_desired(self, state, mode):
        # Decomposing state vectors:
        self.x, self.v, self.R, self.W = state_decomposition(state)

        # Generate desired traj: 
        if mode == self.mode:
            self.is_mode_changed = False
        else:
            self.is_mode_changed = True
            self.mode = mode
            self.mark_traj_start(state)

        self.calculate_desired()

        return self.xd, self.vd, self.b1d, self.b1d_dot, self.Wd


    def get_desired_geometric_controller(self):

        return self.xd, self.vd, self.xd_2dot, self.xd_3dot, self.xd_4dot, \
               self.b1d, self.b1d_dot, self.b1d_2dot
    
    
    def calculate_desired(self):
        if self.manual_mode:
            self.manual()
            return
        
        if self.mode == 0:  # idle and warm-up
            if self.init_b1d == True:
                self.set_desired_states_to_zero()
                b1d_temp = self.get_current_b1()
                theta_b1d = np.random.uniform(size=1,low=-25*self.D2R, high=25*self.D2R) 
                self.b1d = self.R_e3(theta_b1d) @ b1d_temp 
                # print(theta_b1d, b1d_temp, self.b1d)
                self.init_b1d = False
        elif self.mode == 1:  # hovering
            self.hovering()
        elif self.mode == 2:  # take-off
            self.takeoff()
        elif self.mode == 3:  # land
            self.land()
        elif self.mode == 4:  # stay
            self.stay()
        elif self.mode == 5:  # circle
            self.circle()
        # elif self.mode == 6:  #  eight-shaped curve
        elif self.mode >= 6: 
            self.eight_shaped_curve()

        #############################################################
        # Compute Wd:
        b3 = self.R@self.e3
        b3_dot = self.R @ hat(self.W) @ self.e3

        b1c = self.b1d - np.dot(self.b1d, b3) * b3
        b1c_dot = self.b1d_dot - (np.dot(self.b1d_dot, b3) * b3 + np.dot(self.b1d, b3_dot) * b3 + np.dot(self.b1d, b3) * b3_dot)
        omega_c = np.cross(b1c, b1c_dot)
        omega_c_3 = b3@omega_c
        self.Wd = np.array([0., 0., omega_c_3])
        #############################################################


    def mark_traj_start(self, state_init):
        self.trajectory_started  = False
        self.trajectory_complete = False

        self.manual_mode_init = False
        self.manual_mode = False
        self.is_landed = False

        self.t0 = datetime.datetime.now()
        self.t = 0.0
        self.t_traj = 0.0

        self.x_offset = np.zeros(3)
        self.yaw_offset = 0.
        self.init_b1d = True
        self.update_initial_state(state_init)


    def mark_traj_end(self, switch_to_manual=False):
        self.trajectory_complete = True

        if switch_to_manual:
            self.manual_mode = True


    def update_initial_state(self, state_init):
        self.x_init, self.v_init, self.R_init, self.W_init = state_decomposition(state_init)
        self.b1_init = self.R_init.dot(self.e1)
        self.theta_init = np.arctan2(self.b1_init[1], self.b1_init[0])


    def set_desired_states_to_zero(self):
        self.xd, self.vd, self.Wd = np.zeros(3), np.zeros(3), np.zeros(3)
        self.b1d = np.array([1.,0.,0.]) # desired heading direction

    
    def set_desired_states_to_current(self):
        self.xd = np.copy(self.x)
        self.vd = np.copy(self.v)
        self.b1d = self.get_current_b1()

    
    def get_current_b1(self):
        b1 = self.R.dot(self.e1)
        theta = np.arctan2(b1[1], b1[0])
        return np.array([np.cos(theta), np.sin(theta), 0.])


    def update_current_time(self):
        if self.is_realtime == True:
            t_now = datetime.datetime.now()
            self.t = (t_now - self.t0).total_seconds()
        else:
            self.t = self.t + self.dt


    def manual(self):
        if not self.manual_mode_init:
            self.set_desired_states_to_current()
            # self.update_initial_state()
            b1 = self.R.dot(self.e1)
            self.theta_init = np.arctan2(b1[1], b1[0])

            self.manual_mode_init = True
            self.x_offset = np.zeros(3)
            self.yaw_offset = 0.

            print('Switched to manual mode')
        
        # self.xd = self.x_init + self.x_offset
        self.vd = np.zeros(3)

        theta = self.theta_init + self.yaw_offset
        self.b1d = np.array([np.cos(theta), np.sin(theta), 0.0])


    def hovering(self):
        if not self.trajectory_started:
            self.set_desired_states_to_current()
            self.trajectory_started = True

            self.x_init = np.copy(self.x)  # initial position [x0, y0, z0]
            self.x_goal = np.array([0., 0., 0.])  # goal position [xg, yg, zg]

            # self.t_traj = 3. # time to reach the origin [sec]
            self.t_traj = np.squeeze(np.random.uniform(size=1,low=2., high=5.)) # [sec] time to reach the origin
            self.smooth_term = -np.log(0.001) / self.t_traj

            # w_b1d = 0.1*np.pi# np.squeeze(np.random.uniform(size=1,low=-0.2*np.pi, high=0.2*np.pi)) # [rad/sec] random yaw rate
            # self.w_b1d = np.random.choice([-1, 1])*w_b1d  # multiply by -1 or 1 for better exploration
            self.w_b1d = np.squeeze(np.random.uniform(size=1,low=-0.15*np.pi, high=0.15*np.pi)) # [rad/sec] random yaw rate

        self.update_current_time()
        
        for i in range(3):
            self.xd[i] = (self.x_init[i] - self.x_goal[i]) * np.exp(-self.smooth_term * self.t) + self.x_goal[i]
            self.vd[i] = -(self.x_init[i] - self.x_goal[i]) * self.smooth_term * np.exp(-self.smooth_term * self.t)

        # yaw-axis:
        self.b1d = np.array([np.cos(self.w_b1d * self.t + self.theta_init), np.sin(self.w_b1d * self.t + self.theta_init), 0.])
        self.b1d_dot = np.array([-self.w_b1d * np.sin(self.w_b1d * self.t + self.theta_init), self.w_b1d * np.cos(self.w_b1d * self.t + self.theta_init), 0.])
        # self.b1d_2dot = np.array([-self.w_b1d**2 * np.cos(self.w_b1d * self.t + self.theta_init), -self.w_b1d**2 * np.sin(self.w_b1d * self.t + self.theta_init), 0.])


    def takeoff(self):
        if not self.trajectory_started:
            self.set_desired_states_to_zero()

            # Take-off starts from the current horizontal position:
            self.xd[0] = self.x[0]
            self.xd[1] = self.x[1]
            self.x_init = self.x

            self.t_traj = (self.takeoff_end_height - self.x[2]) / self.takeoff_velocity

            # Set the take-off yaw to the current yaw:
            self.b1d = self.get_current_b1()

            self.trajectory_started = True

        self.update_current_time()

        if self.t < self.t_traj:
            self.xd[2] = self.x_init[2] + self.takeoff_velocity * self.t 
            self.xd_2dot[2] = self.takeoff_velocity
        else:
            if self.waypoint_reached(self.xd, self.x, 0.04):
                self.xd[2] = self.takeoff_end_height
                self.vd[2] = 0.

                if not self.trajectory_complete:
                    print('Takeoff complete\nSwitching to manual mode')
                
                self.mark_traj_end(True)


    def waypoint_reached(self, waypoint, current, radius):
        delta = waypoint - current
        
        if abs(np.linalg.norm(delta) < radius):
            return True
        else:
            return False
        

    def land(self):
        if not self.trajectory_started:
            self.set_desired_states_to_current()
            self.t_traj = (self.landing_motor_cutoff_height - self.x[2]) / self.landing_velocity

            # Set the take-off yaw to the current yaw:
            self.b1d = self.get_current_b1()

            self.trajectory_started = True

        self.update_current_time()

        if self.t < self.t_traj:
            self.xd[2] = self.x_init[2] + self.landing_velocity * self.t
            self.xd_2dot[2] = self.landing_velocity
        else:
            if self.x[2] > self.landing_motor_cutoff_height:
                self.xd[2] = self.landing_motor_cutoff_height
                self.vd[2] = 0.

                if not self.trajectory_complete:
                    print('Landing complete')

                self.mark_traj_end(False)
                self.is_landed = True
            else:
                self.xd[2] = self.landing_motor_cutoff_height
                self.vd[2] = self.landing_velocity

            
    def stay(self):
        if not self.trajectory_started:
            self.set_desired_states_to_current()
            self.trajectory_started = True
        
        self.mark_traj_end(True)


    def circle(self):
        if not self.trajectory_started:
            self.set_desired_states_to_current()
            self.trajectory_started = True

            self.circle_center = np.copy(self.x)
            self.t_traj = self.circle_radius / self.circle_linear_v \
                        + self.num_circles * 2 * np.pi / self.circle_W

        self.update_current_time()

        if self.t < self.circle_radius / self.circle_linear_v:
            self.xd[0] = self.circle_center[0] + self.circle_linear_v * self.t
            self.vd[0] = self.circle_linear_v

        elif self.t < self.t_traj:
            circle_W = self.circle_W
            circle_radius = self.circle_radius

            t = self.t - circle_radius / self.circle_linear_v
            th = circle_W * t

            circle_W2 = circle_W * circle_W
            circle_W3 = circle_W2 * circle_W
            circle_W4 = circle_W3 * circle_W

            # x-axis:
            self.xd[0] = circle_radius * np.cos(th) + self.circle_center[0]
            self.vd[0] = -circle_radius * circle_W * np.sin(th)
            self.xd_2dot[0] = -circle_radius * circle_W2 * np.cos(th)
            self.xd_3dot[0] =  circle_radius * circle_W3 * np.sin(th)
            self.xd_4dot[0] =  circle_radius * circle_W4 * np.cos(th)

            # y-axis:
            self.xd[1] = circle_radius * np.sin(th) + self.circle_center[1]
            self.vd[1] = circle_radius * circle_W * np.cos(th)
            self.xd_2dot[1] = -circle_radius * circle_W2 * np.sin(th)
            self.xd_3dot[1] = -circle_radius * circle_W3 * np.cos(th)
            self.xd_4dot[1] =  circle_radius * circle_W4 * np.sin(th)

            # yaw-axis:
            w_b1d = self.circle_W
            th_b1d = w_b1d * t + np.pi
            self.b1d = np.array([np.cos(th_b1d), np.sin(th_b1d), 0])
            self.b1d_dot = np.array([- w_b1d * np.sin(th_b1d), \
                w_b1d * np.cos(th_b1d), 0.0])
            self.b1d_2dot = np.array([- w_b1d * w_b1d * np.cos(th_b1d),
                w_b1d * w_b1d * np.sin(th_b1d), 0.0])
            '''
            self.b1d = np.array([1.,0.,0.]) 
            self.b1d_dot, self.b1d_2dot = np.zeros(3), np.zeros(3)
            '''
        else:
            self.mark_traj_end(True)


    def eight_shaped_curve(self):
        """
        8 shaped curve(Lissajous Curve): https://www.youtube.com/watch?v=7-v6wruWGME 
        xd_1(t) = A1*cos(w1*t), xd_2(t) = A2*sin(w2*t), where A1/A2 are amplitude and w1/w2 are frequency.
        Set possible positive values of A1, A2, w1 and w2.
        ex, xd_1(t) = 3cos(3t + 1), xd_2(t) = sin(5t)
        """
        if not self.trajectory_started:
            self.set_desired_states_to_current()
            self.trajectory_started = True

            self.eight_shaped_center = np.copy(self.x)
            self.t_traj = self.num_of_eights * self.eight_T

            self.eight_R_xy_2 = self.eight_exp_xy**2
            self.eight_R_xy_3 = self.eight_exp_xy**3
            self.eight_R_xy_4 = self.eight_exp_xy**4
            
            self.eight_w1_2 = self.eight_w1**2
            self.eight_w1_3 = self.eight_w1**3
            self.eight_w1_4 = self.eight_w1**4

            self.eight_w2_2 = self.eight_w2**2
            self.eight_w2_3 = self.eight_w2**3
            self.eight_w2_4 = self.eight_w2**4

            # self.w_b1d = np.random.choice([-1, 1])*self.eight_w_b1d  # multiply by -1 or 1 for better exploration
            self.w_b1d = self.eight_w_b1d

        self.update_current_time()

        if self.t < self.t_traj:
            # Smooth xy trajectory:
            exp_term = 1. - np.exp(-self.eight_exp_xy * self.t)
            d_dt_exp_term = self.eight_exp_xy * np.exp(-self.eight_exp_xy*self.t)
            
            # x2 commands
            self.xd[0] = self.eight_A2 * (sin(self.eight_w2*self.t) * exp_term) + self.eight_shaped_center[0]
            self.vd[0] = self.eight_A2 * ((self.eight_w2 * cos(self.eight_w2*self.t) * exp_term) \
                                          + (sin(self.eight_w2*self.t) * d_dt_exp_term))
            # x1 commands
            self.xd[1] = self.eight_A1 * (cos(self.eight_w1*self.t) - 1.) * exp_term + self.eight_shaped_center[1]
            self.vd[1] = self.eight_A1 * ((self.eight_w1 * -sin(self.eight_w1*self.t) * exp_term) \
                                          + (cos(self.eight_w1*self.t) - 1.) * d_dt_exp_term)

            """
            # x1 commands
            self.xd[0] = self.eight_A1 * (cos(self.eight_w1*self.t) - 1.) * exp_term + self.eight_shaped_center[0]
            self.vd[0] = self.eight_A1 * ((self.eight_w1 * -sin(self.eight_w1*self.t) * exp_term) \
                                          + (cos(self.eight_w1*self.t) - 1.) * d_dt_exp_term)
            # x2 commands
            self.xd[1] = self.eight_A2 * (sin(self.eight_w2*self.t) * exp_term) + self.eight_shaped_center[1]
            self.vd[1] = self.eight_A2 * ((self.eight_w2 * cos(self.eight_w2*self.t) * exp_term) \
                                          + (sin(self.eight_w2*self.t) * d_dt_exp_term))
            -------------------------------------------------------------
            # x1 commands
            self.xd[0] = self.eight_A1 * (cos(self.eight_w1*self.t) - 1.) + self.eight_shaped_center[0]
            self.vd[0] = self.eight_A1 * (self.eight_w1 * -sin(self.eight_w1*self.t))
            # x2 commands
            self.xd[1] = self.eight_A2 * sin(self.eight_w2*self.t) + self.eight_shaped_center[1]
            self.vd[1] = self.eight_A2 * (self.eight_w2 * cos(self.eight_w2*self.t))
            -------------------------------------------------------------
            # x3 commands
            self.xd[2] = self.eight_shaped_center[2]
            self.vd[2] = 0.
            # x3 commands
            # self.xd[2] = self.eight_alt_d * (1. - np.exp(-self.eight_exp_z * self.t)) + self.eight_shaped_center[2]
            # self.vd[2] = self.eight_alt_d * -self.eight_exp_z * -np.exp(-self.eight_exp_z * self.t)
            """            
            # Synchronized  altitude commands
            z_amplitude = (self.eight_shaped_center[2]-self.eight_alt_d)/ 2  # as one full cycle of (1-cos(t))=[0,2] over interval [0,2pi/w1]
            # Adjust the cosine to match altitude with the horizontal axis 
            self.xd[2] = z_amplitude * (1 - cos(self.eight_w1 * self.t)) + self.eight_shaped_center[2]
            ''' e.g., right edge: zd(0) = z(0), and left edge: zd(pi/w1) = zd '''
            self.vd[2] = z_amplitude * self.eight_w1 * sin(self.eight_w1 * self.t)
            
            # yaw-axis:
            w_b1d_term = self.w_b1d * self.t * exp_term + self.theta_init
            d_dt_w_b1d_term = self.w_b1d * (exp_term + self.t * d_dt_exp_term)
            self.b1d = np.array([np.cos(w_b1d_term), np.sin(w_b1d_term), 0.])
            self.b1d_dot = np.array([-np.sin(w_b1d_term) * d_dt_w_b1d_term, np.cos(w_b1d_term) * d_dt_w_b1d_term, 0.])
            """
            self.b1d = np.array([1.,0.,0.]) 
            self.b1d_dot, self.b1d_2dot = np.zeros(3), np.zeros(3)

            self.b1d = np.array([np.cos(self.w_b1d * self.t + self.theta_init), np.sin(self.w_b1d * self.t + self.theta_init), 0.])
            self.b1d_dot = np.array([-self.w_b1d * np.sin(self.w_b1d*self.t + self.theta_init), 
                                     self.w_b1d * np.cos(self.w_b1d*self.t + self.theta_init), 0.])
            """
        else:
            self.mark_traj_end(True)
            
    # Rotation on e3 axis
    def R_e3(self, theta):
        return np.array([[cos(theta), -sin(theta), 0.],
                         [sin(theta),  cos(theta), 0.],
                         [        0.,          0., 1.]])