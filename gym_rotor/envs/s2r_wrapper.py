import numpy as np
from numpy import linalg
from numpy.linalg import inv
from numpy.random import uniform 
from math import cos, sin, pi
from scipy.integrate import odeint, solve_ivp
from transforms3d.euler import euler2mat, mat2euler

from gym_rotor.envs.quad_utils import *
from gym_rotor.envs.quad import QuadEnv


class Sim2RealWrapper(QuadEnv):

    def __init__(self): 
        super().__init__()
        self.delayed_action = np.zeros(4)

    def reset(self, env_type='train'):
        # Reset states & Normalization:
        state = np.array(np.zeros(18))
        state[6:15] = np.eye(3).reshape(1, 9, order='F')

        # Initial state error:
        self.sample_init_error(env_type)

        # Domain randomization:
        self.set_random_parameters(env_type)

        # x, position:
        state[0] = uniform(size=1, low=-self.init_x_error, high=self.init_x_error) 
        state[1] = uniform(size=1, low=-self.init_x_error, high=self.init_x_error) 
        state[2] = uniform(size=1, low=-self.init_x_error, high=self.init_x_error)

        # v, velocity:
        state[3] = uniform(size=1, low=-self.init_v_error, high=self.init_v_error) 
        state[4] = uniform(size=1, low=-self.init_v_error, high=self.init_v_error) 
        state[5] = uniform(size=1, low=-self.init_v_error, high=self.init_v_error)

        # W, angular velocity:
        state[15] = uniform(size=1, low=-self.init_W_error, high=self.init_W_error) 
        state[16] = uniform(size=1, low=-self.init_W_error, high=self.init_W_error) 
        state[17] = uniform(size=1, low=-self.init_W_error, high=self.init_W_error) 

        # R, attitude:
        pitch = uniform(size=1, low=-self.init_R_error, high=self.init_R_error)
        roll  = uniform(size=1, low=-self.init_R_error, high=self.init_R_error)
        yaw   = uniform(size=1, low=-pi, high=pi) 
        R = euler2mat(roll, pitch, yaw) 
        # Re-orthonormalize:
        if not isRotationMatrix(R):
            ''' https://www.codefull.net/2017/07/orthonormalize-a-rotation-matrix/ '''
            u, s, vh = linalg.svd(R, full_matrices=False)
            R = u @ vh
        R_vec = R.reshape(9, 1, order='F').flatten()
        # self.b1d = get_current_b1(R) # desired heading direction     

        # Normalization: [max, min] -> [-1, 1]
        x_norm, v_norm, _, W_norm = state_normalization(state, self.x_lim, self.v_lim, self.W_lim)
        self.state = np.concatenate((x_norm, v_norm, R_vec, W_norm), axis=0)
        #self.b1d = rot_b1d(x_norm)   

        # Reset forces & moments:
        self.f  = self.m * self.g
        self.f1 = self.hover_force
        self.f2 = self.hover_force
        self.f3 = self.hover_force
        self.f4 = self.hover_force
        self.M  = np.zeros(3)

        return np.array(self.state)

    """
    def action_wrapper(self, action):
        self.total_iter += 1
        if self.total_iter % self.action_update_freq == 0:

            # Linear scale, [-1, 1] -> [min_act, max_act] 
            self.delayed_action = (
                self.scale_act * action + self.avrg_act
                ).clip(self.min_force, self.max_force)

            # Saturated thrust of each motor:
            self.f1 = self.delayed_action[0]
            self.f2 = self.delayed_action[1]
            self.f3 = self.delayed_action[2]
            self.f4 = self.delayed_action[3]

            # Convert each forces to force-moment:
            self.fM = self.forces_to_fM @ self.delayed_action
            self.f = self.fM[0]   # [N]
            self.M = self.fM[1:4] # [Nm]  


        return self.delayed_action
    """

    def set_random_parameters(self, env_type='train'):
        # Nominal quadrotor parameters:
        self.m = 1.85 # mass of quad, [kg]
        self.d = 0.23 # arm length, [m]
        J1, J2, J3 = 0.02, 0.02, 0.04
        self.J = np.diag([J1, J2, J3]) # inertia matrix of quad, [kg m2]
        self.c_tf = 0.0135 # torque-to-thrust coefficients
        self.c_tw = 1.8 # thrust-to-weight coefficients

        if env_type == 'train':
            uncertainty_range = 0.1 # *100 = [%]
            # Quadrotor parameters:
            m_range = self.m * uncertainty_range
            d_range = self.d * uncertainty_range
            J1_range = J1 * uncertainty_range
            J3_range = J3 * uncertainty_range
            c_tf_range = self.c_tf * uncertainty_range
            c_tw_range = self.c_tw * uncertainty_range

            self.m = uniform(low=(self.m - m_range), high=(self.m + m_range)) # [kg]
            self.d = uniform(low=(self.d - d_range), high=(self.d + d_range)) # [m]
            J1 = uniform(low=(J1 - J1_range), high=(J1 + J1_range))
            J2 = J1 
            J3 = uniform(low=(J3 - J3_range), high=(J3 + J3_range))
            self.J  = np.diag([J1, J2, J3]) # [kg m2]
            self.c_tf = uniform(low=(self.c_tf - c_tf_range), high=(self.c_tf + c_tf_range))
            self.c_tw = uniform(low=(self.c_tw - c_tw_range), high=(self.c_tw + c_tw_range))
            
            """
            # Frequency of “Delayed” action updates
            '''
            Example) If `self.freq = 300 # frequency [Hz]`,
            self.action_update_freq = 1    -> 300 Hz
            self.action_update_freq = 1.5  -> 200 Hz
            self.action_update_freq = 2    -> 150 Hz
            self.action_update_freq = 3    -> 100 Hz
            '''
            self.action_update_freq = 1
            # self.action_update_freq = uniform(size=1, low=1, high=2).max() # 100 Hz to 200 Hz
            # Motor and Sensor noise: thrust_noise_ratio, sigma, cutoff_freq
            """

        # Force and Moment:
        self.f = self.m * self.g # magnitude of total thrust to overcome  
                                    # gravity and mass (No air resistance), [N]
        self.hover_force = self.m * self.g / 4.0 # thrust magnitude of each motor, [N]
        self.max_force = self.c_tw * self.hover_force # maximum thrust of each motor, [N]
        self.fM = np.zeros((4, 1)) # Force-moment vector
        self.forces_to_fM = np.array([
            [1.0, 1.0, 1.0, 1.0],
            [0.0, -self.d, 0.0, self.d],
            [self.d, 0.0, -self.d, 0.0],
            [-self.c_tf, self.c_tf, -self.c_tf, self.c_tf]
        ]) # Conversion matrix of forces to force-moment 
        self.fM_to_forces = np.linalg.inv(self.forces_to_fM)
        self.avrg_act = (self.min_force+self.max_force)/2.0 
        self.scale_act = self.max_force-self.avrg_act # actor scaling