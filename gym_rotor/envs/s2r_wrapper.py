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
        self.f1 = self.f_hover
        self.f2 = self.f_hover
        self.f3 = self.f_hover
        self.f4 = self.f_hover
        self.M  = np.zeros(3)

        return np.array(self.state)


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
        


    def set_random_parameters(self, env_type='train'):
        if env_type == 'train':
            # Quadrotor parameters:
            self.m  = uniform(size=1, low=1.7, high=1.9).max() # 1.85; mass of quad, [kg]
            self.d  = uniform(size=1, low=0.22, high=0.24).max() # 0.23; arm length, [m]
            self.J1 = uniform(size=1, low=0.015, high=0.025).max() # 0.02
            self.J2 = self.J1 
            self.J3 = uniform(size=1, low=0.035, high=0.045).max() # 0.04
            self.J  = np.diag([self.J1, self.J2, self.J3]) # [0.02,0.02,0.04]; inertia matrix of quad, [kg m2]
            self.c_tf = uniform(size=1, low=0.01, high=0.015).max() # 0.0135; torque-to-thrust coefficients
            self.c_tw = uniform(size=1, low=1.5, high=2.0).max() # 1.8; thrust-to-weight coefficients
            
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
            
        elif env_type == 'eval':
            # Quadrotor parameters:
            self.m = 1.85 # mass of quad, [kg]
            self.d = 0.23 # arm length, [m]
            self.J = np.diag([0.02, 0.02, 0.04]) # inertia matrix of quad, [kg m2]
            self.c_tf = 0.0135 # torque-to-thrust coefficients
            self.c_tw = 1.8 # thrust-to-weight coefficients