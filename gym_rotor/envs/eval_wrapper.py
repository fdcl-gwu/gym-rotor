import numpy as np
from numpy import linalg
from numpy.linalg import inv
from math import cos, sin, pi

from gym_rotor.envs.ctrl_wrapper import CtrlWrapper
from gym_rotor.envs.equiv_wrapper import EquivWrapper

class EvalWrapperCtrl(CtrlWrapper):

    def __init__(self, ): 
        super().__init__()

    def reset(self):
        # Reset states & Normalization:
        self.state = np.array(np.zeros(18))
        self.state[6:15] = np.eye(3).reshape(1, 9, order='F')

        # x, position:
        init_x = self.x_max_threshold - 1.0 # minus 0.5m
        self.state[0] = np.random.uniform(size=1, low=-init_x, high=init_x) 
        self.state[1] = np.random.uniform(size=1, low=-init_x, high=init_x) 
        self.state[2] = np.random.uniform(size=1, low=-init_x, high=init_x)
        x = np.array([self.state[0], self.state[1], self.state[2]]) # [m]

        # v, velocity:
        init_v_error = 1.0 # initial vel error, [m/s]
        self.state[3] = np.random.uniform(size=1, low=-init_v_error, high=init_v_error) 
        self.state[4] = np.random.uniform(size=1, low=-init_v_error, high=init_v_error) 
        self.state[5] = np.random.uniform(size=1, low=-init_v_error, high=init_v_error)
        v = np.array([self.state[3], self.state[4], self.state[5]]) # [m/s]

        # R, attitude:
        init_R_error = (self.euler_max_threshold - 80.0) * self.D2R # minus 30deg
        phi   = np.random.uniform(size=1, low=-init_R_error, high=init_R_error)
        theta = np.random.uniform(size=1, low=-init_R_error, high=init_R_error)
        psi   = np.random.uniform(size=1, low=-init_R_error, high=init_R_error)
        # NED; https://www.wilselby.com/research/arducopter/modeling/
        self.state[6]  = cos(theta)*cos(psi)
        self.state[7]  = cos(theta)*sin(psi) 
        self.state[8]  = -sin(theta) 
        self.state[9]  = sin(phi)*sin(theta)*cos(psi) - cos(phi)*sin(psi)
        self.state[10] = sin(phi)*sin(theta)*sin(psi) + cos(phi)*cos(psi)
        self.state[11] = sin(phi)*cos(theta)
        self.state[12] = cos(phi)*sin(theta)*cos(psi) + sin(phi)*sin(psi)
        self.state[13] = cos(phi)*sin(theta)*sin(psi) - sin(phi)*cos(psi)
        self.state[14] = cos(phi)*cos(theta)
        
        R_vec = np.array([self.state[6],  self.state[7],  self.state[8],
                          self.state[9],  self.state[10], self.state[11],
                          self.state[12], self.state[13], self.state[14]])
        # Re-orthonormalize:
        R = R_vec.reshape(3, 3, order='F')
        if not self.isRotationMatrix(R):
            ''' https://www.codefull.net/2017/07/orthonormalize-a-rotation-matrix/ '''
            u, s, vh = linalg.svd(R, full_matrices=False)
            R = u @ vh
            R_vec = R.reshape(9, 1, order='F').flatten()

        # W, angular velocity:
        init_W_error = 2*pi # initial ang vel error, [rad/s]
        self.state[15] = np.random.uniform(size=1, low=-init_W_error, high=init_W_error) 
        self.state[16] = np.random.uniform(size=1, low=-init_W_error, high=init_W_error) 
        self.state[17] = np.random.uniform(size=1, low=-init_W_error, high=init_W_error) 
        W = np.array([self.state[15], self.state[16], self.state[17]]) # [rad/s]

        # Normalization
        x /= self.x_max_threshold # [m]
        v /= self.v_max_threshold # [m/s]
        W /= self.W_max_threshold # [rad/s]
        self.state = np.concatenate((x, v, R_vec, W), axis=0)

        # Reset forces & moments:
        self.f  = self.m * self.g
        self.f1 = self.f_each
        self.f2 = self.f_each
        self.f3 = self.f_each
        self.f4 = self.f_each
        self.M  = np.zeros(3)

        return np.array(self.state)


class EvalWrapperEquiv(EquivWrapper):

    def __init__(self, ): 
        super().__init__()

    def reset(self):
        # Reset states & Normalization:
        self.state = np.array(np.zeros(18))
        self.state[6:15] = np.eye(3).reshape(1, 9, order='F')

        # x, position:
        init_x = self.x_max_threshold - 1.0 # minus 0.5m
        self.state[0] = np.random.uniform(size=1, low=-init_x, high=init_x) 
        self.state[1] = np.random.uniform(size=1, low=-init_x, high=init_x) 
        self.state[2] = np.random.uniform(size=1, low=-init_x, high=init_x)
        x = np.array([self.state[0], self.state[1], self.state[2]]) # [m]

        # v, velocity:
        init_v_error = 1.0 # initial vel error, [m/s]
        self.state[3] = np.random.uniform(size=1, low=-init_v_error, high=init_v_error) 
        self.state[4] = np.random.uniform(size=1, low=-init_v_error, high=init_v_error) 
        self.state[5] = np.random.uniform(size=1, low=-init_v_error, high=init_v_error)
        v = np.array([self.state[3], self.state[4], self.state[5]]) # [m/s]

        # R, attitude:
        init_R_error = (self.euler_max_threshold - 80.0) * self.D2R # minus 30deg
        phi   = np.random.uniform(size=1, low=-init_R_error, high=init_R_error)
        theta = np.random.uniform(size=1, low=-init_R_error, high=init_R_error)
        psi   = np.random.uniform(size=1, low=-init_R_error, high=init_R_error)
        # NED; https://www.wilselby.com/research/arducopter/modeling/
        self.state[6]  = cos(theta)*cos(psi)
        self.state[7]  = cos(theta)*sin(psi) 
        self.state[8]  = -sin(theta) 
        self.state[9]  = sin(phi)*sin(theta)*cos(psi) - cos(phi)*sin(psi)
        self.state[10] = sin(phi)*sin(theta)*sin(psi) + cos(phi)*cos(psi)
        self.state[11] = sin(phi)*cos(theta)
        self.state[12] = cos(phi)*sin(theta)*cos(psi) + sin(phi)*sin(psi)
        self.state[13] = cos(phi)*sin(theta)*sin(psi) - sin(phi)*cos(psi)
        self.state[14] = cos(phi)*cos(theta)
        
        R_vec = np.array([self.state[6],  self.state[7],  self.state[8],
                          self.state[9],  self.state[10], self.state[11],
                          self.state[12], self.state[13], self.state[14]])
        # Re-orthonormalize:
        R = R_vec.reshape(3, 3, order='F')
        if not self.isRotationMatrix(R):
            ''' https://www.codefull.net/2017/07/orthonormalize-a-rotation-matrix/ '''
            u, s, vh = linalg.svd(R, full_matrices=False)
            R = u @ vh
            R_vec = R.reshape(9, 1, order='F').flatten()

        # W, angular velocity:
        init_W_error = 2*pi # initial ang vel error, [rad/s]
        self.state[15] = np.random.uniform(size=1, low=-init_W_error, high=init_W_error) 
        self.state[16] = np.random.uniform(size=1, low=-init_W_error, high=init_W_error) 
        self.state[17] = np.random.uniform(size=1, low=-init_W_error, high=init_W_error) 
        W = np.array([self.state[15], self.state[16], self.state[17]]) # [rad/s]

        # Normalization
        x /= self.x_max_threshold # [m]
        v /= self.v_max_threshold # [m/s]
        W /= self.W_max_threshold # [rad/s]
        self.state = np.concatenate((x, v, R_vec, W), axis=0)

        # Reset forces & moments:
        self.f  = self.m * self.g
        self.f1 = self.f_each
        self.f2 = self.f_each
        self.f3 = self.f_each
        self.f4 = self.f_each
        self.M  = np.zeros(3)

        return np.array(self.state)