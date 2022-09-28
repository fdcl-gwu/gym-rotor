import numpy as np
from numpy import linalg
from math import cos, sin, atan2, sqrt, pi

from gym_rotor.envs.quad import QuadEnv

class CtrlWrapper(QuadEnv):

    def __init__(self): 
        super().__init__()


    def reset(self):
        # Reset states:
        self.state = np.array(np.zeros(18))
        self.state[6:15] = np.eye(3).reshape(1, 9, order='F')
        init_error = 0.1 # initial error

        # x, position:
        init_x = self.x_max_threshold - 0.5 # [m]
        self.state[0] = np.random.uniform(size = 1, low = -init_x, high = init_x) 
        self.state[1] = np.random.uniform(size = 1, low = -init_x, high = init_x)  
        self.state[2] = np.random.uniform(size = 1, low = -init_x, high = init_x) 
        
        # v, velocity:
        self.state[3] = np.random.uniform(size = 1, low = -init_error, high = init_error) 
        self.state[4] = np.random.uniform(size = 1, low = -init_error, high = init_error) 
        self.state[5] = np.random.uniform(size = 1, low = -init_error, high = init_error)

        # R, attitude:
        # https://cse.sc.edu/~yiannisr/774/2014/Lectures/15-Quadrotors.pdf
        phi   = np.random.uniform(size = 1, low = -init_error, high = init_error)
        theta = np.random.uniform(size = 1, low = -init_error, high = init_error)
        psi   = np.random.uniform(size = 1, low = -init_error, high = init_error)
        self.state[6]  = cos(psi)*cos(theta)
        self.state[7]  = sin(psi)*cos(theta) 
        self.state[8]  = -sin(theta)  
        self.state[9]  = cos(psi)*sin(theta)*sin(phi) - sin(psi)*cos(phi) 
        self.state[10] = sin(psi)*sin(theta)*sin(phi) + cos(psi)*cos(phi)
        self.state[11] = cos(theta)*sin(phi) 
        self.state[12] = cos(psi)*sin(theta)*cos(phi) + sin(psi)*sin(phi)
        self.state[13] = sin(psi)*sin(theta)*cos(phi) - cos(psi)*sin(phi)
        self.state[14] = cos(theta)*cos(phi)

        # W, angular velocity:
        self.state[15] = np.random.uniform(size = 1, low = -init_error, high = init_error) 
        self.state[16] = np.random.uniform(size = 1, low = -init_error, high = init_error) 
        self.state[17] = np.random.uniform(size = 1, low = -init_error, high = init_error) 

        # Reset forces & moments:
        self.f  = self.m * self.g
        self.f1 = self.f_each
        self.f2 = self.f_each
        self.f3 = self.f_each
        self.f4 = self.f_each
        self.M  = np.zeros(3)

        return np.array(self.state)


    def action_wrapper(self, action):

        action = np.clip(action, self.min_force, self.max_force) # f1, f2, f3, f4

        return action


    def observation_wrapper(self, state):

        x = np.array([state[0], state[1], state[2]]) # [m]
        v = np.array([state[3], state[4], state[5]]) # [m/s]
        R_vec = np.array([state[6],  state[7],  state[8],
                          state[9],  state[10], state[11],
                          state[12], state[13], state[14]])
        W = np.array([state[15], state[16], state[17]]) # [rad/s]

        # Re-orthonormalize:
        ''' https://www.codefull.net/2017/07/orthonormalize-a-rotation-matrix/ '''
        R = R_vec.reshape(3, 3, order='F')
        u, s, vh = linalg.svd(R, full_matrices=False)
        R = u @ vh
        R_vec = R.reshape(9, 1, order='F').flatten()

        # Normalization
        x /= self.x_max_threshold # [m]
        v /= self.v_max_threshold # [m/s]
        W /= self.W_max_threshold # [rad/s]

        obs = np.concatenate((x, v, R_vec, W), axis=0)

        return obs
    

    def reward_wrapper(self, obs, action, prev_action):

        _x = np.array([obs[0], obs[1], obs[2]]) # [m]
        _v = np.array([obs[3], obs[4], obs[5]]) # [m/s]
        _R_vec = np.array([obs[6],  obs[7],  obs[8],
                           obs[9],  obs[10], obs[11],
                           obs[12], obs[13], obs[14]])
        _R = _R_vec.reshape(3, 3, order='F')
        _W = np.array([obs[15], obs[16], obs[17]]) # [rad/s]

        C_X = 2.0  # pos coef.
        C_V = 0.15 # vel coef.
        C_W = 0.2  # ang_vel coef.

        eX = _x - self.xd     # position error
        eV = _v - self.xd_dot # velocity error
                    
        reward = C_X*max(0, 1.0 - linalg.norm(eX, 2)) \
                 - C_V * linalg.norm(eV, 2) - C_W * linalg.norm(_W, 2)

        C_A = 0.03 # for smooth control
        reward -= C_A * (abs(prev_action - action)).sum()
        reward = np.interp(reward, [0.0, 2.0], [0.0, 1.0]) # normalized into [0,1]
        reward *= 0.1 # rescaled by a factor of 0.1
        
        return reward


    def done_wrapper(self, obs):

        _x = np.array([obs[0], obs[1], obs[2]]) # [m]
        _v = np.array([obs[3], obs[4], obs[5]]) # [m/s]
        _R_vec = np.array([obs[6],  obs[7],  obs[8],
                           obs[9],  obs[10], obs[11],
                           obs[12], obs[13], obs[14]])
        _R = _R_vec.reshape(3, 3, order='F')
        _W = np.array([obs[15], obs[16], obs[17]]) # [rad/s]

        # Convert rotation matrix to Euler angles:
        eulerAngles = self.rotationMatrixToEulerAngles(_R)

        done = False
        done = bool(
               (abs(_x) >= 1.0).any() # [m]
            or (abs(_v) >= 1.0).any() # [m/s]
            or (abs(_W) >= 1.0).any() # [rad/s]
            or abs(eulerAngles[0]) >= self.euler_max_threshold # phi
            or abs(eulerAngles[1]) >= self.euler_max_threshold # theta
        )
        '''
        done = bool(
               (abs(_x) >= self.limits_x).any() # [m]
            # or _x[2] >= 0.0 # crashed
            or (abs(_v) >= self.limits_v).any() # [m/s]
            or (abs(_W) >= self.limits_W).any() # [rad/s]
            or abs(eulerAngles[0]) >= self.euler_max_threshold # phi
            or abs(eulerAngles[1]) >= self.euler_max_threshold # theta
        )
        '''

        return done