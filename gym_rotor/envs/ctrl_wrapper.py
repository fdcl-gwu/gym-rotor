import numpy as np
from numpy import linalg
from numpy.linalg import inv
from math import cos, sin, atan2, sqrt, pi
from scipy.integrate import odeint, solve_ivp

from gym_rotor.envs.quad import QuadEnv

class CtrlWrapper(QuadEnv):

    def __init__(self): 
        super().__init__()


    def reset(self):
        # Reset states & Normalization:
        self.state = np.array(np.zeros(18))
        self.state[6:15] = np.eye(3).reshape(1, 9, order='F')

        # x, position:
        init_x = self.x_max_threshold - 1.0 # minus 1.0m
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
        init_R_error = (self.euler_max_threshold - 30.0) * self.D2R # minus 30deg
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
        R = R_vec.reshape(3, 3, order='F')
        if not self.isRotationMatrix(R):
            ''' https://www.codefull.net/2017/07/orthonormalize-a-rotation-matrix/ '''
            u, s, vh = linalg.svd(R, full_matrices=False)
            R = u @ vh
            R_vec = R.reshape(9, 1, order='F').flatten()

        # De-normalization
        x *= self.x_max_threshold # [m]
        v *= self.v_max_threshold # [m/s]
        W *= self.W_max_threshold # [rad/s]
        state = np.concatenate((x, v, R_vec, W), axis=0)

        # Solve ODEs:
        if self.ode_integrator == "euler": # solve w/ Euler's Method
            # Equations of motion of the quadrotor UAV
            x_dot = v
            v_dot = self.g*self.e3 - self.f*R@self.e3/self.m
            R_vec_dot = (R@self.hat(W)).reshape(9, 1, order='F')
            W_dot = inv(self.J)@(-self.hat(W)@self.J@W + self.M)
            state_dot = np.concatenate([x_dot.flatten(), 
                                        v_dot.flatten(),                                                                          
                                        R_vec_dot.flatten(),
                                        W_dot.flatten()])
            self.state = state + state_dot * self.dt
        elif self.ode_integrator == "solve_ivp": # solve w/ 'solve_ivp' Solver
            # method= 'RK45', 'LSODA', 'BDF', 'LSODA', ...
            sol = solve_ivp(self.EoM, [0, self.dt], state, method='DOP853')
            self.state = sol.y[:,-1]

        # Normalization
        x = np.array([self.state[0], self.state[1], self.state[2]]) / self.x_max_threshold
        v = np.array([self.state[3], self.state[4], self.state[5]]) / self.v_max_threshold
        R_vec = np.array([self.state[6],  self.state[7],  self.state[8],
                          self.state[9],  self.state[10], self.state[11],
                          self.state[12], self.state[13], self.state[14]])
        W = np.array([self.state[15], self.state[16], self.state[17]]) / self.W_max_threshold
        self.state = np.concatenate((x, v, R_vec, W), axis=0)

        # Add noise here
        """
        
        """
        obs = self.state

        return obs
    

    def reward_wrapper(self, obs, action, prev_action):
        x = np.array([obs[0], obs[1], obs[2]]) # [m]
        v = np.array([obs[3], obs[4], obs[5]]) # [m/s]
        R_vec = np.array([obs[6],  obs[7],  obs[8],
                          obs[9],  obs[10], obs[11],
                          obs[12], obs[13], obs[14]])
        R = R_vec.reshape(3, 3, order='F')
        W = np.array([obs[15], obs[16], obs[17]]) # [rad/s]

        C_X = 2.0  # pos coef.
        C_V = 0.15 # vel coef.
        C_W = 0.2  # ang_vel coef.
        C_A = 0.03 # for smooth control

        eX = x - self.xd     # position error
        eV = v - self.xd_dot # velocity error
   
        reward = C_X*max(0, 1.0 - linalg.norm(eX, 2)) \
               - C_V * linalg.norm(eV, 2) \
               - C_W * linalg.norm(W, 2) \
               - C_A * (abs(prev_action - action)).sum()
        # reward = np.interp(reward, [0.0, C_X], [0.0, 1.0]) # normalized into [0,1]
        reward *= 0.1 # rescaled by a factor of 0.1
        
        return reward


    def done_wrapper(self, obs):
        x = np.array([obs[0], obs[1], obs[2]]) # [m]
        v = np.array([obs[3], obs[4], obs[5]]) # [m/s]
        R_vec = np.array([obs[6],  obs[7],  obs[8],
                          obs[9],  obs[10], obs[11],
                          obs[12], obs[13], obs[14]])
        R = R_vec.reshape(3, 3, order='F')
        W = np.array([obs[15], obs[16], obs[17]]) # [rad/s]

        # Convert rotation matrix to Euler angles:
        eulerAngles = self.rotationMatrixToEulerAngles(R) * self.R2D
        
        done = False
        done = bool(
               (abs(x) >= 1.0).any() # [m]
            or (abs(v) >= 1.0).any() # [m/s]
            or (abs(W) >= 1.0).any() # [rad/s]
            or abs(eulerAngles[0]) >= self.euler_max_threshold # phi
            or abs(eulerAngles[1]) >= self.euler_max_threshold # theta
        )

        return done