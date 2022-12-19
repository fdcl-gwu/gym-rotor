import gym
from gym import spaces
from gym.utils import seeding
from gym_rotor.envs.quad_utils import *

import numpy as np
from numpy import linalg
from numpy.linalg import inv
from numpy.random import uniform 
from math import cos, sin, atan2, sqrt, pi
from scipy.integrate import odeint, solve_ivp
from transforms3d.euler import euler2mat, mat2euler

class QuadEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self): 
        # Quadrotor parameters:
        self.m = 1.85 # mass of quad, [kg]
        self.d = 0.23 # arm length, [m]
        self.J = np.diag([0.02, 0.02, 0.04]) # inertia matrix of quad, [kg m2]
        self.c_tf = 0.0135 # torque-to-thrust coefficients
        self.c_tw = 1.8 # thrust-to-weight coefficients
        self.g = 9.81  # standard gravity

        # Force and Moment:
        self.f = self.m * self.g # magnitude of total thrust to overcome  
                                 # gravity and mass (No air resistance), [N]
        self.f_hover = self.m * self.g / 4.0 # thrust magnitude of each motor, [N]
        self.min_force = 0.5 # minimum thrust of each motor, [N]
        self.max_force = self.c_tw * self.f_hover # maximum thrust of each motor, [N]
        self.f1 = self.f_hover # thrust of each 1st motor, [N]
        self.f2 = self.f_hover # thrust of each 2nd motor, [N]
        self.f3 = self.f_hover # thrust of each 3rd motor, [N]
        self.f4 = self.f_hover # thrust of each 4th motor, [N]
        self.M  = np.zeros(3) # magnitude of moment on quadrotor, [Nm]

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

        # Simulation parameters:
        self.total_iter = 0 # num of total timesteps
        self.freq = 200 # frequency [Hz]
        self.dt = 1./self.freq # discrete timestep, t(2) - t(1), [sec]
        self.ode_integrator = "solve_ivp" # or "euler", ODE solvers
        self.R2D = 180/pi # [rad] to [deg]
        self.D2R = pi/180 # [deg] to [rad]
        self.e3 = np.array([0.0, 0.0, 1.0])

        # Coefficients in reward function:
        self.C_X = 0.35 # pos coef.
        self.C_V = 0.15 # vel coef.
        self.C_W = 0.25 # ang_vel coef.
        self.C_R = 0.25 # att coef.
        self.C_Ad = 0.0 # for smooth control
        self.C_Am = 0.005 # 0.03 for smooth control

        # Commands:
        self.xd     = np.array([0.0, 0.0, 0.0]) # desired tracking position command, [m] 
        self.xd_dot = np.array([0.0, 0.0, 0.0]) # [m/s]
        self.b1d    = np.array([1.0, 0.0, 0.0]) # desired heading direction        

        # limits of states:
        self.x_lim = 2.0 # [m]
        self.v_lim = 4.0 # [m/s]
        self.W_lim = 2*pi # [rad/s]
        self.euler_lim = 80 # [deg]
        self.low = np.concatenate([-self.x_lim * np.ones(3),  
                                   -self.v_lim * np.ones(3),
                                   -np.ones(9),
                                   -self.W_lim * np.ones(3)])
        self.high = np.concatenate([self.x_lim * np.ones(3),  
                                    self.v_lim * np.ones(3),
                                    np.ones(9),
                                    self.W_lim * np.ones(3)])

        # Observation space:
        self.observation_space = spaces.Box(
            low=self.low, 
            high=self.high, 
            dtype=np.float64
        )
        # Action space:
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            # low=self.min_force, 
            # high=self.max_force, 
            shape=(4,),
            dtype=np.float64
        ) 

        # Init:
        self.state = None
        self.viewer = None
        self.render_index = 1 
        self.seed()


    def step(self, action_step):
        # De-concatenate `action_step`:
        action = action_step[0:4]
        prev_action = action_step[4:8]

        # Action:
        action = self.action_wrapper(action) # [N] 

        # States: (x[0:3]; v[3:6]; R_vec[6:15]; W[15:18])
        state = (self.state).flatten()
                 
        # Observation:
        obs = self.observation_wrapper(state)

        # Reward function:
        reward = self.reward_wrapper(obs, action, prev_action)

        # Terminal condition:
        done = self.done_wrapper(obs)

        return obs, reward, done, {}


    def reset(self, env_type='train'):
        # Reset states & Normalization:
        self.state = np.array(np.zeros(18))
        self.state[6:15] = np.eye(3).reshape(1, 9, order='F')

        # Initial state error:
        self.sample_init_error(env_type)

        # x, position:
        self.state[0] = uniform(size=1, low=-self.init_x_error, high=self.init_x_error) 
        self.state[1] = uniform(size=1, low=-self.init_x_error, high=self.init_x_error) 
        self.state[2] = uniform(size=1, low=-self.init_x_error, high=self.init_x_error)

        # v, velocity:
        self.state[3] = uniform(size=1, low=-self.init_v_error, high=self.init_v_error) 
        self.state[4] = uniform(size=1, low=-self.init_v_error, high=self.init_v_error) 
        self.state[5] = uniform(size=1, low=-self.init_v_error, high=self.init_v_error)

        # W, angular velocity:
        self.state[15] = uniform(size=1, low=-self.init_W_error, high=self.init_W_error) 
        self.state[16] = uniform(size=1, low=-self.init_W_error, high=self.init_W_error) 
        self.state[17] = uniform(size=1, low=-self.init_W_error, high=self.init_W_error) 

        # R, attitude:
        pitch = uniform(size=1, low=-self.init_R_error, high=self.init_R_error)
        roll  = uniform(size=1, low=-self.init_R_error, high=self.init_R_error)
        yaw   = uniform(size=1, low=-pi, high=pi) 
        R = euler2mat(roll, pitch, yaw) 
        """
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
        R = R_vec.reshape(3, 3, order='F')
        #R = rand_uniform_rot3d()
        """
        # Re-orthonormalize:
        if not isRotationMatrix(R):
            ''' https://www.codefull.net/2017/07/orthonormalize-a-rotation-matrix/ '''
            u, s, vh = linalg.svd(R, full_matrices=False)
            R = u @ vh
        R_vec = R.reshape(9, 1, order='F').flatten()
        # self.b1d = get_current_b1(R) # desired heading direction     

        # Normalization
        x_norm = np.array([self.state[0], self.state[1], self.state[2]]) / self.x_lim # [m]
        v_norm = np.array([self.state[3], self.state[4], self.state[5]]) / self.v_lim # [m/s]
        W_norm = np.array([self.state[15], self.state[16], self.state[17]]) / self.W_lim # [rad/s]
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
        # Linear scale, [-1, 1] -> [min_act, max_act] 
        action = (
            self.scale_act * action + self.avrg_act
            ).clip(self.min_force, self.max_force)

        # Saturated thrust of each motor:
        self.f1 = action[0]
        self.f2 = action[1]
        self.f3 = action[2]
        self.f4 = action[3]

        # Convert each forces to force-moment:
        self.fM = self.forces_to_fM @ action
        self.f = self.fM[0]   # [N]
        self.M = self.fM[1:4] # [Nm]  

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
        if not isRotationMatrix(R):
            ''' https://www.codefull.net/2017/07/orthonormalize-a-rotation-matrix/ '''
            u, s, vh = linalg.svd(R, full_matrices=False)
            R = u @ vh
            R_vec = R.reshape(9, 1, order='F').flatten()

        # De-normalization:
        x *= self.x_lim # [m]
        v *= self.v_lim # [m/s]
        W *= self.W_lim # [rad/s]
        state = np.concatenate((x, v, R_vec, W), axis=0)

        # Solve ODEs:
        if self.ode_integrator == "euler": # solve w/ Euler's Method
            # Equations of motion of the quadrotor UAV
            x_dot = v
            v_dot = self.g*self.e3 - self.f*R@self.e3/self.m
            R_vec_dot = (R@hat(W)).reshape(9, 1, order='F')
            W_dot = inv(self.J)@(-hat(W)@self.J@W + self.M)
            state_dot = np.concatenate([x_dot.flatten(), 
                                        v_dot.flatten(),                                                                          
                                        R_vec_dot.flatten(),
                                        W_dot.flatten()])
            self.state = state + state_dot * self.dt
        elif self.ode_integrator == "solve_ivp": # solve w/ 'solve_ivp' Solver
            # method = 'RK45', 'DOP853', 'BDF', 'LSODA', ...
            sol = solve_ivp(self.EoM, [0, self.dt], state, method='DOP853')
            self.state = sol.y[:,-1]

        # Normalization:
        x = np.array([self.state[0], self.state[1], self.state[2]]) / self.x_lim
        v = np.array([self.state[3], self.state[4], self.state[5]]) / self.v_lim
        R_vec = np.array([self.state[6],  self.state[7],  self.state[8],
                          self.state[9],  self.state[10], self.state[11],
                          self.state[12], self.state[13], self.state[14]])
        W = np.array([self.state[15], self.state[16], self.state[17]]) / self.W_lim

        # Re-orthonormalize:
        R = R_vec.reshape(3, 3, order='F')
        if not isRotationMatrix(R):
            ''' https://www.codefull.net/2017/07/orthonormalize-a-rotation-matrix/ '''
            u, s, vh = linalg.svd(R, full_matrices=False)
            R = u @ vh
            R_vec = R.reshape(9, 1, order='F').flatten()

        self.state = np.concatenate((x, v, R_vec, W), axis=0)

        return self.state
    

    def reward_wrapper(self, obs, action, prev_action):
        x = np.array([obs[0], obs[1], obs[2]]) # [m]
        v = np.array([obs[3], obs[4], obs[5]]) # [m/s]
        R_vec = np.array([obs[6],  obs[7],  obs[8],
                          obs[9],  obs[10], obs[11],
                          obs[12], obs[13], obs[14]])
        R = R_vec.reshape(3, 3, order='F')
        W = np.array([obs[15], obs[16], obs[17]]) # [rad/s]

        eX = x - self.xd     # position error
        eV = v - self.xd_dot # velocity error

        # New reward
        prev_action = (
            self.scale_act * prev_action + self.avrg_act
            ).clip(self.min_force, self.max_force)

        # Reward function
        C_X = self.C_X # pos coef.
        C_V = self.C_V # vel coef.
        C_W = self.C_W # ang_vel coef.
        C_R = self.C_R # att coef.
        C_Ad = self.C_Ad # for smooth control
        C_Am = self.C_Am # for smooth control

        '''
        reward = C_X*max(0, 1.0 - linalg.norm(eX, 2)) \
               - C_V * linalg.norm(eV, 2) \
               - C_W * linalg.norm(W, 2) \
               - C_Ad * (abs(prev_action - action)).sum() \
               - C_Am * (abs(action - self.f_hover)).sum()
               #C_X*max(0, (1.0-abs(eX)[0]) + (1.0-abs(eX)[1]) + (1.0-abs(eX)[2]))
        reward = np.interp(reward, [-C_X, C_X], [0.0, 1.0]) # normalized into [0,1]
        '''

        eR = angle_of_vectors(get_current_b1(R), self.b1d) # [rad], heading error
        eR = np.interp(eR, [0., pi], [0., 1.0]) # normalized into [0,1]

        # To avoid `-log(0) = inf`
        eps = 1e-10
        eX = np.where(abs(eX)<=eps, eX*eps, eX)
        eR = eps if eR<=eps else eR

        reward = C_X*max(0, -(np.log(abs(eX)[0])+np.log(abs(eX)[1])+0.6*np.log(abs(eX)[2]))) \
               + C_R*max(0, -np.log(eR)) \
               - C_V*linalg.norm(eV, 2) \
               - C_W*linalg.norm(W, 2) \
               - C_Ad*(abs(prev_action - action)).sum() \
               - C_Am*(abs(action)).sum() \
               #- C_R*np.sqrt(eR) \
               #+ C_W*max(0, -(np.log(abs(W)[0])+np.log(abs(W)[1])+np.log(abs(W)[2]))) \

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
        if not isRotationMatrix(R): # Re-orthonormalize:
            ''' https://www.codefull.net/2017/07/orthonormalize-a-rotation-matrix/ '''
            u, s, vh = linalg.svd(R, full_matrices=False)
            R = u @ vh
            R_vec = R.reshape(9, 1, order='F').flatten()
        eulerAngles = rotationMatrixToEulerAngles(R) * self.R2D
        
        done = False
        done = bool(
               (abs(x) >= 1.0).any() # [m]
            or (abs(v) >= 1.0).any() # [m/s]
            or (abs(W) >= 1.0).any() # [rad/s]
            or abs(eulerAngles[0]) >= self.euler_lim # phi
            or abs(eulerAngles[1]) >= self.euler_lim # theta
        )

        return done


    def EoM(self, t, state):
        # https://youtu.be/iS5JFuopQsA
        x = np.array([state[0], state[1], state[2]]) # [m]
        v = np.array([state[3], state[4], state[5]]) # [m/s]
        R_vec = np.array([state[6], state[7], state[8],
                          state[9], state[10], state[11],
                          state[12], state[13], state[14]])
        R = R_vec.reshape(3, 3, order='F')
        W = np.array([state[15], state[16], state[17]]) # [rad/s]

        # Equations of motion of the quadrotor UAV
        x_dot = v
        v_dot = self.g*self.e3 - self.f*R@self.e3/self.m
        R_vec_dot = (R@hat(W)).reshape(1, 9, order='F')
        W_dot = inv(self.J)@(-hat(W)@self.J@W + self.M)
        state_dot = np.concatenate([x_dot.flatten(), 
                                    v_dot.flatten(),                                                                          
                                    R_vec_dot.flatten(),
                                    W_dot.flatten()])

        return np.array(state_dot)


    def sample_init_error(self, env_type='train'):
        if env_type == 'train':
            self.init_x_error = self.x_lim - 0.1 # minus 0.1m
            self.init_v_error = self.v_lim*0.2 # 20%; initial vel error, [m/s]
            self.init_R_error = 10 * self.D2R # 10 deg
            self.init_W_error = self.W_lim*0.1 # initial ang vel error, [rad/s]
        elif env_type == 'eval':
            self.init_x_error = self.x_lim - 0.1 # minus 0.1m
            self.init_v_error = self.v_lim*0.1 # initial vel error, [m/s]
            self.init_R_error = 3 * self.D2R # 3 deg
            self.init_W_error = self.W_lim*0.01 # 1%; initial ang vel error, [rad/s]


    def render(self, mode='human', close=False):
        from vpython import box, sphere, color, vector, rate, canvas, cylinder, ring, arrow, scene, textures

        # Rendering state:
        state_vis = (self.state).flatten()

        x = np.array([state_vis[0], state_vis[1], state_vis[2]]) # [m]
        v = np.array([state_vis[3], state_vis[4], state_vis[5]]) # [m/s]
        R_vec = np.array([state_vis[6], state_vis[7], state_vis[8],
                          state_vis[9], state_vis[10], state_vis[11],
                          state_vis[12], state_vis[13], state_vis[14]])
        W = np.array([state_vis[15], state_vis[16], state_vis[17]]) # [rad/s]

        quad_pos = x # [m]
        cmd_pos  = self.xd # [m]

        # Axis:
        x_axis = np.array([state_vis[6], state_vis[7], state_vis[8]])
        y_axis = np.array([state_vis[9], state_vis[10], state_vis[11]])
        z_axis = np.array([state_vis[12], state_vis[13], state_vis[14]])

        # Init:
        if self.viewer is None:
            # Canvas.
            self.viewer = canvas(title = 'Quadrotor with RL', width = 1024, height = 768, \
                                 center = vector(0, 0, cmd_pos[2]), background = color.white, \
                                 forward = vector(1, 0.3, 0.3), up = vector(0, 0, -1)) # forward = view point
            
            # Quad body.
            self.render_quad1 = box(canvas = self.viewer, pos = vector(quad_pos[0], quad_pos[1], quad_pos[2]), \
                                    axis = vector(x_axis[0], x_axis[1], x_axis[2]), \
                                    length = 0.2, height = 0.05, width = 0.05) # vector(quad_pos[0], quad_pos[1], 0)
            self.render_quad2 = box(canvas = self.viewer, pos = vector(quad_pos[0], quad_pos[1], quad_pos[2]), \
                                    axis = vector(y_axis[0], y_axis[1], y_axis[2]), \
                                    length = 0.2, height = 0.05, width = 0.05)
            # Rotors.
            rotors_offest = 0.02
            self.render_rotor1 = cylinder(canvas = self.viewer, pos = vector(quad_pos[0], quad_pos[1], quad_pos[2]), \
                                          axis = vector(rotors_offest*z_axis[0], rotors_offest*z_axis[1], rotors_offest*z_axis[2]), \
                                          radius = 0.2, color = color.blue, opacity = 0.5)
            self.render_rotor2 = cylinder(canvas = self.viewer, pos = vector(quad_pos[0], quad_pos[1], quad_pos[2]), \
                                          axis = vector(rotors_offest*z_axis[0], rotors_offest*z_axis[1], rotors_offest*z_axis[2]), \
                                          radius = 0.2, color = color.cyan, opacity = 0.5)
            self.render_rotor3 = cylinder(canvas = self.viewer, pos = vector(quad_pos[0], quad_pos[1], quad_pos[2]), \
                                          axis = vector(rotors_offest*z_axis[0], rotors_offest*z_axis[1], rotors_offest*z_axis[2]), \
                                          radius = 0.2, color = color.blue, opacity = 0.5)
            self.render_rotor4 = cylinder(canvas = self.viewer, pos = vector(quad_pos[0], quad_pos[1], quad_pos[2]), \
                                          axis = vector(rotors_offest*z_axis[0], rotors_offest*z_axis[1], rotors_offest*z_axis[2]), \
                                          radius = 0.2, color = color.cyan, opacity = 0.5)

            # Force arrows.
            self.render_force_rotor1 = arrow(pos = vector(quad_pos[0], quad_pos[1], quad_pos[2]), \
                                             axis = vector(z_axis[0], z_axis[1], z_axis[2]), \
                                             shaftwidth = 0.05, color = color.blue)
            self.render_force_rotor2 = arrow(pos = vector(quad_pos[0], quad_pos[1], quad_pos[2]), \
                                             axis = vector(z_axis[0], z_axis[1], z_axis[2]), \
                                             shaftwidth = 0.05, color = color.cyan)
            self.render_force_rotor3 = arrow(pos = vector(quad_pos[0], quad_pos[1], quad_pos[2]), \
                                             axis = vector(z_axis[0], z_axis[1], z_axis[2]), \
                                             shaftwidth = 0.05, color = color.blue)
            self.render_force_rotor4 = arrow(pos = vector(quad_pos[0], quad_pos[1], quad_pos[2]), \
                                             axis = vector(z_axis[0], z_axis[1], z_axis[2]), \
                                             shaftwidth = 0.05, color = color.cyan)
                                    
            # Commands.
            self.render_ref = sphere(canvas = self.viewer, pos = vector(cmd_pos[0], cmd_pos[1], cmd_pos[2]), \
                                     radius = 0.07, color = color.red, \
                                     make_trail = True, trail_type = 'points', interval = 50)									
            
            # Inertial axis.				
            self.e1_axis = arrow(pos = vector(2.5, -2.5, 0), axis = 0.5*vector(1, 0, 0), \
                                 shaftwidth = 0.04, color=color.blue)
            self.e2_axis = arrow(pos = vector(2.5, -2.5, 0), axis = 0.5*vector(0, 1, 0), \
                                 shaftwidth = 0.04, color=color.green)
            self.e3_axis = arrow(pos = vector(2.5, -2.5, 0), axis = 0.5*vector(0, 0, 1), \
                                 shaftwidth = 0.04, color=color.red)

            # Body axis.				
            self.render_b1_axis = arrow(canvas = self.viewer, 
                                        pos = vector(quad_pos[0], quad_pos[1], quad_pos[2]), \
                                        axis = vector(x_axis[0], x_axis[1], x_axis[2]), \
                                        shaftwidth = 0.02, color = color.blue,
                                        make_trail = True, trail_color = color.yellow)
            self.render_b2_axis = arrow(canvas = self.viewer, 
                                        pos = vector(quad_pos[0], quad_pos[1], quad_pos[2]), \
                                        axis = vector(y_axis[0], y_axis[1], y_axis[2]), \
                                        shaftwidth = 0.02, color = color.green)
            self.render_b3_axis = arrow(canvas = self.viewer, 
                                        pos = vector(quad_pos[0], quad_pos[1], quad_pos[2]), \
                                        axis = vector(z_axis[0], z_axis[1], z_axis[2]), \
                                        shaftwidth = 0.02, color = color.red)

            # Floor.
            self.render_floor = box(pos = vector(0,0,0),size = vector(5,5,0.05), axis = vector(1,0,0), \
                                    opacity = 0.2, color = color.black)


        # Update visualization component:
        if self.state is None: 
            return None

        # Update quad body.
        self.render_quad1.pos.x = quad_pos[0]
        self.render_quad1.pos.y = quad_pos[1]
        self.render_quad1.pos.z = quad_pos[2]
        self.render_quad2.pos.x = quad_pos[0]
        self.render_quad2.pos.y = quad_pos[1]
        self.render_quad2.pos.z = quad_pos[2]

        self.render_quad1.axis.x = x_axis[0]
        self.render_quad1.axis.y = x_axis[1]	
        self.render_quad1.axis.z = x_axis[2]
        self.render_quad2.axis.x = y_axis[0]
        self.render_quad2.axis.y = y_axis[1]
        self.render_quad2.axis.z = y_axis[2]

        self.render_quad1.up.x = z_axis[0]
        self.render_quad1.up.y = z_axis[1]
        self.render_quad1.up.z = z_axis[2]
        self.render_quad2.up.x = z_axis[0]
        self.render_quad2.up.y = z_axis[1]
        self.render_quad2.up.z = z_axis[2]

        # Update rotors.
        rotors_offest = -0.02
        rotor_pos = 0.5*x_axis
        self.render_rotor1.pos.x = quad_pos[0] + rotor_pos[0]
        self.render_rotor1.pos.y = quad_pos[1] + rotor_pos[1]
        self.render_rotor1.pos.z = quad_pos[2] + rotor_pos[2]
        rotor_pos = 0.5*y_axis
        self.render_rotor2.pos.x = quad_pos[0] + rotor_pos[0]
        self.render_rotor2.pos.y = quad_pos[1] + rotor_pos[1]
        self.render_rotor2.pos.z = quad_pos[2] + rotor_pos[2]
        rotor_pos = (-0.5)*x_axis
        self.render_rotor3.pos.x = quad_pos[0] + rotor_pos[0]
        self.render_rotor3.pos.y = quad_pos[1] + rotor_pos[1]
        self.render_rotor3.pos.z = quad_pos[2] + rotor_pos[2]
        rotor_pos = (-0.5)*y_axis
        self.render_rotor4.pos.x = quad_pos[0] + rotor_pos[0]
        self.render_rotor4.pos.y = quad_pos[1] + rotor_pos[1]
        self.render_rotor4.pos.z = quad_pos[2] + rotor_pos[2]

        self.render_rotor1.axis.x = rotors_offest*z_axis[0]
        self.render_rotor1.axis.y = rotors_offest*z_axis[1]
        self.render_rotor1.axis.z = rotors_offest*z_axis[2]
        self.render_rotor2.axis.x = rotors_offest*z_axis[0]
        self.render_rotor2.axis.y = rotors_offest*z_axis[1]
        self.render_rotor2.axis.z = rotors_offest*z_axis[2]
        self.render_rotor3.axis.x = rotors_offest*z_axis[0]
        self.render_rotor3.axis.y = rotors_offest*z_axis[1]
        self.render_rotor3.axis.z = rotors_offest*z_axis[2]
        self.render_rotor4.axis.x = rotors_offest*z_axis[0]
        self.render_rotor4.axis.y = rotors_offest*z_axis[1]
        self.render_rotor4.axis.z = rotors_offest*z_axis[2]

        self.render_rotor1.up.x = y_axis[0]
        self.render_rotor1.up.y = y_axis[1]
        self.render_rotor1.up.z = y_axis[2]
        self.render_rotor2.up.x = y_axis[0]
        self.render_rotor2.up.y = y_axis[1]
        self.render_rotor2.up.z = y_axis[2]
        self.render_rotor3.up.x = y_axis[0]
        self.render_rotor3.up.y = y_axis[1]
        self.render_rotor3.up.z = y_axis[2]
        self.render_rotor4.up.x = y_axis[0]
        self.render_rotor4.up.y = y_axis[1]
        self.render_rotor4.up.z = y_axis[2]

        # Update force arrows.
        rotor_pos = 0.5*x_axis
        self.render_force_rotor1.pos.x = quad_pos[0] + rotor_pos[0]
        self.render_force_rotor1.pos.y = quad_pos[1] + rotor_pos[1]
        self.render_force_rotor1.pos.z = quad_pos[2] + rotor_pos[2]
        rotor_pos = 0.5*y_axis
        self.render_force_rotor2.pos.x = quad_pos[0] + rotor_pos[0]
        self.render_force_rotor2.pos.y = quad_pos[1] + rotor_pos[1]
        self.render_force_rotor2.pos.z = quad_pos[2] + rotor_pos[2]
        rotor_pos = (-0.5)*x_axis
        self.render_force_rotor3.pos.x = quad_pos[0] + rotor_pos[0]
        self.render_force_rotor3.pos.y = quad_pos[1] + rotor_pos[1]
        self.render_force_rotor3.pos.z = quad_pos[2] + rotor_pos[2]
        rotor_pos = (-0.5)*y_axis
        self.render_force_rotor4.pos.x = quad_pos[0] + rotor_pos[0]
        self.render_force_rotor4.pos.y = quad_pos[1] + rotor_pos[1]
        self.render_force_rotor4.pos.z = quad_pos[2] + rotor_pos[2]

        force_offest = -0.05
        self.render_force_rotor1.axis.x = force_offest * self.f1 * z_axis[0] 
        self.render_force_rotor1.axis.y = force_offest * self.f1 * z_axis[1]
        self.render_force_rotor1.axis.z = force_offest * self.f1 * z_axis[2]
        self.render_force_rotor2.axis.x = force_offest * self.f2 * z_axis[0]
        self.render_force_rotor2.axis.y = force_offest * self.f2 * z_axis[1]
        self.render_force_rotor2.axis.z = force_offest * self.f2 * z_axis[2]
        self.render_force_rotor3.axis.x = force_offest * self.f3 * z_axis[0]
        self.render_force_rotor3.axis.y = force_offest * self.f3 * z_axis[1]
        self.render_force_rotor3.axis.z = force_offest * self.f3 * z_axis[2]
        self.render_force_rotor4.axis.x = force_offest * self.f4 * z_axis[0]
        self.render_force_rotor4.axis.y = force_offest * self.f4 * z_axis[1]
        self.render_force_rotor4.axis.z = force_offest * self.f4 * z_axis[2]

        # Update commands.
        self.render_ref.pos.x = cmd_pos[0]
        self.render_ref.pos.y = cmd_pos[1]
        self.render_ref.pos.z = cmd_pos[2]

        # Update body axis.
        axis_offest = 0.8
        self.render_b1_axis.pos.x = quad_pos[0]
        self.render_b1_axis.pos.y = quad_pos[1]
        self.render_b1_axis.pos.z = quad_pos[2]
        self.render_b2_axis.pos.x = quad_pos[0]
        self.render_b2_axis.pos.y = quad_pos[1]
        self.render_b2_axis.pos.z = quad_pos[2]
        self.render_b3_axis.pos.x = quad_pos[0]
        self.render_b3_axis.pos.y = quad_pos[1]
        self.render_b3_axis.pos.z = quad_pos[2]

        self.render_b1_axis.axis.x = axis_offest * x_axis[0] 
        self.render_b1_axis.axis.y = axis_offest * x_axis[1] 
        self.render_b1_axis.axis.z = axis_offest * x_axis[2] 
        self.render_b2_axis.axis.x = axis_offest * y_axis[0] 
        self.render_b2_axis.axis.y = axis_offest * y_axis[1] 
        self.render_b2_axis.axis.z = axis_offest * y_axis[2] 
        self.render_b3_axis.axis.x = (axis_offest/2) * z_axis[0] 
        self.render_b3_axis.axis.y = (axis_offest/2) * z_axis[1]
        self.render_b3_axis.axis.z = (axis_offest/2) * z_axis[2]

        # Screen capture:
        """
        if (self.render_index % 5) == 0:
            self.viewer.capture('capture'+str(self.render_index))
        self.render_index += 1        
        """

        rate(30) # FPS

        return True


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def close(self):
        if self.viewer:
            self.viewer = None