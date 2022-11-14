import gym
from gym import spaces
from gym.utils import seeding

import numpy as np
from numpy import linalg
from numpy.linalg import inv
from math import cos, sin, atan2, sqrt, pi

class QuadEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self): 

        # Quadrotor parameters:
        self.m = 1.65 # mass of quad, [kg]
        self.d = 0.23 # arm length, [m]
        self.J = np.diag([0.02, 0.02, 0.04]) # inertia matrix of quad, [kg m2]
        self.c_tf = 0.0135 # torques and thrusts coefficients
        self.g = 9.81  # standard gravity

        # Force and Moment:
        self.f = self.m * self.g # magnitude of total thrust to overcome  
                                 # gravity and mass (No air resistance), [N]
        self.f_each = self.m * self.g / 4.0 # thrust magnitude of each motor, [N]
        self.min_force = 0.0 # minimum thrust of each motor, [N]
        self.max_force = 2 * self.f_each # maximum thrust of each motor, [N]
        self.f1 = self.f_each # thrust of each 1st motor, [N]
        self.f2 = self.f_each # thrust of each 2nd motor, [N]
        self.f3 = self.f_each # thrust of each 3rd motor, [N]
        self.f4 = self.f_each # thrust of each 4th motor, [N]
        self.M  = np.zeros(3) # magnitude of moment on quadrotor, [Nm]

        self.fM = np.zeros((4, 1)) # Force-moment vector
        self.forces_to_fM = np.array([
            [1.0, 1.0, 1.0, 1.0],
            [0.0, -self.d, 0.0, self.d],
            [self.d, 0.0, -self.d, 0.0],
            [-self.c_tf, self.c_tf, -self.c_tf, self.c_tf]
        ]) # Conversion matrix of forces to force-moment 
        self.fM_to_forces = np.linalg.inv(self.forces_to_fM)

        # Simulation parameters:
        self.freq = 200 # frequency [Hz]
        self.dt = 1./self.freq # discrete timestep, t(2) - t(1), [sec]
        self.ode_integrator = "euler" # or "euler", ODE solvers
        self.R2D = 180/pi # [rad] to [deg]
        self.D2R = pi/180 # [deg] to [rad]
        self.e3 = np.array([0.0, 0.0, 1.0])

        # Commands:
        self.xd     = np.array([0.0, 0.0, 0.0]) # desired tracking position command, [m] 
        self.xd_dot = np.array([0.0, 0.0, 0.0]) # [m/s]
        self.b1d    = np.array([1.0, 0.0, 0.0]) # desired heading direction        

        # limits of states:
        self.x_max_threshold = 3.0 # [m]
        self.v_max_threshold = 5.0 # [m/s]
        self.W_max_threshold = 5.0 # [rad/s]
        self.euler_max_threshold = 90 # [deg]

        self.limits_x = self.x_max_threshold * np.ones(3) # [m]
        self.limits_v = self.v_max_threshold * np.ones(3) # [m/s]
        self.limits_R = np.ones(9) 
        self.limits_W = self.W_max_threshold * np.ones(3) # [rad/s]

        self.low = np.concatenate([-self.limits_x,  
                                   -self.limits_v,
                                   -self.limits_R,
                                   -self.limits_W])
        self.high = np.concatenate([self.limits_x,  
                                    self.limits_v,
                                    self.limits_R,
                                    self.limits_W])

        # Observation space:
        self.observation_space = spaces.Box(
            self.low, 
            self.high, 
            dtype=np.float64
        )
        # Action space:
        self.action_space = spaces.Box(
            low=self.min_force, 
            high=self.max_force, 
            shape=(4,),
            dtype=np.float64
        ) 

        # Init:
        self.state = None
        self.viewer = None
        self.render_quad1  = None
        self.render_quad2  = None
        self.render_rotor1 = None
        self.render_rotor2 = None
        self.render_rotor3 = None
        self.render_rotor4 = None
        self.render_ref = None
        self.render_force_rotor1 = None
        self.render_force_rotor2 = None
        self.render_force_rotor3 = None
        self.render_force_rotor4 = None
        self.render_index = 1 

        self.seed()
        self.reset()


    def step(self, action, prev_action):

        # Saturated actions:
        action = self.action_wrapper(action) # [N]
        self.f1 = action[0]
        self.f2 = action[1]
        self.f3 = action[2]
        self.f4 = action[3]

        # Convert each forces to force-moment:
        self.fM = self.forces_to_fM @ action
        self.f = self.fM[0]   # [N]
        self.M = self.fM[1:4] # [Nm]        

        # States: (x[0:3]; v[3:6]; R_vec[6:15]; W[15:18])
        state = (self.state).flatten()
                 
        # Observation:
        obs = self.observation_wrapper(state)

        # Reward function:
        reward = self.reward_wrapper(obs, action, prev_action)

        # Terminal condition:
        done = self.done_wrapper(obs)

        return obs, reward, done, {}


    def reset(self):
        raise NotImplementedError


    def action_wrapper(self, action):
        raise NotImplementedError


    def observation_wrapper(self, state):
        raise NotImplementedError
    

    def reward_wrapper(self, obs, action, prev_action):
        raise NotImplementedError


    def done_wrapper(self, obs):
        raise NotImplementedError


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
        R_vec_dot = (R@self.hat(W)).reshape(1, 9, order='F')
        W_dot = inv(self.J)@(-self.hat(W)@self.J@W + self.M)

        state_dot = np.concatenate([x_dot.flatten(), 
                                    v_dot.flatten(),                                                                          
                                    R_vec_dot.flatten(),
                                    W_dot.flatten()])

        return np.array(state_dot)


    def hat(self, x):
        hat_x = np.array([[0.0, -x[2], x[1]], \
                          [x[2], 0.0, -x[0]], \
                          [-x[1], x[0], 0.0]])
                        
        return np.array(hat_x)


    def vee(self, M):
        # vee map: inverse of the hat map
        vee_M = np.array([[M[2,1]], \
                          [M[0,2]], \
                          [M[1,0]]])

        return np.array(vee_M)


    def eulerAnglesToRotationMatrix(self, theta) :
        # Calculates Rotation Matrix given euler angles.
        R_x = np.array([[1,              0,               0],
                        [0,  cos(theta[0]),  -sin(theta[0])],
                        [0,  sin(theta[0]),   cos(theta[0])]])

        R_y = np.array([[ cos(theta[1]),   0,  sin(theta[1])],
                        [             0,   1,              0],
                        [-sin(theta[1]),   0,  cos(theta[1])]])

        R_z = np.array([[cos(theta[2]),  -sin(theta[2]),  0],
                        [sin(theta[2]),   cos(theta[2]),  0],
                        [            0,               0,  1]])

        R = np.dot(R_z, np.dot( R_y, R_x ))

        return R


    def isRotationMatrix(self, R):
        # Checks if a matrix is a valid rotation matrix.
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype = R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6


    def rotationMatrixToEulerAngles(self, R):
        # Calculates rotation matrix to euler angles.
        assert(self.isRotationMatrix(R))

        sy = sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

        singular = sy < 1e-6

        if  not singular:
            x = atan2(R[2,1] , R[2,2])
            y = atan2(-R[2,0], sy)
            z = atan2(R[1,0], R[0,0])
        else:
            x = atan2(-R[1,2], R[1,1])
            y = atan2(-R[2,0], sy)
            z = 0

        return np.array([x, y, z])


    def close(self):
        if self.viewer:
            self.viewer = None


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]