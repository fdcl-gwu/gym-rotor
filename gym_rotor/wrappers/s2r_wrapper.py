import numpy as np
from numpy import linalg
from numpy.linalg import inv
from numpy.random import uniform 
from math import cos, sin, pi
from scipy.integrate import odeint, solve_ivp
from transforms3d.euler import euler2mat, mat2euler

from gym_rotor.envs.quad import QuadEnv
from gym_rotor.envs.quad_utils import *
from gym_rotor.wrappers.sensor_noise import SensorNoise


class Sim2RealWrapper(QuadEnv):

    def __init__(self): 
        super().__init__()
        self.motor_init_has_run = False
        self.sensor_noise = SensorNoise(bypass=False)

    def reset(self, env_type='train'):
        # Initial states:
        QuadEnv.reset(self, env_type)
        
        # Domain randomization:
        self.set_random_parameters(env_type)

        # Flags for motor and sensor noises:
        #self.set_noise(env_type)

        return np.array(self.state)

    """
    def action_wrapper(self, action):
        # Motor initialize:        
        if not self.motor_init_has_run:
            self.LPF = LowPassFilter(prev_filtered_act=action, cutoff_freq=100, dt=self.dt) # 7 - 15
            # sigma = 0.2 gives roughly max noise of -1 .. 1
            thrust_noise_ratio = 0.05
            self.motor_noise = OUNoise(self.action_space.shape[0], sigma=0.2*thrust_noise_ratio)
            self.motor_init_has_run = True

        # Motor model:
        motor_noise = action * self.motor_noise.noise() if self.motor_noise else np.zeros(4)
        delayed_action = self.LPF.filter(action) + motor_noise # Motor delay

        # Linear scale, [-1, 1] -> [min_act, max_act] 
        delayed_action = (
            self.scale_act * delayed_action + self.avrg_act
            ).clip(self.min_force, self.max_force)

        # Saturated thrust of each motor:
        self.f1 = delayed_action[0]
        self.f2 = delayed_action[1]
        self.f3 = delayed_action[2]
        self.f4 = delayed_action[3]

        # Convert each forces to force-moment:
        self.fM = self.forces_to_fM @ delayed_action
        self.f = self.fM[0]   # [N]
        self.M = self.fM[1:4] # [Nm]  

        return delayed_action
    """

    """
    def observation_wrapper(self, state):

        # De-normalization: [-1, 1] -> [max, min]
        pos, vel, rot, omega = state_de_normalization(state, self.x_lim, self.v_lim, self.W_lim)

        # Sensor model:
        x, v, R, W = self.sensor_noise.add_noise(
            pos=pos, vel=vel, rot=rot, omega=omega, dt=self.dt)
        # Re-orthonormalize:
        if not isRotationMatrix(R):
            ''' https://www.codefull.net/2017/07/orthonormalize-a-rotation-matrix/ '''
            u, s, vh = linalg.svd(R, full_matrices=False)
            R = u @ vh
        R_vec = R.reshape(9, 1, order='F').flatten()
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

        # Normalization: [max, min] -> [-1, 1]
        x_norm, v_norm, R, W_norm = state_normalization(self.state, self.x_lim, self.v_lim, self.W_lim)
        R_vec = R.reshape(9, 1, order='F').flatten()
        self.state = np.concatenate((x_norm, v_norm, R_vec, W_norm), axis=0)

        return self.state
    """

    def set_random_parameters(self, env_type='train'):
        # Nominal quadrotor parameters:
        self.m = 1.85 # mass of quad, [kg]
        self.d = 0.23 # arm length, [m]
        J1, J2, J3 = 0.02, 0.02, 0.04
        self.J = np.diag([J1, J2, J3]) # inertia matrix of quad, [kg m2]
        self.c_tf = 0.0135 # torque-to-thrust coefficients
        self.c_tw = 2.4 # thrust-to-weight coefficients

        if env_type == 'train':
            
            uncertainty_range = 0.20 # *100 = [%]
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
            
            # TODO: Motor and Sensor noise: thrust_noise_ratio, sigma, cutoff_freq
            

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

        print('m:',f'{self.m:.3f}','d:',f'{self.d:.3f}','J:',f'{J1:.4f}',f'{J3:.4f}','c_tf:',f'{self.c_tf:.4f}','c_tw:',f'{self.c_tw:.3f}')


    def set_noise(self, env_type='train'):
        self.motor_init_has_run = False

        if env_type == 'train':
            self.motor_noise = True
            self.sensor_noise = SensorNoise(bypass=False)
        elif env_type == 'eval':  
            self.motor_noise = False
            self.sensor_noise = SensorNoise(bypass=True)