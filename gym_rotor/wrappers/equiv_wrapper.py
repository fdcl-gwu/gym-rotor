import numpy as np
from numpy import linalg
from numpy.linalg import inv
from numpy.random import uniform 
from math import cos, sin, pi
from scipy.integrate import odeint, solve_ivp
from transforms3d.euler import euler2mat, mat2euler

from gym_rotor.envs.quad import QuadEnv
from gym_rotor.envs.quad_utils import *
from gym_rotor.wrappers.equiv_utils import *

class EquivWrapper(QuadEnv):
    def __init__(self): 
        super().__init__()
        self.integral_eX_norm = True

        # Actual integral terms:
        if self.integral_eX_norm:
            self.eIX_actual = IntegralError() # Position integral error
        else:
            self.eIX_actual = IntegralErrorVec3() # Position integral error
        self.eIR_actual = IntegralError() # Attitude integral error
        self.eIX_actual.set_zero() # Set all integrals to zero
        self.eIR_actual.set_zero()

        # Equivariant integral terms:
        if self.integral_eX_norm:
            self.eIX_equiv = IntegralError() # Position integral error
        else:
            self.eIX_equiv = IntegralErrorVec3() # Position integral error
        self.eIR_equiv = IntegralError() # Attitude integral error
        self.eIX_equiv.set_zero() # Set all integrals to zero
        self.eIR_equiv.set_zero()


    def reset(self, env_type='train'):
        # Initial states:
        QuadEnv.reset(self, env_type)
 
        # Actual integral terms:
        self.eIX_actual.set_zero() # Set all integrals to zero
        self.eIR_actual.set_zero()

        # Equivariant integral terms:
        self.eIX_equiv.set_zero() # Set all integrals to zero
        self.eIR_equiv.set_zero()
        return np.array(self.state)

    
    def reward_wrapper(self, obs):

        # Reward function coefficients:
        C_X = self.C_X # pos coef.
        C_R = self.C_R # att coef.
        C_V = self.C_V # vel coef.
        C_W = self.C_W # ang_vel coef.
        CI_X = self.CI_X 
        CI_R = self.CI_R 

        # Actual quadrotor states:
        x_actual, v_actual, R_actual, _ = state_decomposition(obs)

        # Actual errors:
        eX_actual = x_actual - self.xd     # actual position err
        eV_actual = v_actual - self.xd_dot # actual velocity err
        b1d_actual = self.b1d # actual heading cmd; get_actual_b1d(x_actual, b1d_equiv)
        eR_actual = ang_btw_two_vectors(get_current_b1(R_actual), b1d_actual) # actual heading err
        if self.integral_eX_norm:
            self.eIX_actual.integrate(linalg.norm(eX_actual, 2), self.dt)
        else:
            self.eIX_actual.integrate(eX_actual, self.dt)
        self.eIX_actual.error = np.clip(self.eIX_actual.error, -self.sat_sigma, self.sat_sigma)
        self.eIR_actual.integrate(eR_actual, self.dt) 
        self.eIR_actual.error = np.clip(self.eIR_actual.error, -self.sat_sigma, self.sat_sigma)

        # Equivariant states:
        x_equiv, v_equiv, R_equiv, W = equiv_state_decomposition(obs)

        # Equivariant errors:
        eX_equiv = x_equiv - self.xd     # position error
        eV_equiv = v_equiv - self.xd_dot # velocity error
        b1d_equiv = get_equiv_b1d(x_actual, b1d_actual)
        eR_equiv = ang_btw_two_vectors(get_current_b1(R_equiv), b1d_equiv) # heading error [rad]
		# Calculate integral terms to steady-state errors:
        if self.integral_eX_norm:
            self.eIX_equiv.integrate(linalg.norm(eX_equiv, 2), self.dt) 
        else:
            self.eIX_equiv.integrate(eX_equiv, self.dt)
        self.eIX_equiv.error = np.clip(self.eIX_equiv.error, -self.sat_sigma, self.sat_sigma)
        self.eIR_equiv.integrate(eR_equiv, self.dt) 
        self.eIR_equiv.error = np.clip(self.eIR_equiv.error, -self.sat_sigma, self.sat_sigma)

        # Check if rewards are equivariant:
        satisfy_equiv = True
        satisfy_equiv = bool(
                abs((linalg.norm(eX_actual, 2) - linalg.norm(eX_equiv, 2))) < 1e-7 # eX errors
            # and abs((linalg.norm(self.eIX_actual.error, 2) - linalg.norm(self.eIX_equiv.error, 2))) < 1e-7 # eIX errors
            # and abs(self.eIX_actual.error - self.eIX_equiv.error) < 1e-7 # eIX errors
            and abs((linalg.norm(eV_actual, 2) - linalg.norm(eV_equiv, 2))) < 1e-7 # eV errors
            and abs((eR_actual - eR_equiv)) < 1e-7 # eR errors
        )
        assert satisfy_equiv == True, "satisfy_equiv should be True"
    
        # Reward function:
        reward_eX  = -C_X*(linalg.norm(eX_equiv, 2)**2) 
        # reward_eIX = -CI_X*(linalg.norm(self.eIX_equiv.error, 2)**2)
        if self.integral_eX_norm:
            reward_eIX = -CI_X*(self.eIX_actual.error**2)
        else:
            reward_eIX = -CI_X*(linalg.norm(self.eIX_actual.error, 2)**2)
        reward_eR  = -C_R*(eR_equiv/pi) # [0., pi] -> [0., 1.0]
        reward_eIR = 0. #-CI_R*eIR
        reward_eV  = -C_V*(linalg.norm(eV_equiv, 2)**2)
        reward_eW  = -C_W*(linalg.norm(W, 2)**2)

        reward = self.reward_alive + (reward_eX + reward_eIX + reward_eR + reward_eIR + reward_eV + reward_eW)

        return reward