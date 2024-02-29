import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
from scipy.integrate import odeint, solve_ivp

from gym_rotor.envs.quad import QuadEnv
from gym_rotor.envs.quad_utils import *
from gym_rotor.wrappers.decoupled_yaw_utils import *
from typing import Optional
import args_parse

class DecoupledWrapper(QuadEnv):
    # metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode: Optional[str] = None): 
        super().__init__()

        # Hyperparameters:
        parser = args_parse.create_parser()
        args = parser.parse_args()
        self.alpha, self.beta = args.alpha, args.beta # addressing noise or delay

        # b3d commands:
        self.b3d = np.array([0.,0.,1])

        # limits of states:
        self.eIx_lim  = 10.0 
        self.eIb1_lim = 10.0 


    def reset(self, env_type='train',
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        # super().reset(seed=seed)
        QuadEnv.reset(self, env_type)

        # Reset forces & moments:
        self.fM = np.zeros((4, 1)) # Force-moment vector
        self.M3 = 0. # [Nm]

        # Reset errors:
        self.ex, self.ev = np.zeros(3), np.zeros(3)
        _, _, R, _ = state_decomposition(self.state) # decomposing state vectors
        b1 = R @ np.array([1.,0.,0.])
        b3 = R @ np.array([0.,0.,1.])
        b1c = -(hat(b3) @ hat(b3)) @ self.b1d # desired b1 
        self.eb1 = norm_ang_btw_two_vectors(b1c, b1) # b1 error, [-1, 1)
        self.eb3 = norm_ang_btw_two_vectors(self.b3d, b3) # b3 error, [-1, 1)

        # Reset integral terms:
        self.eIx  = np.zeros(3)
        self.eIb1 = 0.
        self.eIX.set_zero() # Set all integrals to zero
        self.eIR.set_zero()

        # Agent1's obs:
        obs_1 = np.concatenate((decoupled_obs1_decomposition(self.state, self.eIx)), axis=None)
        # Agent2's obs:
        obs_2 = np.concatenate((decoupled_obs2_decomposition(self.state, self.eb1, self.eIb1)), axis=None)

        return [obs_1, obs_2]
        
        
    def action_wrapper(self, action):
        # Linear scale, [-1, 1] -> [min_act, max_act] 
        f_total = (
            4 * (self.scale_act * action[0] + self.avrg_act)
            ).clip(4*self.min_force, 4*self.max_force)

        self.f   = f_total # [N]
        self.tau = action[1:4] 
        self.M3  = action[4] # [Nm]
        
        return action


    def observation_wrapper(self, state):
        # De-normalization: [-1, 1] -> [max, min]
        x, v, R, W = state_de_normalization(state, self.x_lim, self.v_lim, self.W_lim)
        R_vec = R.reshape(9, 1, order='F').flatten()
        state = np.concatenate((x, v, R_vec, W), axis=0)

        # Convert each forces to force-moment:
        self.fM[0] = self.f
        b1, b2 = R@self.e1, R@self.e2
        self.fM[1] = b1.T @ self.tau + self.J[2,2]*W[2]*W[1] # M1
        self.fM[2] = b2.T @ self.tau - self.J[2,2]*W[2]*W[0] # M2
        self.fM[3] = self.M3

        # Solve ODEs: method = 'DOP853', 'BDF', 'Radau', 'RK45', ...
        sol = solve_ivp(self.decouple_EoM, [0, self.dt], state, method='DOP853')
        self.state = sol.y[:,-1]

        # Normalization: [max, min] -> [-1, 1]
        x_norm, v_norm, R_vec, W_norm = state_normalization(self.state, self.x_lim, self.v_lim, self.W_lim)
        self.state = np.concatenate((x_norm, v_norm, R_vec, W_norm), axis=0)

        # Update integral terms:
        x, v, b3, _, _ = decoupled_obs1_decomposition(self.state, self.eIx) # Agent1's obs
        self.ex = x - self.xd # position error
        self.ev = v - self.vd # velocity error
        self.eIX.integrate(-self.alpha*self.eIX.error + x_norm*self.x_lim, self.dt) 
        self.eIx = clip(self.eIX.error/self.eIx_lim, -self.sat_sigma, self.sat_sigma)

        b1, _, _, _ = decoupled_obs2_decomposition(self.state, self.eb1, self.eIb1) # Agent2's obs
        b1c = -(hat(b3) @ hat(b3)) @ self.b1d # desired b1 
        self.eb1 = norm_ang_btw_two_vectors(b1c, b1) # b1 error, [-1, 1)
        self.eb3 = norm_ang_btw_two_vectors(self.b3d, b3) # b3 error, [-1, 1)
        self.eIR.integrate(-self.beta*self.eIR.error + self.eb1*np.pi, self.dt) # b1 integral error
        self.eIb1 = clip(self.eIR.error/self.eIb1_lim, -self.sat_sigma, self.sat_sigma)

        # Agent1's obs:
        obs_1 = np.concatenate((decoupled_obs1_decomposition(self.state, self.eIx)), axis=None)
        # Agent2's obs:
        obs_2 = np.concatenate((decoupled_obs2_decomposition(self.state, self.eb1, self.eIb1)), axis=None)

        return [obs_1, obs_2]
    

    def reward_wrapper(self, obs):
        # Agent1's obs
        _, _, _, w12, eIx = decoupled_obs1_decomposition(self.state, self.eIx) 

        # Agent1's reward:
        reward_eX   = -self.Cx*(norm(self.ex, 2)**2) 
        reward_eIX  = -self.CIx*(norm(eIx, 2)**2)
        reward_eV   = -self.Cv*(norm(self.ev, 2)**2)
        reward_eb3  = -self.Cb3*abs(self.eb3)
        reward_ew12 = -self.Cw12*(norm(w12, 2)**2)
        rwd_1 = reward_eX + reward_eIX+ reward_eV + reward_eb3 + reward_ew12

        # Agent2's obs
        _, W3, eb1, eIb1 = decoupled_obs2_decomposition(self.state, self.eb1, self.eIb1)

        # Agent2's reward:
        reward_eb1  = -self.Cb1*abs(eb1)
        reward_eIb1 = -self.CIb1*abs(eIb1)
        reward_eW3  = -self.CW3*(abs(W3)**2)
        rwd_2 = reward_eb1 + reward_eIb1 + reward_eW3 

        return [rwd_1, rwd_2]


    def done_wrapper(self, obs):
        # Decomposing state vectors
        _, _, _, W = state_decomposition(self.state)
        # Agent1's obs
        x, v, _, _, eIx = decoupled_obs1_decomposition(self.state, self.eIx) 

        # Agent1's terminal states:
        done_1 = False
        done_1 = bool(
               (abs(x) >= 1.0).any() # [m]
            or (abs(v) >= 1.0).any() # [m/s]
            or (abs(W[0]) >= 1.0) # [rad/s]
            or (abs(W[1]) >= 1.0) # [rad/s]
            #or (abs(eIx) >= 1.0).any()
        )

        # Agent2's obs
        _, W3, _, eIb1 = decoupled_obs2_decomposition(self.state, self.eb1, self.eIb1)

        # Agent2's terminal states:
        done_2 = False
        done_2 = bool(
            (abs(W3) >= 1.0) # [rad/s]
            or (abs(self.eb1) >= 1.0)
            # or (abs(eIb1) >= 1.0).any()
        )

        return [done_1, done_2]


    def decouple_EoM(self, t, state):
        # Parameters:
        m, g, J = self.m, self.g, self.J

        # Decomposing state vectors
        _, v, R, W = state_decomposition(state)

        M = self.fM[1:4].ravel()
        # Equations of motion of the quadrotor UAV
        x_dot = v
        v_dot = g*self.e3 - self.f*R@self.e3/m
        R_vec_dot = (R@hat(W)).reshape(1, 9, order='F')
        W_dot = inv(J)@(-hat(W)@J@W + M)
        state_dot = np.concatenate([x_dot.flatten(), 
                                    v_dot.flatten(),                                                                          
                                    R_vec_dot.flatten(),
                                    W_dot.flatten()])

        return np.array(state_dot)