import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
from scipy.integrate import odeint, solve_ivp

from gym_rotor.envs.quad import QuadEnv
from gym_rotor.envs.quad_utils import *
from gym_rotor.wrappers.wrapper_utils import *
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

        # limits of states:
        self.eIx_lim  = 3.0 
        self.eIb1_lim = 3.0 


    def reset(self, 
        env_type='train',
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        # super().reset(seed=seed)
        QuadEnv.reset(self, env_type)

        # Reset forces & moments:
        self.fM = np.zeros((4, 1)) # Force-moment vector
        self.M3 = 0. # [Nm]

        # Reset integral terms:
        self.eIx.set_zero() # Set all integrals to zero
        self.eIx_norm = np.zeros(3)
        self.eIb1.set_zero()
        self.eIb1_norm = 0.

        return np.array(self.state, dtype=np.float32)
        
        
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
        # Decomposing state vectors:
        x, v, R, W = state_decomposition(state)
        R_vec = R.reshape(9, 1, order='F').flatten()
        current_state = np.concatenate((x, v, R_vec, W), axis=0)

        # Convert each forces to force-moment:
        self.fM[0] = self.f
        b1, b2 = R@self.e1, R@self.e2
        self.fM[1] = b1.T @ self.tau + self.J[2,2]*W[2]*W[1] # M1
        self.fM[2] = b2.T @ self.tau - self.J[2,2]*W[2]*W[0] # M2
        self.fM[3] = self.M3

        # Solve ODEs: method = 'DOP853', 'BDF', 'Radau', 'RK45', ...
        sol = solve_ivp(self.decouple_EoM, [0, self.dt], current_state, method='DOP853')
        next_state = sol.y[:,-1]

        # Next state vec: (x_next[0:3]; v_next[3:6]; R_next[6:15]; W_next[15:18])
        self.state = next_state

        # Modular agents' obs:
        """
        norm_obs_1 = (ex_norm, eIx_norm, ev_norm, b3, ew12_norm)
        norm_obs_2 = (eb1_norm, eIb1_norm, eW3_norm)
        """
        obs = self.get_norm_error_state(self.framework)

        return obs
    

    def reward_wrapper(self, obs):
        # Agent1's obs
        ex_norm, eIx_norm, ev_norm, _, ew12_norm = obs1_decomposition(obs[0]) 

        # Agent1's reward:
        reward_eX = -self.Cx*(norm(ex_norm, 2)**2) 
        reward_eIX = -self.CIx*(norm(eIx_norm, 2)**2)
        reward_eV = -self.Cv*(norm(ev_norm, 2)**2)
        reward_ew12 = -self.Cw12*(norm(ew12_norm, 2)**2)
        rwd_1 = reward_eX + reward_eIX+ reward_eV + reward_ew12

        # Agent2's obs
        eb1_norm, eIb1_norm, eW3_norm = obs2_decomposition(obs[1])

        # Agent2's reward:
        reward_eb1  = -self.Cb1*abs(eb1_norm)
        reward_eIb1 = -self.CIb1*(abs(eIb1_norm)**2)
        reward_eW3  = -self.CW3*(abs(eW3_norm)**2)

        rwd_2 = reward_eb1 + reward_eIb1 + reward_eW3

        return [rwd_1, rwd_2]


    def done_wrapper(self, obs):
        # Agent1's obs
        ex_norm, eIx_norm, ev_norm, _, ew12_norm = obs1_decomposition(obs[0]) 

        # Agent1's terminal states:
        done_1 = False
        done_1 = bool(
               (abs(ex_norm) >= 1.0).any() 
            or (abs(ev_norm) >= 1.0).any() 
            or (abs(ew12_norm) >= 1.0).any() 
            #or (abs(eIx_norm) >= 1.0).any()
        )

        # Agent2's obs
        eb1_norm, eIb1_norm, eW3_norm = obs2_decomposition(obs[1])

        # Agent2's terminal states:
        done_2 = False
        done_2 = bool(
            (abs(eW3_norm) >= 1.0) 
            # or (abs(eb1_norm) >= 1.0) 
            # or (abs(eIb1_norm) >= 1.0).any()
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