import numpy as np
from numpy.linalg import norm
from scipy.integrate import odeint, solve_ivp

from gym_rotor.envs.quad import QuadEnv
from gym_rotor.envs.quad_utils import *
from gym_rotor.wrappers.decoupled_yaw_utils import *
from typing import Optional
import args_parse

class CoupledWrapper(QuadEnv):
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

        # Single-agent's obs:
        obs = np.concatenate((self.state, self.eIx, self.eb1, self.eIb1), axis=None)

        return [obs]


    def action_wrapper(self, action):
        # Linear scale, [-1, 1] -> [min_act, max_act] 
        f_total = (
            4 * (self.scale_act * action[0] + self.avrg_act)
            ).clip(4*self.min_force, 4*self.max_force)

        self.f = f_total # [N]
        self.M = action[1:4] # [Nm]
        
        return action


    def observation_wrapper(self, state):
        # De-normalization: [-1, 1] -> [max, min]
        x, v, R, W = state_de_normalization(state, self.x_lim, self.v_lim, self.W_lim)
        R_vec = R.reshape(9, 1, order='F').flatten()
        state = np.concatenate((x, v, R_vec, W), axis=0)

        # Solve ODEs: method = 'DOP853', 'BDF', 'Radau', 'RK45', ...
        sol = solve_ivp(self.EoM, [0, self.dt], state, method='DOP853')
        self.state = sol.y[:,-1]

        # Normalization: [max, min] -> [-1, 1]
        x_norm, v_norm, R_vec, W_norm = state_normalization(self.state, self.x_lim, self.v_lim, self.W_lim)
        self.state = np.concatenate((x_norm, v_norm, R_vec, W_norm), axis=0)

        # Update integral terms:
        x, v, R, _ = state_decomposition(self.state) 
        b1, b3 = R@self.e1, R@self.e3
        self.ex = x - self.xd # position error
        self.ev = v - self.vd # velocity error
        self.eIX.integrate(-self.alpha*self.eIX.error + x_norm*self.x_lim, self.dt) 
        self.eIx = clip(self.eIX.error/self.eIx_lim, -self.sat_sigma, self.sat_sigma)

        b1c = -(hat(b3) @ hat(b3)) @ self.b1d # desired b1 
        self.eb1 = norm_ang_btw_two_vectors(b1c, b1) # b1 error, [-1, 1)
        self.eb3 = norm_ang_btw_two_vectors(self.b3d, b3) # b3 error, [-1, 1)
        self.eIR.integrate(-self.beta*self.eIR.error + self.eb1*np.pi, self.dt) # b1 integral error
        self.eIb1 = clip(self.eIR.error/self.eIb1_lim, -self.sat_sigma, self.sat_sigma)

        # Single-agent's obs:
        obs = np.concatenate((self.state, self.eIx, self.eb1, self.eIb1), axis=None)

        return [obs]
    

    def reward_wrapper(self, obs):
        # Single-agent's obs:
        _, _, _, W = state_decomposition(self.state) 

        # Single-agent's reward:
        reward_eX   = -self.Cx*(norm(self.ex, 2)**2) 
        reward_eIX  = -self.CIx*(norm(self.eIx, 2)**2)
        reward_eV   = -self.Cv*(norm(self.ev, 2)**2)
        reward_eb1  = -self.Cb1*abs(self.eb1)
        reward_eIb1 = -self.CIb1*abs(self.eIb1)
        reward_eb3  = -self.Cb3*abs(self.eb3)
        reward_eW   = -self.CW*(norm(W, 2)**2)
        
        rwd = reward_eX + reward_eIX + reward_eV + reward_eb1 + reward_eIb1 + reward_eb3 + reward_eW

        return [rwd]


    def done_wrapper(self, obs):
        # Single-agent's obs:
        x, v, _, W = state_decomposition(self.state) 

        # Single-agent's terminal states:
        done = False
        done = bool(
               (abs(x) >= 1.0).any() # [m]
            or (abs(v) >= 1.0).any() # [m/s]
            or (abs(W) >= 1.0).any() # [rad/s]
            # or (abs(self.eIx) >= 1.0).any()
            or (abs(self.eb1) >= 1.0)
            # or (abs(self.eIb1) >= 1.0)
        )

        return [done]