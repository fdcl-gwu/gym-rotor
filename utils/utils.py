import os
import torch
import random
import numpy as np
from numpy import interp
from numpy.linalg import norm

def set_seed(env, seed: int = 1992) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)  # set a fixed value for the hash seed
    env.action_space.seed(seed)
    env.observation_space.seed(seed)


def get_error_state(norm_obs_n, x_lim, v_lim, eIx_lim, eIb1_lim, args):
    # norm_obs_1 = (ex_norm, eIx_norm, ev_norm, b3, ew12_norm)
    # norm_obs_2 = (eb1_norm, eIb1_norm, eW3_norm)
    if args.framework == "MODUL":
        norm_obs_1, norm_obs_2 = norm_obs_n[0], norm_obs_n[1]
        ex = norm_obs_1[0:3] * x_lim
        eIx = norm_obs_1[3:6] * eIx_lim
        ev = norm_obs_1[6:9] * v_lim
        eb1 = norm_obs_2[0] * np.pi
        eIb1 = norm_obs_2[1] * eIb1_lim
    # norm_obs = (ex_norm, eIx_norm, ev_norm, R_vec, eb1_norm, eIb1_norm, eW_norm)
    elif args.framework == "MONO":
        ex = norm_obs_n[0][0:3] * x_lim
        eIx = norm_obs_n[0][3:6] * eIx_lim
        ev = norm_obs_n[0][6:9] * v_lim 
        eb1 = norm_obs_n[0][18] * np.pi
        eIb1 = norm_obs_n[0][19] * eIb1_lim

    return ex, eIx, ev, eb1, eIb1


def benchmark_reward_func(ex, eb1):    
    reward_eX = -norm(ex)
    reward_eb1 = -abs(eb1)
    rwd = reward_eX + reward_eb1

    return interp(rwd, [-2., 0.], [0., 1.]) # linear interpolation [0,1]
