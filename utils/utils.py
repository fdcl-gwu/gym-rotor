import numpy as np
from numpy import interp
from numpy.linalg import norm

def benchmark_reward_func(error_state, args):
    ex_norm = error_state[0]
    ev_norm = error_state[1]
    eW_norm = error_state[2]
    eb1 = error_state[4]
    
    reward_eX   = -args.Cx*(norm(ex_norm, 2))
    reward_eV   = -args.Cv*(norm(ev_norm, 2))
    reward_eW   = -args.Cw12*(norm(eW_norm, 2)) 
    reward_eb1  = -args.Cb1*abs(eb1)
    rwd = reward_eX + reward_eV + reward_eb1 + reward_eW

    rwd_min = -np.ceil(args.Cx+args.Cv+args.Cw12+args.Cb1)
    return interp(rwd, [rwd_min, 0.], [0., 1.]) # linear interpolation [0,1]