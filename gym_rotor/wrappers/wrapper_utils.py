import numpy as np

def obs_decomposition(obs):
    ex_norm = obs[0:3]
    eIx_norm = obs[3:6]
    ev_norm = obs[6:9]
    R_vec = obs[9:18]
    eb1_norm = obs[18]
    eIb1_norm = obs[19]
    eW_norm = obs[20:23]

    return ex_norm, eIx_norm, ev_norm, R_vec, eb1_norm, eIb1_norm, eW_norm

def obs1_decomposition(obs):
    ex_norm = obs[0:3]
    eIx_norm = obs[3:6]
    ev_norm = obs[6:9]
    b3 = obs[9:12]
    ew12_norm = obs[12:15]

    return ex_norm, eIx_norm, ev_norm, b3, ew12_norm


def obs2_decomposition(obs):
    eb1_norm = obs[0]
    eIb1_norm = obs[1]
    eW3_norm = obs[2]

    return eb1_norm, eIb1_norm, eW3_norm