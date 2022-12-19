import numpy as np
from numpy import linalg
from math import cos, sin, pi

def ctrl_sat(action, eX, min_act, max_act, env):

    # Normalized hovering thrust:
    hover_force = np.interp(env.f_each, [env.min_force, env.max_force], [-1., 1.]) # normalized into [-1, 1]

    # Normalized position error:
    eX = linalg.norm(eX, 2)/env.x_lim # normalized into [-1, 1]

    # Slopes:
    c1 = max_act - hover_force
    c2 = -(hover_force - min_act)
    '''
    c3 = -c1
    c4 = -c2

    # Saturation:
    if eX >= 0:
        action = np.clip(action, c2*eX + hover_force, c1*eX + hover_force)
    else:
        action = np.clip(action, c4*eX + hover_force, c3*eX + hover_force)
    '''
    # Saturation:
    action = np.clip(action, c2*eX + hover_force, c1*eX + hover_force)
    return action

# Rotation on e3 axis
def R_e3(theta):
    return np.array([[cos(theta), -sin(theta), 0.0],
                     [sin(theta),  cos(theta), 0.0],
                     [       0.0,         0.0, 1.0]])

# Equivariant Reinforcement Learning for Quadrotor UAV
def rot_e3(state):
    # Actual quadrotor states
    x = np.array([state[0], state[1], state[2]])
    v = np.array([state[3], state[4], state[5]])
    R_vec = np.array([state[6],  state[7],  state[8],
                      state[9],  state[10], state[11],
                      state[12], state[13], state[14]])
    R = R_vec.reshape(3, 3, order='F')
    W = np.array([state[15], state[16], state[17]])

    # Compute theta
    theta_x = np.arctan2(state[1], state[0]) 

    # Imaginary quadrotor states
    x_equiv = R_e3(-theta_x) @ x
    v_equiv = R_e3(-theta_x) @ v
    R_equiv = R_e3(-theta_x) @ R
    R_vec_equiv = R_equiv.reshape(1, 9, order='F')
    state_equiv = np.concatenate((x_equiv[0], x_equiv[2], v_equiv, R_vec_equiv, W), axis=None)

    return state_equiv


def derot_e3(next_state):
    # Next actual quadrotor states
    next_x = np.array([next_state[0], next_state[1], next_state[2]])
    next_v = np.array([next_state[3], next_state[4], next_state[5]])
    next_R_vec = np.array([next_state[6],  next_state[7],  next_state[8],
                           next_state[9],  next_state[10], next_state[11],
                           next_state[12], next_state[13], next_state[14]])
    next_R = next_R_vec.reshape(3, 3, order='F')
    next_W = np.array([next_state[15], next_state[16], next_state[17]])

    # Compute theta
    next_theta_x = np.arctan2(next_state[1], next_state[0]) 

    # Next imaginary quadrotor states
    next_x_equiv = R_e3(-next_theta_x) @ next_x
    next_v_equiv = R_e3(-next_theta_x) @ next_v
    next_R_equiv = R_e3(-next_theta_x) @ next_R
    next_R_vec_equiv = next_R_equiv.reshape(1, 9, order='F')
    next_state_equiv = np.concatenate((next_x_equiv[0], next_x_equiv[2], next_v_equiv, \
                                       next_R_vec_equiv, next_W), axis=None)


    return next_state_equiv