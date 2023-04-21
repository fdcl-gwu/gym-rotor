import numpy as np
from math import cos, sin, pi
from gym_rotor.envs.quad_utils import *

# Rotation on e3 axis
def R_e3(theta):
    return np.array([[cos(theta), -sin(theta), 0.0],
                     [sin(theta),  cos(theta), 0.0],
                     [       0.0,         0.0, 1.0]])

# Equivariant Reinforcement Learning for Quadrotor UAV
def equiv_state(state):
    # Actual quadrotor states
    x, v, R_vec, W = state[0:3], state[3:6], state[6:15], state[15:18]
    R = R_vec.reshape(3, 3, order='F')

    # Compute theta
    theta_x = np.arctan2(state[1], state[0]) 

    # Imaginary quadrotor states
    x_equiv = R_e3(-theta_x) @ x
    v_equiv = R_e3(-theta_x) @ v
    R_equiv = R_e3(-theta_x) @ R
    R_vec_equiv = R_equiv.reshape(1, 9, order='F')

    state_equiv = np.concatenate((x_equiv[0], x_equiv[2], v_equiv, R_vec_equiv, W), axis=None)

    return state_equiv

# Decomposing equivariant state vectors
def equiv_state_decomposition(state):
    # Actual quadrotor states
    x, v, R_vec, W = state[0:3], state[3:6], state[6:15], state[15:18]
    R = R_vec.reshape(3, 3, order='F')

    # Compute theta
    theta_x = np.arctan2(state[1], state[0]) 

    # Imaginary quadrotor states
    x_equiv = R_e3(-theta_x) @ x
    v_equiv = R_e3(-theta_x) @ v
    R_equiv = R_e3(-theta_x) @ R

    return x_equiv, v_equiv, R_equiv, W


# Check if heading error is equivariant 
def get_actual_b1d(x_actual, b1d_equiv):
    # Compute theta:
    theta_x = np.arctan2(x_actual[1], x_actual[0]) 

    # Actual heading commands:
    b1d_actual = R_e3(theta_x) @ b1d_equiv

    return b1d_actual

# Equivariant b1d
# b1d: Real desired heading direction:
def get_equiv_b1d(x, b1d):

    # Compute theta:
    theta_x = np.arctan2(x[1], x[0]) 

    # Imaginary desired heading direction:
    b1d_equiv = R_e3(-theta_x) @ b1d

    return b1d_equiv