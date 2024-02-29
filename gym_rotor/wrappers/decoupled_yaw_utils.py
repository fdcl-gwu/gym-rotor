import numpy as np

def decoupled_obs1_decomposition(state, eIx):
    """
    Decompose the given state for the first agent.

    Parameters:
    - state (ndarray): The state vector containing the position, velocity, rotation matrix, and angular velocity.
    - eIx (ndarray): The integral error of ex.

    Returns:
    - x (ndarray): The position vector.
    - v (ndarray): The velocity vector.
    - b3 (ndarray): The third body-fixed axis
    - w12 (ndarray): The angular velocity of b3 resolved in the inertial frame.
    - eIx (ndarray): The integral error of ex.
    """
    x, v, R_vec, W = state[0:3], state[3:6], state[6:15], state[15:18]
    R = R_vec.reshape(3, 3, order='F')
    b1 = R @ np.array([1.,0.,0.])
    b2 = R @ np.array([0.,1.,0.])
    b3 = R @ np.array([0.,0.,1.])
    w12 = W[0]*b1 + W[1]*b2

    return x, v, b3, w12, eIx


def decoupled_obs2_decomposition(state, eb1, eIb1):
    """
    Decompose the given state for the second agent.

    Parameters:
    - state (ndarray): The state vector containing the position, velocity, rotation matrix, and angular velocity.
    - eb1 (float): The yaw error.
    - eIb1 (float): The integral error of eb1.

    Returns:
    - b1 (ndarray): The first body-fixed axis
    - W3 (float): The third component of angular velocity.
    - eb1 (float): The yaw error.
    - eIb1 (float): The integral error of eb1.
    """
    R_vec, W = state[6:15], state[15:18]
    R = R_vec.reshape(3, 3, order='F')
    b1 = R @ np.array([1.,0.,0.])

    return b1, W[2], eb1, eIb1
