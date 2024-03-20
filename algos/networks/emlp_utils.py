import torch
from torch import zeros_like, ones_like, atan2, cos, sin

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA found.")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS found.")
else:
    device = torch.device("cpu")


def get_equiv_state(state):
    """
    Equivariant Reinforcement Learning for Quadrotor UAV.
    Convert the input state to the equivariant state vector.

    Args:
        state (torch.Tensor): The quadrotor state. Shape: (1, 21) or (1, 18)

    Returns:
        state_equiv (torch.Tensor): The equivariant state representation. Shape: (1, 20) or (1, 17)
    """
    # Quadrotor states:
    state_dim = state.size(1)
    if state_dim == 18: # when s = {x, v, R_vec, W}
        x, v, R_vec, W = state[0,0:3], state[0,3:6], state[0,6:15], state[0,15:18]
    elif state_dim == 23: # when s = {x, v, R_vec, W, eIx, eb1, eIb1}
        x, v, R_vec, W, eIx, eb1eIb1 = state[0,0:3], state[0,3:6], state[0,6:15], state[0,15:18], state[0,18:21], state[0,21:23]
    R = R_vec.reshape(3, 3).T 

    # Compute \theta and R_e3(theta)
    theta_x = -atan2(x[1], x[0]) 
    R_e3 = Rot_e3(theta_x)

    # Imaginary quadrotor states
    x_equiv = R_e3 @ x
    v_equiv = R_e3 @ v
    R_equiv = R_e3 @ R
    R_vec_equiv = R_equiv.T.reshape(9) # R.reshape(9, 1, order='F')
    if state_dim == 18: # when s = {x, v, R_vec, W}
        state_equiv = torch.cat((x_equiv[[0,2]], v_equiv, R_vec_equiv, W), dim=0)
        # state_equiv = torch.cat((x_equiv, v_equiv, R_vec_equiv, W), dim=0)
    elif state_dim == 23: # when s = {x, v, R_vec, W, eIx, eb1, eIb1}
        eIx_equiv = R_e3 @ eIx
        state_equiv = torch.cat((x_equiv[[0,2]], v_equiv, R_vec_equiv, W, eIx_equiv, eb1eIb1), dim=0)   

    return state_equiv


def get_equiv_state_batch(state):
    """
    Convert the input state to the equivariant state representation for a batch of states.

    Args:
        state (torch.Tensor): The input state. Shape: (N, 21) or (N, 18)

    Returns:
        state_equiv (torch.Tensor): The equivariant state representation. Shape: (N, 20) or (N, 17)
    """
    # Quadrotor states:
    if state.shape[1] == 18: # when s = {x, v, R_vec, W}
        x, v, R_vec, W = state[:,0:3], state[:,3:6], state[:,6:15], state[:,15:18]
    elif state.shape[1] == 23: # when s = {x, v, R_vec, W, eIx, eb1, eIb1}
        x, v, R_vec, W, eIx, eb1eIb1 = state[:,0:3], state[:,3:6], state[:,6:15], state[:,15:18], state[:,18:21], state[:,21:23]
    R = torch.transpose(torch.reshape(R_vec, (R_vec.shape[0], 3, 3)), 1, 2)

    # Compute \theta and R_e3(theta)
    theta_x_batch = -atan2(x[:,1], x[:,0])
    R_e3_batch = Rot_e3_batch(theta_x_batch)

    # Imaginary quadrotor states
    x_equiv = torch.einsum('ijk,ik->ij', R_e3_batch, x)
    v_equiv = torch.einsum('ijk,ik->ij', R_e3_batch, v)
    R_equiv = R_e3_batch @ R
    R_vec_equiv = torch.reshape(torch.transpose(R_equiv, 1, 2), (R_vec.shape[0], 9))
    eIx_equiv = torch.einsum('ijk,ik->ij', R_e3_batch, eIx)
    state_equiv = torch.cat((x_equiv[:, [0,2]], v_equiv, R_vec_equiv, W, eIx_equiv, eb1eIb1), dim=1)

    return state_equiv


def Rot_e3(theta):
    """
    Rotation matrix on e3 axis.

    Args:
        theta (torch.Tensor): The rotation angle in radians.

    Returns:
        A PyTorch tensor of shape (3, 3).
    """
    cos_theta = cos(theta)
    sin_theta = sin(theta)

    R = torch.zeros((3, 3), device=theta.device)
    R[0, 0], R[0, 1] = cos_theta, -sin_theta
    R[1, 0], R[1, 1] = sin_theta, cos_theta
    R[2, 2] = 1.

    return R

def Rot_e3_batch(theta_batch):
    """
    Rotation matrix on e3 axis.

    Args:
        theta (torch.Tensor): The rotation angle in radians. Shape: (N)

    Returns:
        A PyTorch tensor of shape (N, 3, 3) where N is the number of rotations (batch size).
    """
    cos_theta = cos(theta_batch)
    sin_theta = sin(theta_batch)
    zeros = zeros_like(theta_batch)

    return torch.stack(
          [torch.stack([cos_theta, -sin_theta, zeros], dim=1),
           torch.stack([sin_theta,  cos_theta, zeros], dim=1),
           torch.stack([zeros, zeros, ones_like(theta_batch)], dim=1)]
           , dim=1)