import torch
from algos.emlp_torch.nn import EMLPBlock, Linear

def approx_spectral_norm(W, device):
    """
    Approximate the spectral norm of a weight matrix W.
    
    Args:
        W (torch.Tensor): Weight matrix.
        device (torch.device): Device on which to perform computations.
    
    Returns:
        sigma1 (float): Estimated spectral norm.
    """
    n = W.shape[1]
    x = torch.normal(0., 1., (n,)).to(device)

    for _ in range(10):
        x = W.T @ W @ x
        x = x / torch.linalg.vector_norm(x)
    
    v1 = x
    sigma1 = torch.linalg.vector_norm(W @ v1)
    #u1 = (W@v1)/sigma1
    
    return sigma1

def spectral_norm(network, device):
    """
    Compute the spectral norm regularization term for a given network.
    
    Args:
        network (torch.nn.Sequential): Neural network containing layers.
        device (torch.device): Device on which to perform computations.
    
    Returns:
        reg_spectral (float): Sum of squared spectral norms of all layers.
    """
    reg_spectral = 0.
    for layer in network:
        if isinstance(layer, EMLPBlock):
            reg_spectral += approx_spectral_norm(layer.linear.weight, device)**2 + torch.norm(layer.bilinear.bi_params)**2
        elif isinstance(layer, Linear):
            reg_spectral += approx_spectral_norm(layer.weight, device)**2
    return reg_spectral
