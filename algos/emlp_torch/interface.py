""" Equivariant probabilistic IO interface. """
import torch
from torch import nn

from .reps import Vector
from .groups import O
from .nn import EMLP


class GroupAugmentation(nn.Module):
    """ Group equivariant data augmentation """
    def __init__(self, network, rep_in, rep_out, group, n_samples, test_aug, test_n_samples):
        super().__init__()
        self.G = group
        self.rep_in = rep_in(group)
        self.rep_out = rep_out(group)
        self.rho_in = torch.vmap(lambda g: self.rep_in.rho(g).to_dense())
        self.rho_out = torch.vmap(lambda g: self.rep_out.rho(g).to_dense())
        self.model = network
        self.n_samples = n_samples
        self.test_aug = test_aug
        self.test_n_samples = test_n_samples

    def symmetrized_model(self, x, gs):
        """ Group equivariant data augmentation """
        rhout_inv = torch.linalg.inv(self.rho_out(gs))
        return (rhout_inv@self.model((self.rho_in(gs)@x[..., None])[..., 0])[..., None])[..., 0]

    def forward(self, x):
        """ Group equivariant data augmentation """
        if self.model.training or self.test_aug:
            n_samples = self.n_samples if self.model.training else self.test_n_samples
            if n_samples > 1:
                x = x[None, ...].repeat(n_samples, 1, 1).reshape(-1, *x.shape[1:])
                gs = self.G.samples(x.size(0))
                output = self.symmetrized_model(x, gs)
                return output.reshape(n_samples, -1, *output.shape[1:]).mean(0)
            gs = self.G.samples(x.size(0))
            return self.symmetrized_model(x, gs)
        return self.model(x)


def batched_projection(bu, bv):
    """ Batched projection of bv onto bu. """
    return (bv*bu).sum(-1,keepdim=True)/(bu*bu).sum(-1,keepdim=True)*bu


def batched_gram_schmidt(bvv):
    """ Batched Gram-Schmidt orthogonalization. """
    nk = bvv.size(1)
    buu = torch.zeros_like(bvv, device=bvv.device)
    buu[:, :, 0] = bvv[:, :, 0].clone()
    for k in range(1, nk):
        bvk = bvv[:, k].clone()
        buk = 0
        for j in range(0, k):
            buj = buu[:, :, j].clone()
            buk = buk + batched_projection(buj, bvk)
        buu[:, :, k] = bvk - buk
    for k in range(nk):
        buk = buu[:, :, k].clone()
        buu[:, :, k] = buk / torch.linalg.vector_norm(buk, dim=-1, ord=2, keepdim=True)
    return buu


class Interface(nn.Module):
    """ Equivariant probabilistic IO interface. """
    def __init__(self, network, rep_in, rep_out, group,
                 n_samples, test_aug, test_n_samples, device):
        super().__init__()
        assert isinstance(group, O)
        self.G = group
        self.rep_in = rep_in(group)
        self.rep_out = rep_out(group)
        self.rho_in = torch.vmap(lambda g: self.rep_in.rho(g).to_dense())
        self.rho_out = torch.vmap(lambda g: self.rep_out.rho(g).to_dense())
        self.noise_scale = nn.Parameter(torch.ones(self.rep_in.size(), device=device))
        self.model = network
        self.io = EMLP(self.rep_in, group.d*Vector, group, ch=384, num_layers=1, device=device)
        self.n_samples = n_samples
        self.test_aug = test_aug
        self.test_n_samples = test_n_samples

    def samples(self, x):
        """ Sample from the interface. """
        z = torch.randn(self.rep_in.size(), device=x.device)
        out = self.io(x + self.noise_scale*z)
        # TODO: need thorougher check on whether transpose is needed
        out = out.view(x.size(0), self.G.d, self.G.d).permute(0, 2, 1)
        return batched_gram_schmidt(out)

    def symmetrized_model(self, x, gs):
        """ Group equivariant data augmentation """
        rhout_inv = torch.linalg.inv(self.rho_out(gs))
        return (rhout_inv@self.model((self.rho_in(gs)@x[..., None])[..., 0])[..., None])[..., 0]

    def forward(self, x):
        """ Group equivariant data augmentation """
        if self.model.training or self.test_aug:
            n_samples = self.n_samples if self.model.training else self.test_n_samples
            if n_samples > 1:
                x = x[None, ...].repeat(n_samples, 1, 1).reshape(-1, *x.shape[1:])
                gs = self.samples(x)
                output = self.symmetrized_model(x, gs)
                return output.reshape(n_samples, -1, *output.shape[1:]).mean(0)
            gs = self.samples(x)
            return self.symmetrized_model(x, gs)
        return self.model(x)
