""" Neural network modules for the EMLP PyTorch implementation. """
import math
import logging
from functools import lru_cache as cache
import torch
import torch.nn as nn  # pylint: disable=R0402:consider-using-from-import
import torch.nn.functional as F
from scipy.special import binom
import numpy as np
from .reps import T, Rep, Scalar, bilinear_weights, SumRep


class Linear(nn.Linear):
    """ Basic equivariant Linear layer from repin to repout."""

    def __init__(self, repin, repout):
        nin, nout = repin.size(), repout.size()
        super().__init__(nin, nout)
        torch.nn.init.orthogonal_(self.weight)
        torch.nn.init.uniform_(self.bias, 0., 1/math.sqrt(nout))
        rep_W = repout*repin.t()
        rep_bias = repout
        Pw = rep_W.equivariant_projector()
        Pb = rep_bias.equivariant_projector()

        def proj_b(b):
            return Pb@b
        def proj_w(w):
            return (Pw@w.reshape(-1)).reshape(nout, nin)
        self.proj_b = proj_b
        self.proj_w = proj_w
        logging.info("Linear W components:%r rep:%r", rep_W.size(), rep_W)

    def forward(self, x):  # (cin) -> (cout)
        """ Forward pass of the linear layer. """
        return F.linear(x, self.proj_w(self.weight), self.proj_b(self.bias))


class BiLinear(nn.Module):
    """ Cheap bilinear layer (adds parameters for each part of the input which can be
        interpreted as a linear map from a part of the input to the output representation)."""

    def __init__(self, repin, repout):
        super().__init__()
        Wdim, weight_proj = bilinear_weights(repout, repin)
        self.weight_proj = weight_proj
        self.bi_params = nn.Parameter(torch.randn(Wdim))
        logging.info("BiW components: dim:%r", Wdim)

    def forward(self, x):
        """ Forward pass of the bilinear layer. """
        # compatible with non sumreps? need to check
        W = self.weight_proj(self.bi_params, x)
        out = .1*(W@x[..., None])[..., 0]
        return out


def gated(ch_rep: Rep) -> Rep:
    """ Returns the rep with an additional scalar 'gate' for each of the nonscalars and non regular
        reps in the input. To be used as the output for linear (and or bilinear) layers directly
        before a :func:`GatedNonlinearity` to produce its scalar gates. """
    if isinstance(ch_rep, SumRep):
        return ch_rep+sum(Scalar(rep.G) for rep in ch_rep
                           if rep != Scalar and not rep.is_permutation)
    return ch_rep+Scalar(ch_rep.G) if not ch_rep.is_permutation else ch_rep


# TODO: add support for mixed tensors and non sumreps
class GatedNonlinearity(nn.Module):
    """ Gated nonlinearity. Requires input to have the additional gate scalars
        for every non regular and non scalar rep. Applies swish to regular and
        scalar reps. (Right now assumes rep is a SumRep)"""

    def __init__(self, rep):
        super().__init__()
        self.rep = rep

    def forward(self, values):
        """ Forward pass of the gated nonlinearity. """
        gate_scalars = values[..., gate_indices(self.rep)]
        activations = gate_scalars.sigmoid() * values[..., :self.rep.size()]
        return activations


class EMLPBlock(nn.Module):
    """ Basic building block of EMLP consisting of G-Linear, biLinear,
        and gated nonlinearity. """

    def __init__(self, rep_in, rep_out):
        super().__init__()
        self.linear = Linear(rep_in, gated(rep_out))
        self.bilinear = BiLinear(gated(rep_out), gated(rep_out))
        self.nonlinearity = GatedNonlinearity(rep_out)

    def forward(self, x):
        """ Forward pass of EMLP block. """
        lin = self.linear(x)
        preact = self.bilinear(lin)+lin
        return self.nonlinearity(preact)


def uniform_rep(ch, group):
    """ A heuristic method for allocating a given number of channels (ch)
        into tensor types. Attempts to distribute the channels evenly across
        the different tensor types. Useful for hands off layer construction.

        Args:
            ch (int): total number of channels
            group (Group): symmetry group

        Returns:
            SumRep: The direct sum representation with dim(V)=ch
        """
    d = group.d
    Ns = np.zeros((lambertW(ch, d)+1,), int)  # number of tensors of each rank
    while ch > 0:
        # compute the max rank tensor that can fit up to
        max_rank = lambertW(ch, d)
        Ns[:max_rank+1] += np.array([d**(max_rank-r)
                                    for r in range(max_rank+1)], dtype=int)
        ch -= (max_rank+1)*d**max_rank  # compute leftover channels
    sum_rep = sum(binomial_allocation(nr, r, group) for r, nr in enumerate(Ns))
    sum_rep, perm = sum_rep.canonicalize()
    return sum_rep


def lambertW(ch, d):
    """ Returns solution to x*d^x = ch rounded down."""
    max_rank = 0
    while (max_rank+1)*d**max_rank <= ch:
        max_rank += 1
    max_rank -= 1
    return max_rank


def binomial_allocation(N, rank, G):
    """ Allocates N of tensors of total rank r=(p+q) into
        T(k,r-k) for k=0,1,...,r to match the binomial distribution.
        For orthogonal representations there is no
        distinction between p and q, so this op is equivalent to N*T(rank)."""
    if N == 0:
        return 0
    n_binoms = N//(2**rank)
    n_leftover = N%(2**rank)
    even_split = sum(n_binoms*int(binom(rank, k))*T(k, rank-k, G)
                     for k in range(rank+1))
    ps = np.random.binomial(rank, .5, n_leftover)
    ragged = sum(T(int(p), rank-int(p), G) for p in ps)
    out = even_split+ragged
    return out


class EMLP(nn.Module):
    """ Equivariant MultiLayer Perceptron. 
        If the input ch argument is an int, uses the hands off uniform_rep heuristic.
        If the ch argument is a representation, uses this representation for the hidden layers.
        Individual layer representations can be set explicitly by using a list of ints or a list of
        representations, rather than use the same for each hidden layer.

        Args:
            rep_in (Rep): input representation
            rep_out (Rep): output representation
            group (Group): symmetry group
            ch (int or list[int] or Rep or list[Rep]): number of channels in the hidden layers
            num_layers (int): number of hidden layers

        Returns:
            Module: the EMLP objax module."""

    def __init__(self, rep_in, rep_out, group, ch=384, num_layers=3, device='cuda'):
        super().__init__()
        logging.info("Initing EMLP (PyTorch)")
        group = group.to(device)
        self.G = group
        self.rep_in = rep_in(group)
        self.rep_out = rep_out(group)
        # Parse ch as a single int, a sequence of ints, a single Rep, a sequence of Reps
        if isinstance(ch, int):
            middle_layers = num_layers*[uniform_rep(ch, group)]
        elif isinstance(ch, Rep):
            middle_layers = num_layers*[ch(group)]
        else:
            middle_layers = [(c(group) if isinstance(c, Rep)
                              else uniform_rep(c, group)) for c in ch]
        reps = [self.rep_in]+middle_layers
        self.network = nn.Sequential(
            *[EMLPBlock(rin, rout) for rin, rout in zip(reps, reps[1:])],
            Linear(reps[-1], self.rep_out))
        self.network.to(device)

    def forward(self, x):
        """ Forward pass of algos.emlp_torch. """
        return self.network(x)


class Linear_jax_init(nn.Linear):
    """ Linear layer for equivariant representations. """

    def __init__(self, cin, cout):
        super().__init__(cin, cout)
        nn.init.xavier_normal_(self.weight)
        nn.init.zeros_(self.bias)


def MLPBlock(cin, cout):
    """ Basic building block of MLP consisting of Linear, SiLU (Swish), and dropout. """
    return nn.Sequential(Linear_jax_init(cin, cout), nn.SiLU())


class MLP(nn.Module):
    """ Standard baseline MLP. Representations and group are used for shapes only. """

    def __init__(self, rep_in, rep_out, group, ch=384, num_layers=3, device='cuda'):
        super().__init__()
        group = group.to(device)
        self.G = group
        self.rep_in = rep_in(group)
        self.rep_out = rep_out(group)
        chs = [self.rep_in.size()] + num_layers*[ch]
        cout = self.rep_out.size()
        logging.info("Initing MLP")
        self.net = nn.Sequential(
            *[MLPBlock(cin, cout) for cin, cout in zip(chs, chs[1:])],
            Linear_jax_init(chs[-1], cout)
        )
        self.net.to(device)

    def forward(self, x):
        """ Forward pass of MLP. """
        y = self.net(x)
        return y


class Standardize(nn.Module):
    """ A convenience module to wrap a given module, normalize its input
        by some dataset x mean and std stats, and unnormalize its output by
        the dataset y mean and std stats. 

        Args:
            model (Module): model to wrap
            ds_stats ((μx,σx,μy,σy) or (μx,σx)): tuple of the normalization stats

        Returns:
            Module: Wrapped model with input normalization (and output unnormalization)"""

    def __init__(self, model, ds_stats):
        super().__init__()
        self.model = model
        self.ds_stats = ds_stats

    def forward(self, x):
        """ Standardize the input and unstandardize the output. """
        x = x.to(next(self.model.parameters()).device)
        if len(self.ds_stats) == 2:
            muin, sin = self.ds_stats
            return self.model((x-muin)/sin)
        muin, sin, muout, sout = self.ds_stats
        y = sout*self.model((x-muin)/sin)+muout
        return y


@cache(maxsize=None)
def gate_indices(ch_rep: Rep) -> torch.tensor:
    """ Indices for scalars, and also additional scalar gates
        added by gated(sumrep)"""
    channels = ch_rep.size()
    perm = ch_rep.perm
    indices = torch.arange(channels)

    if not isinstance(ch_rep, SumRep):  # If just a single rep, only one scalar at end
        return indices if ch_rep.is_permutation else torch.ones(ch_rep.size())*ch_rep.size()

    num_nonscalars = 0
    i = 0
    for rep in ch_rep:
        if rep != Scalar and not rep.is_permutation:
            indices[perm[i:i+rep.size()]] = channels+num_nonscalars
            num_nonscalars += 1
        i += rep.size()
    return indices
