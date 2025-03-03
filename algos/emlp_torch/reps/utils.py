""" Utilities for computing equivariant representations and their subspaces"""
import math
import logging
from tqdm.auto import tqdm
import torch
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

_TORCH_DTYPES = {
    'float': torch.float,
    'float32': torch.float32,
    'float64': torch.float64,
    'double': torch.float64,
    'complex64': torch.complex64,
    'cfloat': torch.complex64,
    'complex128': torch.complex128,
    'cdouble': torch.complex128,
    'float16': torch.float16,
    'half': torch.float16,
    'bfloat16': torch.bfloat16,
    'uint8': torch.uint8,
    'int8': torch.int8,
    'int16': torch.int16,
    'short': torch.int16,
    'int32': torch.int32,
    'int': torch.int,
    'int64': torch.int64,
    'long': torch.int64,
    'bool': torch.bool,
}

def torch_dtype(dtype):
    """ Convert a string representation of a torch dtype to the actual
    torch dtype. """
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype in _TORCH_DTYPES:
        return _TORCH_DTYPES[dtype]
    return torch.empty(0).to(dtype).dtype


def dtype_cast(A, B):
    """ Casts A and B to the same dtype, preferring complex dtypes over real dtypes. """
    if A.dtype in (torch.complex64, torch.complex128):
        B = B.to(A.dtype)
    if B.dtype in (torch.complex64, torch.complex128):
        A = A.to(B.dtype)
    return A, B


def torch_device(device):
    """ Get the device from a string or torch.device """
    if device is None:
        device = 'cpu'
    return torch.empty(0, device=device).device


def device_cast(A, B):
    """ Casts A and B to the same device, preferring GPU over CPU. """
    if A.device.type == 'cuda':
        B = B.to(A.device)
    if B.device.type == 'cuda':
        A = A.to(B.device)
    return A, B


def get_dtype(operators, dtypes=None):
    """ Returns the dtype of the first operator that has a dtype attribute. """
    if dtypes is None:
        dtypes = []
    for obj in operators:
        if obj is not None and hasattr(obj, 'dtype'):
            dtypes.append(obj.dtype)
    return dtypes[0]


def get_device(operators, devices=None):
    """ Returns the device of the first operator that has a device attribute. """
    if devices is None:
        devices = []
    for obj in operators:
        if obj is not None and hasattr(obj, 'device') and obj.device.type != 'cpu':
            return obj.device
    return torch.device('cpu')


def orthogonal_complement(proj):
    """ Computes the orthogonal complement to a given matrix proj"""
    _, S, Vh = torch.linalg.svd(proj, full_matrices=True)
    rank = (S > 1e-5).sum()
    return Vh[rank:].conj().t()


def krylov_constraint_solve(C, tol=1e-5):
    """ Computes the solution basis Q for the linear constraint CQ=0  and QᵀQ=I
        up to specified tolerance with C expressed as a LinearOperator. """
    r = 5
    if C.size(0)*r*2 > 2e9:
        raise RuntimeError(f"Solns for contraints {C.shape} too large to fit in memory")
    found_rank = 5
    while found_rank == r:
        r *= 2  # Iterative doubling of rank until large enough to include the full solution space
        if C.size(0)*r > 2e9:
            logging.error("Hit memory limits, switching to "
                          "sample equivariant subspace of size %r", found_rank)
            break
        Q = krylov_constraint_solve_upto_r(C, r, tol)
        found_rank = Q.size(-1)
    return Q


def krylov_constraint_solve_upto_r(C, r, tol=1e-5, lr=1e-2):
    """ Iterative routine to compute the solution basis to the constraint CQ=0 and QᵀQ=I
        up to the rank r, with given tolerance. Uses gradient descent (+ momentum) on the
        objective |CQ|^2, which provably converges at an exponential rate."""
    W = torch.randn(C.size(-1), r, device=C.device) / math.sqrt(C.size(-1))  # if W0 is None else W0
    W.requires_grad = True
    opt = torch.optim.SGD([W], lr=lr, momentum=.9)

    def loss(W):
        # added absolute for complex support
        return (torch.abs(C@W)**2).sum()/2

    # setup progress bar
    pbar = tqdm(total=100, desc=f'Krylov Solving for Equivariant Subspace r<={r}', bar_format=
                "{l_bar}{bar}| {n:.3g}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")
    prog_val = 0

    lstart = loss(W).item()

    for i in range(20000):
        opt.zero_grad()
        lossval = loss(W)
        lossval.backward()
        opt.step()
        # update progress bar
        progress = max(100*math.log(lossval.item()/lstart)/math.log(tol**2/lstart)-prog_val, 0)
        progress = min(100-prog_val, progress)
        if progress > 0:
            prog_val += progress
            pbar.update(progress)

        if torch.sqrt(lossval) < tol:  # check convergence condition
            pbar.close()
            break  # has converged
        if lossval > 2e3 and i > 100:  # Solve diverged due to too high learning rate
            logging.warning("Constraint solving diverged, trying lower learning rate %.2e", lr/3)
            if lr < 1e-4:
                raise ConvergenceError(
                    f"Failed to converge even with smaller learning rate {lr:.2e}")
            return krylov_constraint_solve_upto_r(C, r, tol, lr=lr/3)
    else:
        raise ConvergenceError("Failed to converge.")

    W.requires_grad = False

    # Orthogonalize solution at the end
    U, S, _ = torch.linalg.svd(W, full_matrices=False)
    # Would like to do economy SVD here (to not have the unecessary O(n^2) memory cost)
    # but this is not supported in numpy (or Jax) unfortunately.
    rank = (S > 10*tol).sum()
    Q = U[:, :rank]
    # final_L
    final_L = loss(Q)
    if final_L > tol:
        logging.warning("Normalized basis has too high error %.2e for tol %.2e", final_L, tol)
    scutoff = (S[rank] if r > rank else 0)
    assert rank == 0 or scutoff < S[rank-1]/100, f"Singular value gap too small: {S[rank-1]:.2e} \
        above cutoff {scutoff:.2e} below cutoff. Final L {final_L:.2e}, earlier {S[rank-5:rank]}"
    return Q


class ConvergenceError(Exception):
    """ Convergence failure exception """


def sparsify_basis(Q, lr=1e-2):  # (n,r)
    """ Convenience function to attempt to sparsify a given basis by applying an orthogonal
        transformation W, Q' = QW where Q' has only 1s, 0s and -1s. Notably this method does not
        have the convergence guarantees of krylov_constraint_solve and can fail (even silently).
        Intended to be used only for visualization purposes, use at your own risk. """
    W = torch.randn(Q.size(-1), Q.size(-1), device=Q.device)
    W, _ = torch.linalg.qr(W)
    W = W.to(torch.float32)
    W.requires_grad = True

    opt = torch.optim.Adam([W], lr=lr)

    def loss(W):
        return torch.abs(Q@W.t()).mean() + \
            .1*(torch.abs(W.t()@W-torch.eye(W.size(0), device=W.device))).mean() + \
            .01*torch.linalg.slogdet(W)[1]**2

    for i in tqdm(range(3000), desc='sparsifying basis'):
        opt.zero_grad()
        lossval = loss(W)
        lossval.backward()
        opt.step()

        if lossval > 1e2 and i > 100:  # Solve diverged due to too high learning rate
            logging.warning("basis sparsification diverged, trying lower learning rate %.2e", lr/3)
            return sparsify_basis(Q, lr=lr/3)
    Q = (Q@W.t()).clone()
    Q[Q.abs() < 1e-2] = 0
    Q[Q.abs() > 1e-2] /= Q[Q.abs() > 1e-2].abs()
    A = Q@(1+torch.arange(Q.size(-1), device=Q.device)).to(torch.float)
    if len(torch.unique(A.abs())) != Q.size(-1)+1 and len(torch.unique(A.abs())) != Q.size(-1):
        logging.error("Basis elems did not separate: found only %r/%r",
                      len(torch.unique(A.abs())), Q.size(-1))
    return Q


def vis(repin, repout, cluster=True):
    """ A function to visualize the basis of equivariant maps repin>>repout
        as an image. Only use cluster=True if you know Pv will only have
        r distinct values (true for G<S(n) but not true for many continuous groups). """
    rep = repin >> repout
    P = rep.equivariant_projector()  # compute the equivariant basis
    Q = rep.equivariant_basis()
    v = torch.randn(P.size(1), device=P.device)  # sample random vector
    # project onto equivariant subspace (and round)
    v = torch.round(P@v, decimals=4).detach().cpu()
    if cluster:  # cluster nearby values for better color separation in plot
        # send to cpu
        v = KMeans(n_clusters=Q.size(-1)).fit(v.reshape(-1, 1)).labels_
    plt.imshow(v.reshape(repout.size(), repin.size()))
    plt.axis('off')
    plt.savefig(f'{rep}-torch.png', bbox_inches='tight')


def scale_adjusted_rel_error(t1, t2, g):
    """ Computes the relative error of t1 and t2, adjusted for the scale of t1 and t2 and g. """
    error = torch.sqrt(torch.mean(torch.abs(t1-t2)**2))
    tscale = torch.sqrt(torch.mean(torch.abs(t1)**2))+torch.sqrt(torch.mean(torch.abs(t2)**2))
    gscale = torch.sqrt(torch.mean(torch.abs(g-torch.eye(g.size(-1), device=t1.device))**2))
    scale = torch.max(tscale, gscale)
    return error/torch.clamp(scale, min=1e-7)


def equivariance_error(W, repin, repout, G):
    """ Computes the equivariance relative error rel_err(Wρ₁(g),ρ₂(g)W)
        of the matrix W (dim(repout),dim(repin)) [or basis Q: (dim(repout)xdim(repin), r)]
        according to the input and output representations and group G. """
    W = W.reshape(repout.size(), repin.size(), -1).transpose((2, 0, 1))[None]

    # Sample 5 group elements and verify the equivariance for each
    gs = G.samples(5)
    ring = torch.vmap(repin.rho_dense)(gs)[:, None]
    routg = torch.vmap(repout.rho_dense)(gs)[:, None]
    equiv_err = scale_adjusted_rel_error(W@ring, routg@W, gs)
    return equiv_err
