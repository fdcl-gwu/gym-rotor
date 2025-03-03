""" Abstract linear algebra library. """
from functools import reduce
import torch

from .linear_operator_base import LinearOperator, Lazy
from .utils import dtype_cast, device_cast, get_device


def product(c):
    """ Product of a list of numbers. """
    return reduce(lambda a, b: a*b, c)


def lazify(x):
    """ Convert a tensor LinearOperator. """
    if isinstance(x, LinearOperator):
        return x
    if torch.is_tensor(x):
        return Lazy(x)
    raise NotImplementedError


def densify(x):
    """ Convert a LinearOperator to a dense tensor. """
    if isinstance(x, LinearOperator):
        return x.to_dense()
    if torch.is_tensor(x):
        return x
    raise NotImplementedError


class I(LinearOperator):
    """ Identity operator. """

    def __init__(self, d, device=None):
        super().__init__()
        shape = (d, d)
        self.init(None, shape, device)

    def _matmat(self, V):  # (c,k)
        return V

    def _matvec(self, v):
        return v

    def _adjoint(self):
        return self

    def invt(self):
        return self


class LazyKron(LinearOperator):
    """ Lazy tensor product. """

    def __init__(self, Ms):
        super().__init__()
        self.Ms = Ms
        shape = product([Mi.size(0) for Mi in Ms]), product([Mi.size(1) for Mi in Ms])
        device = get_device(Ms)
        self.init(None, shape, device)
        self.to(self.device)

    def _matvec(self, v):
        return self._matmat(v).reshape(-1)

    def _matmat(self, V):
        eV = V.reshape(*[Mi.size(-1) for Mi in self.Ms], -1)
        for i, M in enumerate(self.Ms):
            eV_front = torch.movedim(eV, i, 0)
            MeV_front = (M@eV_front.reshape(M.size(-1), -1)).reshape(M.size(0), *eV_front.shape[1:])
            eV = torch.movedim(MeV_front, 0, i)
        return eV.reshape(self.size(0), eV.size(-1))

    def _adjoint(self):
        return LazyKron([Mi.t() for Mi in self.Ms])

    def invt(self):
        return LazyKron([M.invt() for M in self.Ms])

    def to_dense(self):
        self.to(self.device)
        Ms = [M.to_dense() if isinstance(M, LinearOperator) else M for M in self.Ms]
        return reduce(torch.kron, Ms)

    def __new__(cls, Ms):
        if len(Ms) == 1:
            return Ms[0]
        return super().__new__(cls)

    def to(self, device):
        self.Ms = [M.to(device) for M in self.Ms]
        self.device = torch.empty(0).to(device).device
        return self


def kronsum(A, B):
    """ Tensor sum. """
    A = A.contiguous()
    B = B.contiguous()
    A, B = device_cast(A, B)
    A, B = dtype_cast(A, B)
    return torch.kron(A, torch.eye(B.size(-1), device=A.device)) + \
        torch.kron(torch.eye(A.size(-1), device=A.device), B)


class LazyKronsum(LinearOperator):
    """ Lazy tensor sum. """

    def __init__(self, Ms):
        super().__init__()
        self.Ms = Ms
        shape = product([Mi.size(0) for Mi in Ms]), product([Mi.size(1) for Mi in Ms])
        dtype = torch.float
        device = get_device(Ms)
        self.init(dtype, shape, device)
        self.to(self.device)

    def _matvec(self, v):
        return self._matmat(v).reshape(-1)

    def _matmat(self, V):
        eV = V.reshape(*[Mi.size(-1) for Mi in self.Ms], -1)
        out = 0*eV
        for i, M in enumerate(self.Ms):
            eV_front = torch.movedim(eV, i, 0)
            M, eV_front = dtype_cast(M, eV_front)
            MeV_front = (M@eV_front.reshape(M.size(-1), -1)).reshape(M.size(0), *eV_front.shape[1:])
            out, MeV_front = dtype_cast(out, MeV_front)
            out += torch.movedim(MeV_front, 0, i)
        return out.reshape(self.size(0), eV.size(-1))

    def _adjoint(self):
        return LazyKronsum([Mi.t() for Mi in self.Ms])

    def to_dense(self):
        Ms = [M.to_dense() if isinstance(M, LinearOperator) else M for M in self.Ms]
        return reduce(kronsum, Ms)

    def __new__(cls, Ms):
        if len(Ms) == 1:
            return Ms[0]
        return super().__new__(cls)

    # could also be implemented as follows,
    # but fusing the sum into a single linearOperator is faster
    # def lazy_kronsum(Ms):
    #     n = len(Ms)
    #     lprod = np.cumprod([1]+[mi.size(-1) for mi in Ms])
    #     rprod = np.cumprod([1]+[mi.size(-1) for mi in reversed(Ms)])[::-1]
    #     return reduce(lambda a,b: a+b,[lazy_kron([I(lprod[i]),Mi,I(rprod[i+1])])
    #                                    for i,Mi in enumerate(Ms)])

    def to(self, device):
        self.Ms = [M.to(device) for M in self.Ms]
        self.device = torch.empty(0).to(device).device
        return self


class LazyJVP(LinearOperator):
    """ Lazy Jacobian-vector product. """

    def __init__(self, operator_fn, X, TX):
        super().__init__()
        self.operator_fn = operator_fn
        self.X = X
        self.TX = TX
        self.init(torch.float, operator_fn(X).shape, X.device)
        self.to(self.device)

    def vjp(self, v):
        """ Computes the vector-Jacobian product """
        return torch.autograd.functional.jvp(
            lambda x: self.operator_fn(x)@v, [self.X], [self.TX])[1]

    def vjp_T(self, v):
        """ Computes the vector-Jacobian product """
        return torch.autograd.functional.jvp(
            lambda x: self.operator_fn(x).t()@v, [self.X], [self.TX])[1]

    def _matmat(self, V):
        return self.vjp(V)

    def _matvec(self, v):
        return self.vjp(v)

    def _rmatmat(self, V):
        return self.vjp_T(V)

    def to(self, device):
        self.X = self.X.to(device)
        self.TX = self.TX.to(device)
        self.device = self.X.device
        return self


class ConcatLazy(LinearOperator):
    """ Produces a linear operator equivalent to concatenating
        a collection of matrices Ms along axis=0 """

    def __init__(self, Ms):
        super().__init__()
        self.Ms = Ms
        assert all(M.size(0) == Ms[0].size(0) for M in Ms),\
            f"Trying to concatenate matrices of different sizes {[M.shape for M in Ms]}"
        shape = (sum(M.size(0) for M in Ms), Ms[0].size(1))
        device = get_device(Ms)
        self.init(None, shape, device)
        self.to(self.device)

    def _matmat(self, V):
        return torch.cat([M@V for M in self.Ms])

    def _rmatmat(self, V):
        Vs = torch.chunk(V, len(self.Ms))
        return sum(Mi.t()@Vi for Mi, Vi in zip(self.Ms, Vs))

    def to_dense(self):
        dense_Ms = [M.to_dense() if isinstance(M, LinearOperator) else M for M in self.Ms]
        return torch.cat(dense_Ms)

    def to(self, device):
        self.Ms = [M.to(device) for M in self.Ms]
        self.device = torch.empty(0).to(device).device
        return self


class LazyDirectSum(LinearOperator):
    """ Lazy direct sum. """

    def __init__(self, Ms, multiplicities=None):
        super().__init__()
        self.Ms = Ms
        self.multiplicities = [1 for _ in Ms] if multiplicities is None else multiplicities
        shape = (sum(Mi.size(0)*c for Mi, c in zip(Ms, multiplicities)),
                 sum(Mi.size(0)*c for Mi, c in zip(Ms, multiplicities)))
        device = get_device(Ms)
        self.init(None, shape, device)
        self.to(self.device)

    def _matvec(self, v):
        return lazy_direct_matmat(v, self.Ms, self.multiplicities)

    def _matmat(self, V):  # (n,k)
        return lazy_direct_matmat(V, self.Ms, self.multiplicities)

    def _adjoint(self):
        return LazyDirectSum([Mi.t() for Mi in self.Ms])

    def invt(self):
        return LazyDirectSum([M.invt() for M in self.Ms])

    def to_dense(self):
        Ms_all = [M for M, c in zip(self.Ms, self.multiplicities)
                  for _ in range(c)]
        Ms_all = [Mi.to_dense() if isinstance(Mi, LinearOperator)
                  else Mi for Mi in Ms_all]
        return torch.block_diag(*Ms_all)

    def to(self, device):
        self.Ms = [M.to(device) for M in self.Ms]
        self.device = torch.empty(0).to(device).device
        return self


def lazy_direct_matmat(v, Ms, mults):
    """ Computes the matrix-vector product of a direct sum of matrices
        with a vector. """
    k = v.size(1) if len(v.shape) > 1 else 1
    i = 0
    y = []
    for M, multiplicity in zip(Ms, mults):
        i_end = i+multiplicity*M.size(-1)
        elems = M@v[i:i_end][None].reshape(k*multiplicity, M.size(-1)).t()
        y.append(elems.t().reshape(k, multiplicity*M.size(0)).t())
        i = i_end
    y = torch.cat(y)  # concatenate over rep axis
    return y


class LazyPerm(LinearOperator):
    """ Lazy permutation. """

    def __init__(self, perm):
        super().__init__()
        self.perm = perm
        shape = (len(perm), len(perm))
        self.init(None, shape, perm.device)

    def _matmat(self, V):
        return V[self.perm]

    def _matvec(self, v):
        return v[self.perm]

    def _adjoint(self):
        return LazyPerm(torch.argsort(self.perm))

    def invt(self):
        return self

    def to(self, device):
        self.perm = self.perm.to(device)
        self.device = self.perm.device
        return self


class LazyShift(LinearOperator):
    """ Lazy shift. """

    def __init__(self, n, k=1, device=None):
        super().__init__()
        self.k = k
        shape = (n, n)
        self.init(None, shape, device)

    def _matmat(self, V):  # (c,k) #Still needs to be tested??
        return torch.roll(V, self.k, dims=0)

    def _matvec(self, v):
        return torch.roll(v, self.k, dims=0)

    def _adjoint(self):
        return LazyShift(self.size(0), -self.k, self.device)

    def invt(self):
        return self


class SwapMatrix(LinearOperator):
    """ Swap rows. """

    def __init__(self, swaprows, n):
        super().__init__()
        self.swaprows = swaprows
        shape = (n, n)
        self.init(None, shape, swaprows.device)

    def _matmat(self, V):  # (c,k)
        V_new = V.clone()
        V_new[self.swaprows].copy_(V[self.swaprows[::-1]])
        return V_new

    def _matvec(self, v):
        return self._matmat(v)

    def _adjoint(self):
        return self

    def invt(self):
        return self


class Rot90(LinearOperator):
    """ Matrix rotation. """

    def __init__(self, n, k, device=None):
        super().__init__()
        shape = (n*n, n*n)
        self.n = n
        self.k = k
        self.init(None, shape, device)

    def _matmat(self, V):  # (c,k)
        return torch.rot90(V.reshape((self.n, self.n, -1)), self.k).reshape(V.shape)

    def _matvec(self, v):
        return torch.rot90(v.reshape((self.n, self.n, -1)), self.k).reshape(v.shape)

    def invt(self):
        return self
