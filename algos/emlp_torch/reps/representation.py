""" The base Representation class. """
import math
import logging
import itertools
from functools import lru_cache as cache, reduce
from collections import defaultdict
from plum import dispatch
import torch
from torch import nn

from ..groups import Group
from .linear_operator_base import LinearOperator
from .linear_operators import ConcatLazy, I, lazify, densify, LazyJVP, LazyPerm, \
    LazyDirectSum, LazyKron, LazyKronsum, lazy_direct_matmat, product
from .utils import orthogonal_complement, krylov_constraint_solve, get_device


class Rep(nn.Module):
    """ The base Representation class. Representation objects formalize the vector space V
       on which the group acts, the group representation matrix ρ(g), and the Lie Algebra
       representation dρ(A) in a single object. Representations act as types for vectors coming
       from V. These types can be manipulated and transformed with the built in operators
       ⊕,⊗,dual, as well as incorporating custom representations. Representation objects should
       be immutable.

       At minimum, new representations need to implement ``rho``, ``__str__``."""

    def __init__(self):
        super().__init__()
        self.is_permutation = False
        self._size = None
        self.G = None

    def rho(self, M):
        """ Group representation of the matrix M of shape (d,d)"""
        raise NotImplementedError

    def drho(self, A):
        """ Lie Algebra representation of the matrix A of shape (d,d)"""
        In = torch.eye(A.size(0), dtype=A.dtype, device=A.device)
        return LazyJVP(self.rho, In, A)

    def forward(self, G):
        """ Instantiate (nonconcrete) representation with a symmetry group (forward) """
        raise NotImplementedError

    def __str__(self):
        return repr(self)

    def __repr__(self):
        raise NotImplementedError

    def __eq__(self, other):
        if type(self) is not type(other):  # pylint: disable=unidiomatic-typecheck
            return False
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        raise NotImplementedError

    def size(self):
        """ Dimension dim(V) of the representation """
        if self._size is not None:
            return self._size
        if self.concrete() and isinstance(self.G, Group):
            self._size = self.rho(self.G.sample()).size(-1)
            return self._size
        raise NotImplementedError

    def canonicalize(self):
        """ An optional method to convert the representation into a canonical form
            in order to reuse equivalent solutions in the solver. Should return
            both the canonically ordered representation, along with a permutation
            which can be applied to vectors of the current representation to achieve
            that ordering. """
        # return canonicalized rep
        return self, torch.arange(self.size())

    def rho_dense(self, M):
        """ A convenience function which returns rho(M) as a dense matrix."""
        return densify(self.rho(M))

    def drho_dense(self, A):
        """ A convenience function which returns drho(A) as a dense matrix."""
        return densify(self.drho(A))

    def constraint_matrix(self):
        """ Constructs the equivariance constrant matrix (lazily) by concatenating
        the constraints (ρ(hᵢ)-I) for i=1,...M and dρ(Aₖ) for k=1,..,D from the generators
        of the symmetry group. """
        n = self.size()
        constraints = []
        constraints.extend([lazify(self.rho(h)).to(self.G.device)-I(n, device=self.G.device) \
                            for h in self.G.discrete_generators])
        constraints.extend([lazify(self.drho(A)).to(self.G.device) for A in self.G.lie_algebra])
        return ConcatLazy(constraints) if constraints else lazify(
            torch.zeros((1, n), device=self.G.device))

    solcache = {}

    def equivariant_basis(self):
        """ Computes the equivariant solution basis for the given representation of size N.
            Canonicalizes problems and caches solutions for reuse. Output [Q (N,r)] """
        if self == Scalar:
            return torch.ones((1, 1), device=self.G.device)
        canon_rep, perm = self.canonicalize()
        invperm = torch.argsort(perm)
        if canon_rep not in self.solcache:
            logging.info("%r cache miss", canon_rep)
            logging.info("Solving basis for %r%s", self,
                         f", for G={self.G}" if self.G is not None else "")
            C_lazy = canon_rep.constraint_matrix()
            if C_lazy.size(0)*C_lazy.size(1) > 3e7:  # Too large to use SVD
                result = krylov_constraint_solve(C_lazy)
            else:
                C_dense = C_lazy.to_dense()
                result = orthogonal_complement(C_dense)
            self.solcache[canon_rep] = result
        return self.solcache[canon_rep][invperm]

    def equivariant_projector(self):
        """ Computes the (lazy) projection matrix P=QQᵀ that projects to the equivariant basis."""
        Q = self.equivariant_basis()
        Q_lazy = lazify(Q)
        P = Q_lazy@Q_lazy.H()
        return P

    def concrete(self):
        """ Concreteness """
        return isinstance(self.G, Group)

    def __add__(self, other):
        """ Direct sum (⊕) of representations. """
        if isinstance(other, int):
            if other == 0:
                return self
            return self+other*Scalar
        if both_concrete(self, other):
            return SumRep(self, other)
        return DeferredSumRep(self, other)

    def __radd__(self, other):
        if isinstance(other, int):
            if other == 0:
                return self
            return other*Scalar+self
        return NotImplemented

    def __mul__(self, other):
        """ Tensor sum (⊗) of representations. """
        return mul_reps(self, other)

    def __rmul__(self, other):
        return mul_reps(other, self)

    def __pow__(self, other):
        """ Iterated tensor product. """
        assert isinstance(other, int), \
            f"Power only supported for integers, not {type(other)}"
        assert other >= 0, f"Negative powers {other} not supported"
        return reduce(lambda a, b: a*b, other*[self], Scalar)

    def __rshift__(self, other):
        """ Linear maps from self -> other """
        return other*self.t()

    def __lshift__(self, other):
        """ Linear maps from other -> self """
        return self*other.t()

    def __lt__(self, other):
        """ less than defined to disambiguate ordering multiple different representations.
            Canonical ordering is determined first by Group, then by size, then by hash"""
        if other == Scalar:
            return False
        try:
            if self.G < other.G:
                return True
            if self.G > other.G:
                return False
        except (AttributeError, TypeError):
            pass
        if self.size() < other.size():
            return True
        if self.size() > other.size():
            return False
        return hash(self) < hash(other)  # For sorting purposes only

    def t(self):
        """ Dual representation V*, rho*, drho*."""
        if isinstance(self.G, Group) and self.G.is_orthogonal:
            return self
        return Dual(self)


@dispatch
def mul_reps(ra, rb: int):
    """ Product of a scalar and a representation. """
    if rb == 1:
        return ra
    if rb == 0:
        return 0
    if ra.concrete():
        return SumRep(*(rb*[ra]))
    return DeferredSumRep(*(rb*[ra]))


@dispatch
def mul_reps(ra: int, rb):  # pylint: disable=function-redefined
    """ Product of a scalar and a representation. """
    return mul_reps(rb, ra)  # pylint: disable=W1114:arguments-out-of-order


class ScalarRep(Rep):
    """ The trivial representation of the group G. """
    def __init__(self, G=None):
        super().__init__()
        self.G = G
        self.is_permutation = True

    def forward(self, G):
        self.G = G
        return self

    def size(self):
        return 1

    def canonicalize(self):
        return self, torch.zeros(1, dtype=torch.long)

    def __repr__(self):
        return "V⁰"

    def t(self):
        return self

    def rho(self, M):
        return torch.eye(1, device=self.G.device)

    def drho(self, A):
        return 0*torch.eye(1, device=self.G.device)

    def __hash__(self):
        return 0

    def __eq__(self, other):
        return isinstance(other, ScalarRep)

    def __mul__(self, other):
        if isinstance(other, int):
            return super().__mul__(other)
        return other

    def __rmul__(self, other):
        if isinstance(other, int):
            return super().__rmul__(other)
        return other

    def concrete(self):
        return True


class Base(Rep):
    """ Base representation V of a group."""

    def __init__(self, G=None):
        super().__init__()
        self.G = G
        if G is not None:
            self.is_permutation = G.is_permutation

    def forward(self, G):
        return self.__class__(G)

    def rho(self, M):
        if isinstance(self.G, Group) and isinstance(M, dict):
            M = M[self.G]
        return M

    def drho(self, A):
        if isinstance(self.G, Group) and isinstance(A, dict):
            A = A[self.G]
        return A

    def size(self):
        assert self.G is not None, f"must know G to find size for rep={self}"
        return self.G.d

    def __repr__(self):
        return "V"

    def __hash__(self):
        return hash((type(self), self.G))

    def __eq__(self, other):
        return type(other) is type(self) and self.G == other.G

    def __lt__(self, other):
        if isinstance(other, Dual):
            return True
        return super().__lt__(other)


class Dual(Rep):
    """ Dual representation V*, rho*, drho*."""

    def __init__(self, rep):
        super().__init__()
        self.rep = rep
        self.G = rep.G
        if hasattr(rep, "is_permutation"):
            self.is_permutation = rep.is_permutation

    def forward(self, G):
        return self.rep(G).t()

    def rho(self, M):
        rho = self.rep.rho(M)
        rhoinvt = rho.invt() if isinstance(rho, LinearOperator) else torch.linalg.inv(rho).t()
        return rhoinvt

    def drho(self, A):
        return -self.rep.drho(A).t()

    def __repr__(self):
        return repr(self.rep)+"*"

    def t(self):
        return self.rep

    def __eq__(self, other):
        return type(other) is type(self) and self.rep == other.rep

    def __hash__(self):
        return hash((type(self), self.rep))

    def __lt__(self, other):
        if other == self.rep:
            return False
        return super().__lt__(other)

    def size(self):
        return self.rep.size()


# Alias V or Vector for an instance of the Base representation of a group
V = Vector = Base()

# An instance of the Scalar representation, equivalent to V**0
Scalar = ScalarRep()


def T(p, q=0, G=None):
    """ A convenience function for creating rank (p,q) tensors."""
    return (V**p*V.t()**q)(G)


def bilinear_weights(out_rep, in_rep):
    """ Bilinear weights for a linear operator from in_rep to out_rep. """
    # TODO: replace lazy_projection function with LazyDirectSum LinearOperator
    W_rep, W_perm = (in_rep >> out_rep).canonicalize()
    # TODO: possible bug when in_rep and out_rep are both non sumreps? investigate
    inv_perm = torch.argsort(W_perm)
    mat_shape = out_rep.size(), in_rep.size()
    x_rep = in_rep
    W_multiplicities = W_rep.reps
    x_multiplicities = x_rep.reps
    x_multiplicities = {rep: n for rep, n in x_multiplicities.items() if rep != Scalar}

    def nelems(nx, rep):
        return min(nx, rep.size())
    active_dims = sum(W_multiplicities.get(rep, 0)*nelems(n, rep)
                      for rep, n in x_multiplicities.items())
    reduced_indices_dict = {rep: ids[torch.randint(
        len(ids), size=(nelems(len(ids), rep),))].reshape(-1)
        for rep, ids in x_rep.as_dict(torch.arange(x_rep.size())).items()}
    # Apply the projections for each rank, concatenate, and permute back to orig rank order
    # (r,), (*c)
    # TODO: find out why backwards of this function is so slow
    def lazy_projection(params, x):
        bshape = x.shape[:-1]
        x = x.reshape(-1, x.size(-1))
        bs = x.size(0)
        i = 0
        Ws = []
        for rep, W_mult in W_multiplicities.items():
            if rep not in x_multiplicities:
                Ws.append(torch.zeros((bs, W_mult*rep.size()), device=x.device))
                continue
            x_mult = x_multiplicities[rep]
            n = nelems(x_mult, rep)
            i_end = i+W_mult*n
            bids = reduced_indices_dict[rep]
            bilinear_params = params[i:i_end].reshape(W_mult, n)  # bs,nK-> (nK,bs)
            i = i_end  # (bs,W_mult,d^r) = (W_mult,n)@(n,d^r,bs)
            bilinear_elems = bilinear_params@x[..., bids].t().reshape(n, rep.size()*bs)
            bilinear_elems = bilinear_elems.reshape(W_mult*rep.size(), bs).t()
            Ws.append(bilinear_elems)
        Ws = torch.cat(Ws, axis=-1)  # concatenate over rep axis
        # reorder to original rank ordering
        return Ws[..., inv_perm].reshape(*bshape, *mat_shape)
    return active_dims, lazy_projection


class SumRep(Rep):
    """ A sum of representations, e.g. V+V.T. """
    def __init__(self, *reps, extra_perm=None, skip_init=False):
        """ Constructs a tensor type based on a list of tensor ranks
            and possibly the symmetry generators gen."""
        super().__init__()
        if skip_init:
            return
        # Integers can be used as shorthand for scalars.
        reps = [SumRepFromCollection({Scalar: rep}) if isinstance(rep, int) else \
                rep for rep in reps]
        # Get reps and permutations
        reps, perms = zip(*[rep.canonicalize() for rep in reps])
        rep_counters = [rep.reps if isinstance(rep, SumRep) else {rep: 1} for rep in reps]
        # Combine reps and permutations: ∑_a + ∑_b = ∑_{a∪b}
        self.reps, perm = self.compute_canonical(rep_counters, perms)
        self.perm = extra_perm[perm] if extra_perm is not None else perm
        self.invperm = torch.argsort(self.perm)
        self.canonical = (self.perm == torch.arange(len(self.perm))).all()
        self.is_permutation = all(rep.is_permutation for rep in self.reps.keys())

    def size(self):
        return sum(rep.size()*count for rep, count in self.reps.items())

    def rho(self, M):
        rhos = [rep.rho(M) for rep in self.reps]
        multiplicities = self.reps.values()
        return LazyPerm(self.invperm)@LazyDirectSum(rhos, multiplicities)@LazyPerm(self.perm)

    def drho(self, A):
        drhos = [rep.drho(A) for rep in self.reps]
        multiplicities = self.reps.values()
        return LazyPerm(self.invperm)@LazyDirectSum(drhos, multiplicities)@LazyPerm(self.perm)

    def __eq__(self, other):
        return self.reps == other.reps and (self.perm == other.perm).all()

    def __hash__(self):
        # assert self.canonical
        return hash(tuple(self.reps.items()))

    def t(self):
        """ only swaps to adjoint representation, does not reorder elems"""
        return SumRep(*[rep.t() for rep, c in self.reps.items() for _ in range(c)],
                      extra_perm=self.perm)

    def __repr__(self):
        return "+".join(f"{count if count > 1 else ''}{repr(rep)}"
                        for rep, count in self.reps.items())

    def canonicalize(self):
        """Returns a canonically ordered rep with order np.arange(self.size()) and the
            permutation which achieves that ordering"""
        return SumRepFromCollection(self.reps), self.perm

    def forward(self, G):
        return SumRepFromCollection({rep(G): c for rep, c in self.reps.items()}, perm=self.perm)

    def concrete(self):
        return True

    def equivariant_basis(self):
        """ Overrides default implementation with a more efficient version
            which decomposes the constraints across the sum."""
        Qs = {rep: rep.equivariant_basis() for rep in self.reps}
        device = self.G.device if self.G is not None else get_device(list(Qs.values()))
        Qs = {rep: (Q.to(device).to(torch.float) if torch.is_tensor(Q) else Q) \
            for rep, Q in Qs.items()}
        active_dims = sum(self.reps[rep]*Qs[rep].size(-1) for rep in Qs.keys())
        multiplicities = self.reps.values()

        def lazy_Q(array):
            return lazy_direct_matmat(array, Qs.values(), multiplicities)[self.invperm]
        return LinearOperator(shape=(self.size(), active_dims), device=device,
                              matvec=lazy_Q, matmat=lazy_Q)

    def equivariant_projector(self):
        """ Overrides default implementation with a more efficient version
            which decomposes the constraints across the sum."""
        Ps = {rep: rep.equivariant_projector() for rep in self.reps}
        multiplicities = self.reps.values()

        def lazy_P(array):
            return lazy_direct_matmat(array[self.perm], Ps.values(), multiplicities)[self.invperm]
        device = self.G.device if self.G is not None else get_device(list(Ps.values()))
        return LinearOperator(shape=(self.size(), self.size()), device=device,
                              matvec=lazy_P, matmat=lazy_P)

    # TODO: investigate why these more idiomatic definitions with Lazy Operators end up slower
    # def equivariant_basis(self):
    #     Qs = [rep.equivariant_basis() for rep in self.reps]
    #     Qs = [(jax.device_put(Q.astype(np.float32)) if
    #           isinstance(Q,(np.ndarray)) else Q) for Q in Qs]
    #     multiplicities  = self.reps.values()
    #     return LazyPerm(self.invperm)@LazyDirectSum(Qs,multiplicities)
    # def equivariant_projector(self):
    #     Ps = [rep.equivariant_projector() for rep in self.reps]
    #     Ps = (jax.device_put(P.astype(np.float32)) if isinstance(P,(np.ndarray)) else P)
    #     multiplicities  = self.reps.values()
    #     return LazyPerm(self.invperm)@LazyDirectSum(Ps,multiplicities)@LazyPerm(self.perm)

    # Some additional SumRep specific methods to be used for internal purposes
    @staticmethod
    def compute_canonical(rep_cnters, rep_perms):
        """ given that rep1_perm and rep2_perm are the canonical orderings for
            rep1 and rep2 (ie v[rep1_perm] is in canonical order) computes
            the canonical order for rep1 + rep2"""
        # First: merge counters
        unique_reps = sorted(
            reduce(lambda a, b: a | b, [cnter.keys() for cnter in rep_cnters]))
        merged_cnt = defaultdict(int)
        permlist = []
        ids = [0]*len(rep_cnters)
        shifted_perms = []
        n = 0
        for perm in rep_perms:
            shifted_perms.append(n+perm)
            n += len(perm)
        for rep in unique_reps:
            for i, items in enumerate(zip(rep_cnters, shifted_perms)):
                rep_cnter, shifted_perm = items
                c = rep_cnter.get(rep, 0)
                permlist.append(shifted_perm[ids[i]:ids[i]+c*rep.size()])
                ids[i] += +c*rep.size()
                merged_cnt[rep] += c
        return dict(merged_cnt), torch.cat(permlist)

    def __iter__(self):  # not a great idea to use this method (ignores permutation ordering)
        return (rep for rep, c in self.reps.items() for _ in range(c))

    def __len__(self):
        return sum(multiplicity for multiplicity in self.reps.values())

    def as_dict(self, v):
        """ as dictionary """
        out_dict = {}
        i = 0
        for rep, c in self.reps.items():
            chunk = c*rep.size()
            out_dict[rep] = v[..., self.perm[i:i+chunk]].reshape(v.shape[:-1]+(c, rep.size()))
            i += chunk
        return out_dict


def both_concrete(rep1, rep2):
    """ Returns True if both reps are concrete, False otherwise """
    return all(rep.concrete() for rep in (rep1, rep2))


@dispatch.multi((SumRep, Rep), (Rep, SumRep), (SumRep, SumRep))
def mul_reps(ra, rb):  # pylint: disable=function-redefined
    """ Multiplication of two representations """
    if not both_concrete(ra, rb):
        return DeferredProductRep(ra, rb)
    return distribute_product([ra, rb])


@dispatch
def mul_reps(ra, rb):  # pylint: disable=function-redefined
    """ Multiplication of two representations """
    if isinstance(ra, ScalarRep):
        return rb
    if isinstance(rb, ScalarRep):
        return ra
    if not both_concrete(ra, rb):
        return DeferredProductRep(ra, rb)
    if isinstance(ra.G, Group) and isinstance(rb.G, Group) and ra.G == rb.G:
        return ProductRep(ra, rb)
    return DirectProduct(ra, rb)


class SumRepFromCollection(SumRep):
    """ A different constructor for SumRep """
    def __init__(self, counter, perm=None):
        super().__init__(skip_init=True)
        self.reps = counter
        self.perm = torch.arange(self.size()) if perm is None else perm
        self.reps, self.perm = self.compute_canonical([counter], [self.perm])
        self.invperm = torch.argsort(self.perm)
        self.canonical = (self.perm == torch.arange(len(self.perm))).all()
        self.is_permutation = all(rep.is_permutation for rep in self.reps.keys())


def distribute_product(reps, extra_perm=None):
    """ For expanding products of sums into sums of products, (ρ₁⊕ρ₂)⊗ρ₃ = (ρ₁⊗ρ₃)⊕(ρ₂⊗ρ₃).
        takes in a sequence of reps=[ρ₁,ρ₂,ρ₃,...] which are to be multiplied together and at
        least one of the reps is a SumRep, and distributes out the terms."""
    reps, perms = zip(*[repsum.canonicalize() for repsum in reps])
    reps = [rep if isinstance(rep, SumRep) else SumRepFromCollection({rep: 1}) for rep in reps]
    # compute axis_wise perm to canonical vector ordering along each axis
    axis_sizes = [len(perm) for perm in perms]

    order = torch.arange(product(axis_sizes)).reshape(tuple(len(perm) for perm in perms))
    for i, perm in enumerate(perms):
        order = torch.swapaxes(torch.swapaxes(order, 0, i)[perm, ...], 0, i)
    order = order.reshape(-1)
    # Compute permutation from multilinear map ordering -> vector ordering (decomposing the blocks)
    repsizes_all = []
    for rep in reps:
        this_rep_sizes = []
        for r, c in rep.reps.items():
            this_rep_sizes.extend([c*r.size()])
        repsizes_all.append(tuple(this_rep_sizes))
    block_perm = rep_permutation(tuple(repsizes_all))
    # must go from itertools product ordering to multiplicity grouped ordering
    ordered_reps = []
    each_perm = []
    i = 0
    for prod in itertools.product(*[rep.reps.items() for rep in reps]):
        rs, cs = zip(*prod)
        prod_rep, canonicalizing_perm = (product(cs)*reduce(lambda a, b: a*b, rs)).canonicalize()
        # print(f"{rs}:{cs} in distribute yield prod_rep {prod_rep}")
        ordered_reps.append(prod_rep)
        shape = []
        for r, c in prod:
            shape.extend([c, r.size()])
        axis_perm = torch.cat([2*torch.arange(len(prod)), 2*torch.arange(len(prod))+1])
        mul_perm = torch.arange(len(canonicalizing_perm)).reshape(
            shape).permute(axis_perm.tolist()).reshape(-1)
        each_perm.append(mul_perm[canonicalizing_perm]+i)
        i += len(canonicalizing_perm)
    each_perm = torch.cat(each_perm)
    total_perm = order[block_perm[each_perm]]
    if extra_perm is not None:
        total_perm = extra_perm[total_perm]
    # TODO: could achieve additional reduction by canonicalizing at this step,
    # but unnecessary for now
    return SumRep(*ordered_reps, extra_perm=total_perm)


@cache(maxsize=None)
def rep_permutation(repsizes_all):
    """Permutation from block ordering to flattened ordering"""
    size_cumsums = [list(itertools.accumulate([0] + list(repsizes)))
                    for repsizes in repsizes_all]
    permutation = torch.zeros([cumsum[-1] for cumsum in size_cumsums],
                              dtype=torch.long)
    arange = torch.arange(math.prod(permutation.size()))
    indices_iter = itertools.product(*[range(len(repsizes)) for repsizes in repsizes_all])
    i = 0
    for indices in indices_iter:
        slices = tuple(slice(cumsum[idx], cumsum[idx + 1])
                       for idx, cumsum in zip(indices, size_cumsums))
        slice_lengths = [sl.stop - sl.start for sl in slices]
        chunk_size = math.prod(slice_lengths)
        permutation[slices] += arange[i:i + chunk_size].reshape(*slice_lengths)
        i += chunk_size
    return torch.argsort(permutation.reshape(-1))


class ProductRep(Rep):
    """ A product of representations """
    def __init__(self, *reps, extra_perm=None, counter=None, skip_init=False):
        """ Initialize the product representation """
        super().__init__()
        if skip_init:
            return
        # Two variants of the constructor:
        if counter is not None:  # one with counter specified directly
            self.reps = counter
            self.reps, self.perm = self.compute_canonical(
                [counter], [torch.arange(self.size()) if extra_perm is None else extra_perm])
        else:  # other with list
            # Get reps and permutations
            reps, perms = zip(*[rep.canonicalize() for rep in reps])
            rep_counters = [rep.reps if isinstance(rep, ProductRep) else {rep: 1} for rep in reps]
            # Combine reps and permutations: ∏_a ⊗ ∏_b = ∏_{a ∪ b}
            self.reps, perm = self.compute_canonical(rep_counters, perms)
            self.perm = extra_perm[perm] if extra_perm is not None else perm

        self.invperm = torch.argsort(self.perm)
        self.canonical = (self.perm == self.invperm).all()
        Gs = tuple(set(rep.G for rep in self.reps.keys()))
        assert len(Gs) == 1, f"Multiple different groups {Gs} in product rep {self}"
        self.G = Gs[0]
        self.is_permutation = all(rep.is_permutation for rep in self.reps.keys())

    def size(self):
        return product([rep.size()**count for rep, count in self.reps.items()])

    def rho(self, Ms, lazy=False):
        if isinstance(self.G, Group) and isinstance(Ms, dict):
            Ms = Ms[self.G]
        canonical_lazy = LazyKron(
            [rep.rho(Ms) for rep, c in self.reps.items() for _ in range(c)])
        return LazyPerm(self.invperm)@canonical_lazy@LazyPerm(self.perm)

    def drho(self, As):
        if isinstance(self.G, Group) and isinstance(As, dict):
            As = As[self.G]
        canonical_lazy = LazyKronsum(
            [rep.drho(As) for rep, c in self.reps.items() for _ in range(c)])
        return LazyPerm(self.invperm)@canonical_lazy@LazyPerm(self.perm)

    def __hash__(self):
        assert self.canonical, f"Not canonical {repr(self)}? perm {self.perm}"
        return hash(tuple(self.reps.items()))

    def __eq__(self, other):  # TODO: worry about non canonical?
        return isinstance(other, ProductRep) and \
            self.reps == other.reps and (self.perm == other.perm).all()

    def concrete(self):
        return True

    def t(self):
        return self.__class__(*[rep.t() for rep, c in self.reps.items()
                                for _ in range(c)], extra_perm=self.perm)

    def __repr__(self):
        superscript = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
        return "⊗".join([str(rep)+(f"{c}".translate(superscript) if c > 1 else "")
                         for rep, c in self.reps.items()])

    def canonicalize(self):
        """Returns a canonically ordered rep with order np.arange(self.size()) and the
            permutation which achieves that ordering"""
        return self.__class__(counter=self.reps), self.perm

    @staticmethod
    def compute_canonical(rep_cnters, rep_perms):
        """ given that rep1_perm and rep2_perm are the canonical orderings for
            rep1 and rep2 (ie v[rep1_perm] is in canonical order) computes
            the canonical order for rep1 * rep2"""
        order = torch.arange(product(len(perm) for perm in rep_perms))
        # First: merge counters
        unique_reps = sorted(
            reduce(lambda a, b: a | b, [cnter.keys() for cnter in rep_cnters]))
        merged_cnt = defaultdict(int)
        # Reshape like the tensor it is
        order = order.reshape(tuple(len(perm) for perm in rep_perms))
        # apply the canonicalizing permutations along each axis
        for i, perm in enumerate(rep_perms):
            order = torch.moveaxis(torch.moveaxis(order, i, 0)[perm, ...], 0, i)
        # sort the axes by canonical ordering
        # get original axis ids
        axis_ids = []
        n = 0
        for cnter in rep_cnters:
            axis_idsi = {}
            for rep, c in cnter.items():
                axis_idsi[rep] = n+torch.arange(c)
                n += c
            axis_ids.append(axis_idsi)
        axes_perm = []
        for rep in unique_reps:
            for i in range(len(rep_perms)):
                c = rep_cnters[i].get(rep, 0)
                if c != 0:
                    axes_perm.append(axis_ids[i][rep])
                    merged_cnt[rep] += c
        axes_perm = torch.cat(axes_perm)
        # reshaped but with inner axes within a collection explicitly expanded
        order = order.reshape(tuple(rep.size() for cnter in rep_cnters for \
                                    rep, c in cnter.items() for _ in range(c)))
        final_order = torch.permute(order, tuple(axes_perm))
        return dict(merged_cnt), final_order.reshape(-1)


class DirectProduct(ProductRep):
    """ Tensor product of representations ρ₁⊗ρ₂, but where the sub representations
        ρ₁ and ρ₂ are representations of distinct groups (ie ρ₁⊗ρ₂ is a representation
        of the direct product of groups G=G₁×G₂). As a result, the solutions for the two
        sub representations can be solved independently and assembled together with the
        kronecker product: Q = Q₁⊗Q₂ and P = P₁⊗P₂"""

    def __init__(self, *reps, counter=None, extra_perm=None):
        super().__init__(skip_init=True)
        # Two variants of the constructor:
        if counter is not None:  # one with counter specified directly
            self.reps = counter
            self.reps, perm = self.compute_canonical([counter], [torch.arange(self.size())])
            self.perm = extra_perm[perm] if extra_perm is not None else perm
        else:  # other with list
            reps, perms = zip(*[rep.canonicalize() for rep in reps])
            # print([type(rep) for rep in reps],type(rep1),type(rep2))
            rep_counters = [rep.reps if isinstance(rep, DirectProduct) else {rep: 1}
                            for rep in reps]
            # Combine reps and permutations: Pi_a + Pi_b = Pi_{a x b}
            reps, perm = self.compute_canonical(rep_counters, perms)
            # print("dprod init",self.reps)
            group_dict = defaultdict(lambda: 1)
            for rep, c in reps.items():
                group_dict[rep.G] = group_dict[rep.G]*rep**c
            sub_products = {rep: 1 for G, rep in group_dict.items()}
            self.reps = counter = sub_products
            self.reps, perm2 = self.compute_canonical([counter], [torch.arange(self.size())])
            self.perm = extra_perm[perm[perm2]] if extra_perm is not None else perm[perm2]
        self.invperm = torch.argsort(self.perm)
        self.canonical = (self.perm == self.invperm).all()
        self.is_permutation = all(rep.is_permutation for rep in self.reps.keys())
        assert all(count == 1 for count in self.reps.values())

    def equivariant_basis(self):
        canon_Q = LazyKron([rep.equivariant_basis() for rep, _ in self.reps.items()])
        invperm = self.invperm.to(canon_Q.device)
        return LazyPerm(invperm)@canon_Q

    def equivariant_projector(self):
        canon_P = LazyKron([rep.equivariant_projector() for rep, _ in self.reps.items()])
        invperm = self.invperm.to(canon_P.device)
        perm = self.perm.to(canon_P.device)
        return LazyPerm(invperm)@canon_P@LazyPerm(perm)

    def rho(self, Ms):
        canonical_lazy = LazyKron(
            [rep.rho(Ms) for rep, c in self.reps.items() for _ in range(c)])
        invperm = self.invperm.to(canonical_lazy.device)
        perm = self.perm.to(canonical_lazy.device)
        return LazyPerm(invperm)@canonical_lazy@LazyPerm(perm)

    def drho(self, As):
        canonical_lazy = LazyKronsum(
            [rep.drho(As) for rep, c in self.reps.items() for _ in range(c)])
        invperm = self.invperm.to(canonical_lazy.device)
        perm = self.perm.to(canonical_lazy.device)
        return LazyPerm(invperm)@canonical_lazy@LazyPerm(perm)

    def __repr__(self):
        return "⊗".join([repr(rep)+f"_{rep.G}" for rep, c in self.reps.items()])


class DeferredSumRep(Rep):
    """ A representation that is a sum of other representations, but where
    the sum is not evaluated until the representation is called with a group G. """
    def __init__(self, *reps):
        super().__init__()
        self.to_sum = []
        for rep in reps:
            self.to_sum.extend(rep.to_sum if isinstance(
                rep, DeferredSumRep) else [rep])

    def forward(self, G):
        if G is None:
            return self
        return SumRep(*[rep(G) for rep in self.to_sum])

    def __hash__(self):
        return hash((type(self), tuple(self.to_sum)))

    def __repr__(self):
        return '('+"+".join(f"{rep}" for rep in self.to_sum)+')'

    def t(self):
        return DeferredSumRep(*[rep.t() for rep in self.to_sum])

    def concrete(self):
        return False


class DeferredProductRep(Rep):
    """ A representation that is a product of other representations, but where
    the product is not evaluated until the representation is called with a group G. """
    def __init__(self, *reps):
        super().__init__()
        self.to_prod = []
        for rep in reps:
            assert not isinstance(rep, ProductRep)
            self.to_prod.extend(rep.to_prod if isinstance(
                rep, DeferredProductRep) else [rep])

    def forward(self, G):
        if G is None:
            return self
        return reduce(lambda a, b: a*b, [rep(G) for rep in self.to_prod])

    def __hash__(self):
        return hash((type(self), tuple(self.to_prod)))

    def __repr__(self):
        return "⊗".join(f"{rep}" for rep in self.to_prod)

    def t(self):
        return DeferredProductRep(*[rep.t() for rep in self.to_prod])

    def concrete(self):
        return False
