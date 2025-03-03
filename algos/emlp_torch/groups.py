""" EMLP PyTorch Implementation """
import torch
from torch import nn

from .reps import LazyShift, Rot90, LazyKron, LazyKronsum, LazyPerm, I, dtype_cast


MAX_POWER = 5  # Maximum power of the matrix exponential to compute


def rel_err(A, B):
    """ Relative error between two tensors """
    assert A.ndim == B.ndim
    return torch.mean(torch.abs(A-B))/(torch.mean(torch.abs(A))+torch.mean(torch.abs(B))+1e-6)


class Group(nn.Module):
    """ Abstract Group Object which new groups should inherit from. """

    def __init__(self):
        super().__init__()
        self.lie_algebra = NotImplemented  # The continuous generators
        self.discrete_generators = NotImplemented  # The discrete generators
        self.z_scale = None  # For scale noise for sampling elements
        self.is_orthogonal = None
        self.is_permutation = None
        self.d = NotImplemented  # The dimension of the base representation
        self.device = torch.device('cpu')
        self.args = None

    def init(self, *args):
        """ Initialize the group object. """
        # get the dimension of the base group representation
        if self.d is NotImplemented:
            if (self.lie_algebra is not NotImplemented) and \
                len(self.lie_algebra) > 0:
                self.d = self.lie_algebra[0].size(-1)
            if (self.discrete_generators is not NotImplemented) and \
                len(self.discrete_generators) > 0:
                self.d = self.discrete_generators[0].size(-1)

        self.args = args

        if self.lie_algebra is NotImplemented:
            self.lie_algebra = torch.zeros((0, self.d, self.d), device=self.device)
        if self.discrete_generators is NotImplemented:
            self.discrete_generators = torch.zeros((0, self.d, self.d), device=self.device)

        self.to(self.device)

        # set orthogonal flag automatically if not specified
        if self.is_permutation:
            self.is_orthogonal = True
        if self.is_orthogonal is None:
            self.is_orthogonal = True
            if len(self.lie_algebra) != 0:
                Id = torch.eye(self.d, device=self.device)
                A_dense = torch.stack([Ai@Id.to(Ai.dtype) for Ai in self.lie_algebra])
                self.is_orthogonal &= rel_err(-A_dense.transpose(2, 1), A_dense) < 1e-6
            if len(self.discrete_generators) != 0:
                Id = torch.eye(self.d, device=self.device)
                h_dense = torch.stack([hi@Id.to(hi.dtype) for hi in self.discrete_generators])
                self.is_orthogonal &= rel_err(h_dense.transpose(2, 1)@h_dense, Id[None]) < 1e-6

        # set regular flag automatically if not specified
        if self.is_orthogonal and (self.is_permutation is None):
            self.is_permutation = True
            # no infinitesmal generators and all rows have one 1
            self.is_permutation &= (len(self.lie_algebra) == 0)
            if len(self.discrete_generators) != 0:
                Id = torch.eye(self.d, device=self.device)
                h_dense = torch.stack([hi@Id.to(hi.dtype) for hi in self.discrete_generators])
                self.is_permutation &= (((h_dense-1).abs()<1e-6).long().sum(-1) == 1).all()

    def exp(self, A):
        """ Matrix exponential """
        return torch.linalg.matrix_exp(A)

    def num_constraints(self):
        """ Number of constraints to solve for the group """
        return len(self.lie_algebra)+len(self.discrete_generators)

    def sample(self):
        """Draw a sample from the group (not necessarily Haar measure)"""
        return self.samples(1)[0]

    def samples(self, N):
        """ Draw N samples from the group (not necessarily Haar measure)"""
        Id = torch.eye(self.d, device=self.device)
        A_dense = torch.stack([Ai@Id.to(Ai.dtype) for Ai in self.lie_algebra]) \
            if len(self.lie_algebra) \
            else torch.zeros((0, self.d, self.d), device=self.device)
        h_dense = torch.stack([hi@Id.to(hi.dtype) for hi in self.discrete_generators]) \
            if len(self.discrete_generators) \
            else torch.zeros((0, self.d, self.d), device=self.device)
        z = torch.randn(N, A_dense.size(0), device=self.device)
        if self.z_scale is not None:
            z *= self.z_scale
        k = torch.randint(-MAX_POWER, MAX_POWER+1, (N, h_dense.size(0), 3), device=self.device)
        return noise2samples(z, k, A_dense, h_dense)

    def check_valid_group_elems(self, g):
        """ Check that the group elements are valid """
        return True

    def __str__(self):
        return repr(self)

    def __repr__(self):
        outstr = f"{self.__class__}"
        if self.args:
            outstr += '('+''.join(repr(arg) for arg in self.args)+')'
        return outstr

    def __eq__(self, G2):  # TODO: more permissive by checking that spans are equal?
        return repr(self) == repr(G2)

    def __hash__(self):
        return hash(repr(self))

    def __lt__(self, other):
        """ For sorting purposes only """
        return hash(self) < hash(other)

    def __mul__(self, other):
        return DirectProduct(self, other)

    def forward(self):
        """ Forward method, unused. """
        return None

    def to(self, *args, **kwargs):
        """ Move the group to the specified device """
        if isinstance(self.lie_algebra, torch.Tensor):
            self.lie_algebra = self.lie_algebra.to(*args, **kwargs)
        elif isinstance(self.lie_algebra, list):
            self.lie_algebra = [Ai.to(*args, **kwargs) for Ai in self.lie_algebra]
        if isinstance(self.discrete_generators, torch.Tensor):
            self.discrete_generators = self.discrete_generators.to(*args, **kwargs)
        elif isinstance(self.discrete_generators, list):
            self.discrete_generators = [hi.to(*args, **kwargs) for hi in self.discrete_generators]
        if self.z_scale is not None:
            self.z_scale = self.z_scale.to(*args, **kwargs)
        self.device = torch.empty(0).to(*args, **kwargs).device
        return self


def matrix_power_simple(M, n):
    """ Compute the matrix power of M to the nth power without control flow """
    M = torch.where(n > 0, M, torch.linalg.inv(M))
    n = torch.where(n > 0, n, -n)
    Id = torch.eye(M.size(-1), device=M.device)
    out = Id
    for i in range(MAX_POWER):
        out = out @ torch.where(n > i, M, Id)
    return out


def noise2sample(z, ks, lie_algebra, discrete_generators):
    """ [z (D,)] [ks (M, K)] [lie_algebra (D, d, d)] [discrete_generators (M, d, d)] 
        Here K is the number of repeats for a given discrete generator. """
    g = torch.eye(lie_algebra.size(-1), device=lie_algebra.device)
    if len(lie_algebra):
        A = (z[:, None, None] * lie_algebra).sum(0)
        g, A = dtype_cast(g, A)
        g = g @ torch.linalg.matrix_exp(A)
    M, K = ks.size(0), ks.size(1)
    if M == 0:
        return g
    for k in torch.arange(K, device=ks.device):  # multiple rounds of discrete generators
        # randomize the order of generators
        for i in torch.randperm(M, device=ks.device):
            g = g @ matrix_power_simple(discrete_generators[i[None]][0], ks[i[None], k[None]][0])
    return g


def noise2samples(zs, ks, lie_algebra, discrete_generators):
    """ Convert noise to samples from the group """
    return torch.vmap(noise2sample, (0, 0, None, None), 0, 'different') \
        (zs, ks, lie_algebra, discrete_generators)


class Trivial(Group):
    """ The trivial group G={I} in n dimensions """

    def __init__(self, n):
        super().__init__()
        self.d = n
        self.init(n)

class Mirror(Group):
    """ The reflection symmetry group G={I, -I} in n dimensions """

    def __init__(self, n):
        super().__init__()
        self.d = n  # The dimension of the base representation
        self.discrete_generators = torch.zeros((1, self.d, self.d))
        self.discrete_generators[0, :, :] = -1.*torch.eye(self.d)
        self.init(n)
        
class SO(Group):
    """ The special orthogonal group SO(n) in n dimensions """

    def __init__(self, n):
        super().__init__()
        self.lie_algebra = torch.zeros(((n*(n-1))//2, n, n))
        k = 0
        for i in range(n):
            for j in range(i):
                self.lie_algebra[k, i, j] = 1
                self.lie_algebra[k, j, i] = -1
                k += 1
        self.init(n)


class O(SO):
    """ The Orthogonal group O(n) in n dimensions """

    def __init__(self, n):
        super().__init__(n)
        self.discrete_generators = torch.eye(n, device=self.device)[None]
        self.discrete_generators[0, 0, 0] = -1


class C(Group):
    """ The Cyclic group Ck in 2 dimensions """

    def __init__(self, k):
        super().__init__()
        theta = torch.tensor(2*torch.pi/k)
        self.discrete_generators = torch.zeros((1, 2, 2))
        self.discrete_generators[0, :, :] = torch.tensor(
            [[ torch.cos(theta), torch.sin(theta)],
             [-torch.sin(theta), torch.cos(theta)]])
        self.init(k)


class D(C):
    """ The Dihedral group Dk in 2 dimensions """

    def __init__(self, k):
        super().__init__(k)
        self.discrete_generators = torch.cat(
            [self.discrete_generators, torch.tensor([[[-1, 0], [0, 1]]])])


class Scaling(Group):
    """ The scaling group in n dimensions """

    def __init__(self, n):
        super().__init__()
        self.lie_algebra = torch.eye(n)[None]
        self.name = f"Scaling({n})"
        self.init(n)


class Parity(Group):
    """ The spacial parity group in 1+3 dimensions """

    def __init__(self):
        super().__init__()
        self.discrete_generators = -torch.eye(4)[None]
        self.discrete_generators[0, 0, 0] = 1
        self.init()


class TimeReversal(Group):
    """ The time reversal group in 1+3 dimensions """

    def __init__(self):
        super().__init__()
        self.discrete_generators = torch.eye(4)[None]
        self.discrete_generators[0, 0, 0] = -1
        self.init()


class SO13p(Group):
    """ The component of Lorentz group connected to identity """

    def __init__(self):
        super().__init__()
        self.lie_algebra = torch.zeros((6, 4, 4))
        self.lie_algebra[3:, 1:, 1:] = SO(3).lie_algebra
        for i in range(3):
            self.lie_algebra[i, 1+i, 0] = self.lie_algebra[i, 0, 1+i] = 1.
        # Adjust variance for samples along boost generators. For equivariance checks
        # the exps for high order tensors can get very large numbers
        self.z_scale = torch.tensor([.3, .3, .3, 1, 1, 1])  # can get rid of now
        self.init()


class SO13(SO13p):
    """ The Lorentz group in 1+3 dimensions """

    def __init__(self):
        super().__init__()
        self.discrete_generators = -torch.eye(4, device=self.device)[None]


class O13(SO13p):
    """ The full lorentz group (including Parity and Time reversal) """

    def __init__(self):
        super().__init__()
        self.discrete_generators = torch.eye(4, device=self.device)[None] + \
            torch.zeros((2, 1, 1), device=self.device)
        self.discrete_generators[0] *= -1
        self.discrete_generators[1, 0, 0] = -1


class Lorentz(O13):
    """ The full lorentz group (including Parity and Time reversal) """


class SO11p(Group):
    """ The identity component of O(1,1) (Lorentz group in 1+1 dimensions) """

    def __init__(self):
        super().__init__()
        self.lie_algebra = torch.tensor([[0., 1.], [1., 0.]])[None]
        self.init()


class O11(SO11p):
    """ The Lorentz group O(1,1) in 1+1 dimensions """

    def __init__(self):
        super().__init__()
        self.discrete_generators = torch.eye(2, device=self.device)[None] + \
            torch.zeros((2, 1, 1), device=self.device)
        self.discrete_generators[0] *= -1
        self.discrete_generators[1, 0, 0] = -1


class Sp(Group):
    """ Symplectic group Sp(m) in 2m dimensions (sometimes referred to as Sp(2m)) """

    def __init__(self, m):
        super().__init__()
        self.lie_algebra = torch.zeros((m*(2*m+1), 2*m, 2*m))
        k = 0
        for i in range(m):  # block diagonal elements
            for j in range(m):
                self.lie_algebra[k, i, j] = 1
                self.lie_algebra[k, m+j, m+i] = -1
                k += 1
        for i in range(m):
            for j in range(i+1):
                self.lie_algebra[k, m+i, j] = 1
                self.lie_algebra[k, m+j, i] = 1
                k += 1
                self.lie_algebra[k, i, m+j] = 1
                self.lie_algebra[k, j, m+i] = 1
                k += 1
        self.init(m)


class Z(Group):
    """ The cyclic group Z_n (discrete translation group) of order n.
       Features a regular base representation. """

    def __init__(self, n):
        super().__init__()
        self.n = n
        self.discrete_generators = [LazyShift(n)]
        self.init(n)


class S(Group):
    """ The permutation group S_n with an n dimensional regular representation. """

    def __init__(self, n):
        super().__init__()
        # Here we choose n-1 generators consisting of swaps between the first element
        # and every other element
        perms = torch.arange(n)[None] + torch.zeros((n-1, 1)).to(torch.int32)
        perms[:, 0] = torch.arange(1, n)
        perms[torch.arange(n-1), torch.arange(1, n)[None]] = 0
        self.discrete_generators = [LazyPerm(perm) for perm in perms]
        # We can also have chosen the 2 generator soln described in the paper, but
        # adding superflous extra generators surprisingly can sometimes actually *decrease*
        # the runtime of the iterative krylov solver by improving the conditioning
        # of the constraint matrix
        self.init(n)


class SL(Group):
    """ The special linear group SL(n) in n dimensions """

    def __init__(self, n):
        super().__init__()
        self.n = n
        self.lie_algebra = torch.zeros((n*n-1, n, n))
        k = 0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue  # handle diag elements separately
                self.lie_algebra[k, i, j] = 1
                k += 1
        for l in range(n-1):
            self.lie_algebra[k, l, l] = 1
            self.lie_algebra[k, -1, -1] = -1
            k += 1
        self.init(n)


class GL(Group):
    """ The general linear group GL(n) in n dimensions """

    def __init__(self, n):
        super().__init__()
        self.lie_algebra = torch.zeros((n*n, n, n))
        k = 0
        for i in range(n):
            for j in range(n):
                self.lie_algebra[k, i, j] = 1
                k += 1
        self.init(n)


class U(Group):  # Of dimension n^2
    """ The unitary group U(n) in n dimensions (complex) """

    def __init__(self, n):
        super().__init__()
        self.n = n
        lie_algebra_real = torch.zeros((n**2, n, n))
        lie_algebra_imag = torch.zeros((n**2, n, n))
        k = 0
        for i in range(n):
            for j in range(i):
                # antisymmetric real generators
                lie_algebra_real[k, i, j] = 1
                lie_algebra_real[k, j, i] = -1
                k += 1
                # symmetric imaginary generators
                lie_algebra_imag[k, i, j] = 1
                lie_algebra_imag[k, j, i] = 1
                k += 1
        for i in range(n):
            # diagonal imaginary generators
            lie_algebra_imag[k, i, i] = 1
            k += 1
        self.lie_algebra = lie_algebra_real + lie_algebra_imag*1j
        self.init(n)


class SU(Group):  # Of dimension n^2-1
    """ The special unitary group SU(n) in n dimensions (complex)"""

    def __init__(self, n):
        super().__init__()
        self.n = n
        if n == 1:
            # Trivial(1)
            self.d = 1
            self.init()
            return
        lie_algebra_real = torch.zeros((n**2-1, n, n))
        lie_algebra_imag = torch.zeros((n**2-1, n, n))
        k = 0
        for i in range(n):
            for j in range(i):
                # Antisymmetric real generators
                lie_algebra_real[k, i, j] = 1
                lie_algebra_real[k, j, i] = -1
                k += 1
                # symmetric imaginary generators
                lie_algebra_imag[k, i, j] = 1
                lie_algebra_imag[k, j, i] = 1
                k += 1
        for i in range(n-1):
            # diagonal traceless imaginary generators
            lie_algebra_imag[k, i, i] = 1
            for j in range(n):
                if i == j:
                    continue
                lie_algebra_imag[k, j, j] = -1/(n-1)
            k += 1
        self.lie_algebra = lie_algebra_real + lie_algebra_imag*1j
        self.init(n)


class Cube(Group):
    """ A discrete version of SO(3) including all 90 degree rotations in 3d space
    Implements a 6 dimensional representation on the faces of a cube"""

    def __init__(self):
        super().__init__()
        # order = np.arange(6) # []
        Fperm = torch.tensor([4, 1, 0, 3, 5, 2])
        Lperm = torch.tensor([3, 0, 2, 5, 4, 1])
        self.discrete_generators = [LazyPerm(perm) for perm in [Fperm, Lperm]]
        self.init()


def pad(permutation):
    """ Pad a permutation to 6x9 """
    assert len(permutation) == 48
    padded = torch.zeros((6, 9)).to(permutation.dtype)
    padded[:, :4] = permutation.reshape(6, 8)[:, :4]
    padded[:, 5:] = permutation.reshape(6, 8)[:, 4:]
    return padded


def unpad(padded_perm):
    """ Unpad a permutation from 6x9 to 48 """
    return torch.cat([padded_perm[:, :4], padded_perm[:, 5:]], -1).reshape(-1)


class RubiksCube(Group):  # 3x3 rubiks cube
    """ The Rubiks cube group G<S_48 consisting of all valid 3x3 Rubik's cube transformations.
       Generated by the a quarter turn about each of the faces."""

    def __init__(self):
        super().__init__()
        # Faces are ordered U,F,R,B,L,D (the net of the cube) # B
        order = torch.arange(48)  # L U R
        order_padded = pad(order)  # include a center element  # F
        # Compute permutation for Up quarter turn             # D
        order_padded[0, :] = torch.rot90(
            order_padded[0].reshape(3, 3), 1).reshape(9)  # Rotate top face
        FRBL = torch.tensor([1, 2, 3, 4])
        order_padded[FRBL, :3] = order_padded[
            torch.roll(FRBL, 1), :3]  # F <- L,R <- F,B <- R,L <- B
        Uperm = unpad(order_padded)
        # Now form all other generators by using full rotations of the cube
        # by 90 clockwise about a given face
        # rotate full cube so that Left face becomes Up, Up becomes Right
        # Right becomes Down, Down becomes Left
        RotFront = pad(torch.arange(48))
        URDL = torch.tensor([0, 2, 5, 4])
        RotFront[URDL, :] = RotFront[torch.roll(URDL, 1), :]
        RotFront = unpad(RotFront)
        RotBack = torch.argsort(RotFront)
        RotLeft = pad(torch.arange(48))
        UFDB = torch.tensor([0, 1, 5, 3])
        RotLeft[UFDB, :] = RotLeft[torch.roll(UFDB, 1), :]
        RotLeft = unpad(RotLeft)
        RotRight = torch.argsort(RotLeft)

        Fperm = RotRight[Uperm[RotLeft]]  # Fperm = RotLeft<-Uperm<-RotRight
        Rperm = RotBack[Uperm[RotFront]]  # Rperm = RotFront<-Uperm<-RotBack
        Bperm = RotLeft[Uperm[RotRight]]  # Bperm = RotRight<-Uperm<-RotLeft
        Lperm = RotFront[Uperm[RotBack]]  # Lperm = RotBack<-Uperm<-RotFront
        # Dperm = RotLeft<-RotLeft<-Uperm<-RotRight<-RotRight
        Dperm = RotRight[RotRight[Uperm[RotLeft[RotLeft]]]]
        self.discrete_generators = [LazyPerm(perm) for perm in [
            Uperm, Fperm, Rperm, Bperm, Lperm, Dperm]]
        self.init()


class ZksZnxZn(Group):
    """ One of the original GCNN groups ℤₖ⋉(ℤₙ×ℤₙ) for translation in x,y
        and rotation with the discrete 90 degree rotations (k=4) or 180 degree (k=2) """

    def __init__(self, k, n):
        super().__init__()
        Zn = Z(n)
        Zk = Z(k)
        nshift = Zn.discrete_generators[0]
        kshift = Zk.discrete_generators[0]
        In = I(n)
        Ik = I(k)
        assert k in [2, 4]
        self.discrete_generators = [
            LazyKron([Ik, nshift, In]),
            LazyKron([Ik, In, nshift]),
            LazyKron([kshift, Rot90(n, 4//k)])]
        self.init(k, n)


class Embed(Group):
    """ A method to embed a given base group representation in larger vector space.
    Inputs: 
    G: the group (and base representation) to embed
    d: the dimension in which to embed
    dim_slice: a slice object specifying which dimensions G acts on. """

    def __init__(self, G, d, dim_slice):
        super().__init__()
        self.lie_algebra = torch.zeros((G.lie_algebra.size(0), d, d))
        self.discrete_generators = torch.zeros((G.discrete_generators.size(0), d, d))
        self.discrete_generators += torch.eye(d)
        self.lie_algebra[:, dim_slice, dim_slice] = G.lie_algebra
        self.discrete_generators[:, dim_slice, dim_slice] = G.discrete_generators
        self.name = f"{G}_R{d}"
        self.init()

    def __repr__(self):
        return self.name


def SO2eR3():
    """ SO(2) embedded in R^3 with rotations about z axis """
    return Embed(SO(2), 3, slice(2))


def O2eR3():
    """ O(2) embedded in R^3 with rotations about z axis """
    return Embed(O(2), 3, slice(2))


def DkeR3(k):
    """ Dihedral D(k) embedded in R^3 with rotations about z axis """
    return Embed(D(k), 3, slice(2))


class DirectProduct(Group):
    """ The direct product of two groups G1 and G2. """

    def __init__(self, G1, G2):
        super().__init__()
        I1, I2 = I(G1.d), I(G2.d)
        self.lie_algebra = [LazyKronsum([A1, 0*I2]) for A1 in G1.lie_algebra] + \
            [LazyKronsum([0*I1, A2]) for A2 in G2.lie_algebra]
        self.discrete_generators = [LazyKron([M1, I2]) for M1 in G1.discrete_generators] + \
            [LazyKron([I1, M2]) for M2 in G2.discrete_generators]
        self.names = (repr(G1), repr(G2))
        self.init()

    def __repr__(self):
        return f"{self.names[0]}x{self.names[1]}"
