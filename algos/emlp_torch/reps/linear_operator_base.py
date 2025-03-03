# pylint: disable=W0212:protected-access
""" Abstract linear algebra library.
This module defines a class hierarchy that implements a kind of "lazy"
matrix representation, called the ``LinearOperator``. It can be used to do
linear algebra with extremely large sparse or structured matrices, without
representing those explicitly in memory. Such matrices can be added,
multiplied, transposed, etc.
As a motivating example, suppose you want have a matrix where almost all of
the elements have the value one. The standard sparse matrix representation
skips the storage of zeros, but not ones. By contrast, a LinearOperator is
able to represent such matrices efficiently. First, we need a compact way to
represent an all-ones matrix::
    >>> import torch
    >>> class Ones(LinearOperator):
    ...     def __init__(self, shape):
    ...         super(Ones, self).__init__(dtype=None, shape=shape)
    ...     def _matvec(self, v):
    ...         return x.sum().repeat(self.size(0))
Instances of this class emulate ``torch.ones(shape)``, but using a constant
amount of storage, independent of ``shape``. The ``_matvec`` method specifies
how this linear operator multiplies with (operates on) a vector. We can now
add this operator to a sparse matrix that stores only offsets from one::
    >>> offsets = torch.tensor([[1, 0, 2], [0, -1, 0], [0, 0, 3]]).to_sparse()
    >>> A = aslinearoperator(offsets) + Ones(offsets.shape)
    >>> A.dot(torch.tensor([1, 2, 3]))
    tensor([13,  4, 15])
The result is the same as that given by its dense, explicitly-stored
counterpart::
    >>> (torch.ones(A.shape, A.dtype) + offsets.to_dense()).dot(torch.tensor([1, 2, 3]))
    tensor([13,  4, 15])
Several algorithms in the ``torch.sparse`` library are able to operate on
``LinearOperator`` instances.
"""
import warnings
import torch
from torch import nn
from .utils import torch_dtype, dtype_cast, torch_device, device_cast, get_dtype


def isscalar(x):
    """ Is x a scalar? """
    return isinstance(x, (int, float, complex))


def isintlike(x):
    """ Is x an integer-like object? """
    return isinstance(x, int)


def isshape(x, nonneg=False):
    """Is x a valid 2-tuple of dimensions?
    If nonneg, also checks that the dimensions are non-negative.
    """
    try:
        # Assume it's a tuple of matrix dimensions (M, N)
        (M, N) = x
    except Exception:
        return False
    else:
        if (isscalar(M) and isscalar(N)) or (isintlike(M) and isintlike(N)):
            if not nonneg or (M >= 0 and N >= 0):
                return True
        return False


class LinearOperator(nn.Module):
    """ Common interface for performing matrix vector products
    Many iterative methods (e.g. cg, gmres) do not need to know the
    individual entries of a matrix to solve a linear system A*x=b.
    Such solvers only require the computation of matrix vector
    products, A*v where v is a dense vector.  This class serves as
    an abstract interface between iterative solvers and matrix-like
    objects.
    To construct a concrete LinearOperator, either pass appropriate
    callables to the constructor of this class, or subclass it.
    A subclass must implement either one of the methods ``_matvec``
    and ``_matmat``, and the attributes/properties ``shape`` (pair of
    integers) and ``dtype`` (may be None). It may call the ``__init__``
    on this class to have these attributes validated. Implementing
    ``_matvec`` automatically implements ``_matmat`` (using a naive
    algorithm) and vice-versa.
    Optionally, a subclass may implement ``_rmatvec`` or ``_adjoint``
    to implement the Hermitian adjoint (conjugate transpose). As with
    ``_matvec`` and ``_matmat``, implementing either ``_rmatvec`` or
    ``_adjoint`` implements the other automatically. Implementing
    ``_adjoint`` is preferable; ``_rmatvec`` is mostly there for
    backwards compatibility.
    Parameters
    ----------
    shape : tuple
        Matrix dimensions (M, N).
    matvec : callable f(v)
        Returns returns A * v.
    rmatvec : callable f(v)
        Returns A^H * v, where A^H is the conjugate transpose of A.
    matmat : callable f(V)
        Returns A * V, where V is a dense matrix with dimensions (N, K).
    dtype : dtype
        Data type of the matrix.
    rmatmat : callable f(V)
        Returns A^H * V, where V is a dense matrix with dimensions (M, K).
    Attributes
    ----------
    args : tuple
        For linear operators describing products etc. of other linear
        operators, the operands of the binary operation.
    ndim : int
        Number of dimensions (this is always 2)
    See Also
    --------
    aslinearoperator : Construct LinearOperators
    Notes
    -----
    The user-defined matvec() function must properly handle the case
    where v has shape (N,) as well as the (N,1) case.  The shape of
    the return type is handled internally by LinearOperator.
    LinearOperator instances can also be multiplied, added with each
    other and exponentiated, all lazily: the result of these operations
    is always a new, composite LinearOperator, that defers linear
    operations to the original operators and combines the results.
    More details regarding how to subclass a LinearOperator and several
    examples of concrete LinearOperator instances can be found in the
    external project `PyLops <https://pylops.readthedocs.io>`_.
    Examples
    --------
    >>> def mv(v):
    ...     return torch.tensor([2*v[0], 3*v[1]])
    ...
    >>> A = LinearOperator((2,2), matvec=mv)
    >>> A
    <2x2 _CustomLinearOperator with dtype=float64>
    >>> A.matvec(torch.ones(2))
    tensor([ 2.,  3.])
    >>> A * torch.ones(2)
    tensor([ 2.,  3.])
    """

    def __new__(cls, *args, **kwargs):
        if cls is LinearOperator:
            # Operate as _CustomLinearOperator factory.
            return super(LinearOperator, cls).__new__(_CustomLinearOperator)

        obj = super(LinearOperator, cls).__new__(cls)
        if (type(obj)._matvec == LinearOperator._matvec
                and type(obj)._matmat == LinearOperator._matmat):
            warnings.warn("LinearOperator subclass should implement"
                            " at least one of _matvec and _matmat.",
                            category=RuntimeWarning, stacklevel=2)
        return obj

    def __init__(self):
        super().__init__()
        self.ndim = 2
        self.dtype = None
        self.shape = None
        self.device = None

    def init(self, dtype, shape, device):
        """ Initialize this LinearOperator.
        To be called by subclasses. ``dtype`` may be None; ``shape`` should
        be convertible to a length-2 tuple.
        Called from subclasses at the end of the __init__ routine.
        """
        if dtype is None:
            dtype = torch.float  # force float 32
        else:
            if not isinstance(dtype, torch.dtype):
                dtype = torch_dtype(dtype)

        shape = tuple(shape)
        if not isshape(shape):
            raise ValueError(f"invalid shape {(shape,)} (must be 2-d)")

        self.dtype = dtype
        self.shape = torch.Size(shape)
        self.device = torch_device(device)

    def size(self, dim=None):
        """ Return the size of this LinearOperator.
        This is a synonym for ``shape``.
        """
        return self.shape if dim is None else self.shape[dim]

    def _matmat(self, V):
        """ Default matrix-matrix multiplication handler.
        Falls back on the user-defined _matvec method, so defining that will
        define matrix multiplication (though in a very suboptimal way).
        """
        return torch.hstack([self.matvec(col.reshape(-1, 1)) for col in V.T])

    def _matvec(self, v):
        """ Default matrix-vector multiplication handler.
        If self is a linear operator of shape (M, N), then this method will
        be called on a shape (N,) or (N, 1) ndarray, and should return a
        shape (M,) or (M, 1) ndarray.
        This default implementation falls back on _matmat, so defining that
        will define matrix-vector multiplication as well.
        """
        return self.matmat(v.reshape(-1, 1))

    def matvec(self, v):
        """ Matrix-vector multiplication.
        Performs the operation y=A*v where A is an MxN linear
        operator and v is a column vector or 1-d array.
        Parameters
        ----------
        v : {matrix, ndarray}
            An array with shape (N,) or (N,1).
        Returns
        -------
        y : {matrix, ndarray}
            A matrix or ndarray with shape (M,) or (M,1) depending
            on the type and shape of the x argument.
        Notes
        -----
        This matvec wraps the user-specified matvec routine or overridden
        _matvec method to ensure that y has the correct shape and type.
        """
        M, N = self.shape
        if v.shape != (N,) and v.shape != (N, 1):
            raise ValueError('dimension mismatch')

        y = self._matvec(v)

        if v.ndim == 1:
            y = y.reshape(M)
        elif v.ndim == 2:
            y = y.reshape(M, 1)
        else:
            raise ValueError('invalid shape returned by user-defined matvec()')

        return y

    def rmatvec(self, v):
        """ Adjoint matrix-vector multiplication.
        Performs the operation y = A^H * v where A is an MxN linear
        operator and v is a column vector or 1-d array.
        Parameters
        ----------
        v : {matrix, ndarray}
            An array with shape (M,) or (M,1).
        Returns
        -------
        y : {matrix, ndarray}
            A matrix or ndarray with shape (N,) or (N,1) depending
            on the type and shape of the v argument.
        Notes
        -----
        This rmatvec wraps the user-specified rmatvec routine or overridden
        _rmatvec method to ensure that y has the correct shape and type.
        """
        M, N = self.shape

        if v.shape != (M,) and v.shape != (M, 1):
            raise ValueError('dimension mismatch')

        y = self._rmatvec(v)

        if v.ndim == 1:
            y = y.reshape(N)
        elif v.ndim == 2:
            y = y.reshape(N, 1)
        else:
            raise ValueError('invalid shape returned by user-defined rmatvec()')

        return y

    def _rmatvec(self, v):
        """ Default implementation of _rmatvec; defers to adjoint. """
        if type(self)._adjoint == LinearOperator._adjoint:
            # _adjoint not overridden, prevent infinite recursion
            raise NotImplementedError
        return self.H().matvec(v)

    def matmat(self, V):
        """ Matrix-matrix multiplication.
        Performs the operation y=A*V where A is an MxN linear
        operator and V dense N*K matrix or ndarray.
        Parameters
        ----------
        V : {matrix, ndarray}
            An array with shape (N,K).
        Returns
        -------
        Y : {matrix, ndarray}
            A matrix or ndarray with shape (M,K) depending on
            the type of the V argument.
        Notes
        -----
        This matmat wraps any user-specified matmat routine or overridden
        _matmat method to ensure that y has the correct type.
        """
        if V.ndim != 2:
            raise ValueError(f'expected 2-d ndarray or matrix, not {V.ndim}-d')

        if V.size(0) != self.size(1):
            raise ValueError(f'dimension mismatch: {self.shape}, {V.shape}')

        Y = self._matmat(V)
        return Y

    def rmatmat(self, V):
        """ Adjoint matrix-matrix multiplication.
        Performs the operation y = A^H * V where A is an MxN linear
        operator and V is a column vector or 1-d array, or 2-d array.
        The default implementation defers to the adjoint.
        Parameters
        ----------
        V : {matrix, ndarray}
            A matrix or 2D array.
        Returns
        -------
        Y : {matrix, ndarray}
            A matrix or 2D array depending on the type of the input.
        Notes
        -----
        This rmatmat wraps the user-specified rmatmat routine.
        """
        if V.ndim != 2:
            raise ValueError(f'expected 2-d matrix, not {V.ndim}-d')

        if V.size(0) != self.size(0):
            raise ValueError(f'dimension mismatch: {self.shape}, {V.shape}')

        Y = self._rmatmat(V)
        return Y

    def _rmatmat(self, V):
        """ Default implementation of _rmatmat defers to rmatvec or adjoint. """
        if type(self)._adjoint == LinearOperator._adjoint:
            return torch.hstack([self.rmatvec(col.reshape(-1, 1)) for col in V.T])
        return self.H().matmat(V)

    def forward(self, v):
        """ Matrix-vector or matrix-matrix multiplication. """
        return self*v

    def __mul__(self, v):
        return self.dot(v)

    def dot(self, v):
        """ Matrix-matrix or matrix-vector multiplication.
        Parameters
        ----------
        v : array_like
            1-d or 2-d array, representing a vector or matrix.
        Returns
        -------
        Av : array
            1-d or 2-d array (depending on the shape of x) that represents
            the result of applying this linear operator on x.
        """
        if isinstance(v, LinearOperator):
            return _ProductLinearOperator(self, v)
        if torch.is_tensor(v):
            if v.ndim == 0:
                return _ScaledLinearOperator(self, v)
            if v.ndim == 1 or v.ndim == 2 and v.size(1) == 1:
                return self.matvec(v)
            if v.ndim == 2:
                return self.matmat(v)
        raise ValueError(f'expected 1-d or 2-d array or matrix, got {v}')

    def __matmul__(self, other):
        if isscalar(other):
            raise ValueError("Scalar operands are not allowed, use '*' instead")
        return self.__mul__(other)

    def __rmatmul__(self, other):
        if isscalar(other):
            raise ValueError("Scalar operands are not allowed, use '*' instead")
        return self.__rmul__(other)

    def __rmul__(self, x):
        if isscalar(x):
            return _ScaledLinearOperator(self, x)
        return NotImplemented

    def __pow__(self, p):
        if isscalar(p):
            return _PowerLinearOperator(self, p)
        return NotImplemented

    def __add__(self, x):
        if isinstance(x, LinearOperator):
            return _SumLinearOperator(self, x)
        if torch.is_tensor(x) and x.ndim == 2:
            return _SumLinearOperator(self, Lazy(x))
        return NotImplemented

    def __radd__(self, x):
        return self.__add__(x)

    def __neg__(self):
        return _ScaledLinearOperator(self, -1)

    def __sub__(self, x):
        return self.__add__(-x)

    def __repr__(self):
        M, N = self.shape
        if self.dtype is None:
            dtype = 'unspecified dtype'
        else:
            dtype = 'dtype=' + str(self.dtype)

        return f'<{M}x{N} {self.__class__.__name__} with {dtype}>'

    def adjoint(self):
        """ Hermitian adjoint.
        Returns the Hermitian adjoint of self, aka the Hermitian
        conjugate or Hermitian transpose. For a complex matrix, the
        Hermitian adjoint is equal to the conjugate transpose.
        Can be abbreviated self.H instead of self.adjoint().
        Returns
        -------
        A_H : LinearOperator
            Hermitian adjoint of self.
        """
        return self._adjoint()

    def H(self):
        """ Hermitian adjoint. """
        return self.adjoint()

    def transpose(self):
        """ Transpose this linear operator.
        Returns a LinearOperator that represents the transpose of this one.
        Can be abbreviated self.T instead of self.transpose().
        """
        return self._transpose()

    def t(self):
        """ Transpose this linear operator. """
        return self.transpose()

    def _adjoint(self):
        """ Default implementation of _adjoint; defers to rmatvec. """
        return _AdjointLinearOperator(self)

    def _transpose(self):
        """ Default implementation of _transpose; defers to rmatvec + conj"""
        return _TransposedLinearOperator(self)

    def invt(self):
        """ Default implementation of inverse transpose; defers to inv + T """
        return (self ** -1).transpose()

    def to_dense(self):
        """ Default implementation of to_dense which produces the dense
            matrix corresponding to the given lazy matrix. Defaults to
            multiplying by the identity """
        return self@torch.eye(self.size(-1), device=self.device)

    def to(self, device):
        """ Move this linear operator to a new device. """
        self.device = torch.empty(0).to(device).device
        return self


class _CustomLinearOperator(LinearOperator):
    """Linear operator defined in terms of user-specified operations."""

    def __init__(self, shape, matvec, rmatvec=None, matmat=None,
                 dtype=None, device=None, rmatmat=None):
        super().__init__()

        self.__matvec_impl = matvec
        self.__rmatvec_impl = rmatvec
        self.__rmatmat_impl = rmatmat
        self.__matmat_impl = matmat

        self.init(dtype, shape, device)

    def _matmat(self, V):
        if self.__matmat_impl is not None:
            return self.__matmat_impl(V)
        return super()._matmat(V)

    def _matvec(self, v):
        return self.__matvec_impl(v)

    def _rmatvec(self, v):
        func = self.__rmatvec_impl
        if func is None:
            raise NotImplementedError("rmatvec is not defined")
        return self.__rmatvec_impl(v)

    def _rmatmat(self, V):
        if self.__rmatmat_impl is not None:
            return self.__rmatmat_impl(V)
        return super()._rmatmat(V)

    def _adjoint(self):
        return _CustomLinearOperator(shape=(self.size(1), self.size(0)),
                                     matvec=self.__rmatvec_impl,
                                     rmatvec=self.__matvec_impl,
                                     matmat=self.__rmatmat_impl,
                                     rmatmat=self.__matmat_impl,
                                     dtype=self.dtype,
                                     device=self.device)


class _AdjointLinearOperator(LinearOperator):
    """Adjoint of arbitrary linear operator"""

    def __init__(self, A):
        super().__init__()
        self.A = A
        self.init(dtype=A.dtype, shape=(A.size(1), A.size(0)), device=A.device)

    def _matvec(self, v):
        return self.A._rmatvec(v)

    def _rmatvec(self, v):
        return self.A._matvec(v)

    def _matmat(self, V):
        return self.A._rmatmat(V)

    def _rmatmat(self, V):
        return self.A._matmat(V)

    def to(self, device):
        self.A = self.A.to(device)
        self.device = self.A.device
        return self


class _TransposedLinearOperator(LinearOperator):
    """Transposition of arbitrary linear operator"""

    def __init__(self, A):
        super().__init__()
        self.A = A
        self.init(dtype=A.dtype, shape=(A.size(1), A.size(0)), device=A.device)

    def _matvec(self, v):
        # torch.conj works also on sparse matrices
        return torch.conj(self.A._rmatvec(torch.conj(v)))

    def _rmatvec(self, v):
        return torch.conj(self.A._matvec(torch.conj(v)))

    def _matmat(self, V):
        # torch.conj works also on sparse matrices
        return torch.conj(self.A._rmatmat(torch.conj(V)))

    def _rmatmat(self, V):
        return torch.conj(self.A._matmat(torch.conj(V)))

    def to(self, device):
        self.A = self.A.to(device)
        self.device = self.A.device
        return self


class _SumLinearOperator(LinearOperator):
    """ Sum of two Linear Operators """

    def __init__(self, A, B):
        super().__init__()
        if not isinstance(A, LinearOperator) or not isinstance(B, LinearOperator):
            raise ValueError('both operands have to be a LinearOperator')
        if A.shape != B.shape:
            raise ValueError(f'cannot add {A} and {B}: shape mismatch')
        self.A = A
        self.B = B
        self.init(get_dtype([A, B]), A.shape, A.device)

    def _matvec(self, v):
        return self.A.matvec(v) + self.B.matvec(v)

    def _rmatvec(self, v):
        return self.A.rmatvec(v) + self.B.rmatvec(v)

    def _rmatmat(self, V):
        return self.A.rmatmat(V) + self.B.rmatmat(V)

    def _matmat(self, V):
        return self.A.matmat(V) + self.B.matmat(V)

    def _adjoint(self):
        return self.A.H() + self.B.H()

    def invt(self):
        """ Inverse transpose this linear operator. """
        return self.A.invt() + self.B.invt()

    def to(self, device):
        self.A = self.A.to(device)
        self.B = self.B.to(device)
        self.device = self.A.device
        return self


class _ProductLinearOperator(LinearOperator):
    """ Product of two Linear Operators """

    def __init__(self, A, B):
        super().__init__()
        if not isinstance(A, LinearOperator) or not isinstance(B, LinearOperator):
            raise ValueError('both operands have to be a LinearOperator')
        if A.size(1) != B.size(0):
            raise ValueError(f'cannot multiply {A} and {B}: shape mismatch')
        self.A = A
        self.B = B
        self.init(get_dtype([A, B]), (A.size(0), B.size(1)), A.device)

    def _matvec(self, v):
        return self.A.matvec(self.B.matvec(v))

    def _rmatvec(self, v):
        return self.B.rmatvec(self.A.rmatvec(v))

    def _rmatmat(self, V):
        return self.B.rmatmat(self.A.rmatmat(V))

    def _matmat(self, V):
        return self.A.matmat(self.B.matmat(V))

    def _adjoint(self):
        return self.B.H() * self.A.H()

    def invt(self):
        return self.A.invt()*self.B.invt()

    def to_dense(self):
        A = self.A.to_dense() if isinstance(self.A, LinearOperator) else self.A
        B = self.B.to_dense() if isinstance(self.B, LinearOperator) else self.B
        A, B = device_cast(A, B)
        A, B = dtype_cast(A, B)
        return A@B

    def to(self, device):
        self.A = self.A.to(device)
        self.B = self.B.to(device)
        self.device = self.A.device
        return self


class _ScaledLinearOperator(LinearOperator):
    """ Scaled linear operator """

    def __init__(self, A, alpha):
        super().__init__()
        if not isinstance(A, LinearOperator):
            raise ValueError('LinearOperator expected as A')
        if not isinstance(alpha, (int, float, complex)):
            raise ValueError('scalar expected as alpha')
        dtype = get_dtype([A], [type(alpha)])
        self.A = A
        self.alpha = alpha
        self.init(dtype, A.shape, A.device)

    def _matvec(self, v):
        return self.alpha * self.A.matvec(v)

    def _rmatvec(self, v):
        return torch.conj(self.alpha) * self.A.rmatvec(v)

    def _rmatmat(self, V):
        return torch.conj(self.alpha) * self.A.rmatmat(V)

    def _matmat(self, V):
        return self.alpha * self.A.matmat(V)

    def _adjoint(self):
        return self.A.H() * torch.conj(self.alpha)

    def invt(self):
        return (1/self.alpha)*self.A.t()

    def to_dense(self):
        return self.alpha*self.A.to_dense()

    def to(self, device):
        self.A = self.A.to(device)
        self.device = self.A.device
        return self


class _PowerLinearOperator(LinearOperator):
    """ Power of a linear operator """

    def __init__(self, A, p):
        super().__init__()
        if not isinstance(A, LinearOperator):
            raise ValueError('LinearOperator expected as A')
        if A.size(0) != A.size(1):
            raise ValueError(f'square LinearOperator expected, got {A}')
        if (not isinstance(p, int)) or p < 0:
            raise ValueError('non-negative integer expected as p')
        self.A = A
        self.p = p
        self.init(get_dtype([A]), A.shape, A.device)

    def _power(self, fun, x):
        res = x.clone()
        for _ in range(self.p):
            res = fun(res)
        return res

    def _matvec(self, v):
        return self._power(self.A.matvec, v)

    def _rmatvec(self, v):
        return self._power(self.A.rmatvec, v)

    def _rmatmat(self, V):
        return self._power(self.A.rmatmat, V)

    def _matmat(self, V):
        return self._power(self.A.matmat, V)

    def _adjoint(self):
        return self.A.H() ** self.p

    def invt(self):
        return self.A.invt()**self.p

    def to(self, device):
        self.A = self.A.to(device)
        self.device = self.A.device
        return self


class MatrixLinearOperator(LinearOperator):
    """ Linear operator from a matrix. """

    def __init__(self, A):
        super().__init__()
        self.A = A
        self.__adj = None
        self.init(A.dtype, A.shape, A.device)

    def _matmat(self, V):
        return self.A.dot(V)

    def _adjoint(self):
        if self.__adj is None:
            self.__adj = _AdjointMatrixOperator(self)
        return self.__adj

    def to(self, device):
        self.A = self.A.to(device)
        self.device = self.A.device
        if self.__adj is not None:
            self.__adj = self.__adj.to(device)
        return self


class _AdjointMatrixOperator(MatrixLinearOperator):
    """ Adjoint of a matrix linear operator. """

    def __init__(self, adjoint):
        super().__init__(adjoint)
        self.A = adjoint.A.t().conj()
        self.__adjoint = adjoint
        self.shape = torch.Size((adjoint.size(1), adjoint.size(0)))

    @property
    def dtype(self):
        """ Data type of this linear operator. """
        return self.__adjoint.dtype

    def _adjoint(self):
        return self.__adjoint

    def to(self, device):
        self.A = self.A.to(device)
        self.__adjoint = self.__adjoint.to(device)
        self.device = self.A.device
        return self


class IdentityOperator(LinearOperator):
    """ Identity operator. """

    def __init__(self, shape, dtype=None, device=None):
        super().__init__()
        self.init(dtype, shape, device)

    def _matvec(self, v):
        return v

    def _rmatvec(self, v):
        return v

    def _rmatmat(self, V):
        return V

    def _matmat(self, V):
        return V

    def _adjoint(self):
        return self


class Lazy(LinearOperator):
    """ Linear operator with lazy evaluation """
    def __init__(self, dense_matrix):
        super().__init__()
        self.A = dense_matrix
        self.init(self.A.dtype, self.A.shape, self.A.device)

    def _matmat(self, V):
        A, V = device_cast(self.A, V)
        A, V = dtype_cast(A, V)
        return A@V

    def _matvec(self, v):
        A, v = device_cast(self.A, v)
        A, v = dtype_cast(A, v)
        return A@v

    def _rmatmat(self, V):
        A, V = device_cast(self.A, V)
        A, V = dtype_cast(A, V)
        return A.t()@V

    def _rmatvec(self, v):
        A, v = device_cast(self.A, v)
        A, v = dtype_cast(A, v)
        return A.t()@v

    def to_dense(self):
        return self.A

    def invt(self):
        return Lazy(torch.linalg.inv(self.A).t())

    def to(self, device):
        self.A = self.A.to(device)
        self.device = self.A.device
        return self
