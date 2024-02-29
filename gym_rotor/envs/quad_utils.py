import numpy as np
from numpy import dot
from numpy import clip 
from numpy import identity
from numpy import transpose
from numpy.linalg import svd
from numpy.linalg import det
from numpy.linalg import norm
from math import cos, sin, atan2, sqrt, acos, degrees

# Decomposing state vectors
def state_decomposition(state):
    x, v, R_vec, W = state[0:3], state[3:6], state[6:15], state[15:18]
    R = ensure_SO3(R_vec.reshape(3, 3, order='F')) # re-orthonormalization if needed

    return x, v, R, W


# Normalization state vectors: [max, min] -> [-1, 1]
def state_normalization(state, x_lim, v_lim, W_lim):
    x_norm, v_norm, W_norm = state[0:3]/x_lim, state[3:6]/v_lim, state[15:18]/W_lim
    R_vec = state[6:15]
    R = ensure_SO3(R_vec.reshape(3, 3, order='F')) # re-orthonormalization if needed
    R_vec = R.reshape(9, 1, order='F').flatten()

    return x_norm, v_norm, R_vec, W_norm


# De-normalization state vectors: [-1, 1] -> [max, min]
def state_de_normalization(state, x_lim, v_lim, W_lim):
    x, v, W = state[0:3]*x_lim, state[3:6]*v_lim, state[15:18]*W_lim 
    R_vec = state[6:15]
    R = ensure_SO3(R_vec.reshape(3, 3, order='F')) # re-orthonormalization if needed

    return x, v, R, W


class IntegralErrorVec3:
    def __init__(self):
        self.error = np.zeros(3)
        self.integrand = np.zeros(3)

    def integrate(self, current_integrand, dt):
        self.error += (self.integrand + current_integrand) * dt / 2.0
        self.integrand = current_integrand

    def set_zero(self):
        self.error = np.zeros(3)
        self.integrand = np.zeros(3)


class IntegralError:
    def __init__(self):
        self.error = 0.0
        self.integrand = 0.0

    def integrate(self, current_integrand, dt):
        self.error += (self.integrand + current_integrand) * dt / 2.0
        self.integrand = current_integrand

    def set_zero(self):
        self.error = 0.0
        self.integrand = 0.0


class TimeDerivativeVec3:
    def __init__(self):
        self.y_dot = np.zeros(3)
        self.previous_y = np.zeros(3)

    def derivative(self, current_y, dt):
        self.y_dot = (current_y - self.previous_y) / dt
        self.previous_y = current_y

    def set_zero(self):
        self.y_dot = np.zeros(3)
        self.previous_y = np.zeros(3)


def hat(x):
    hat_x = np.array([[0.0, -x[2], x[1]], \
                      [x[2], 0.0, -x[0]], \
                      [-x[1], x[0], 0.0]])
                    
    return np.array(hat_x)


def vee(M):
    # vee map: inverse of the hat map
    vee_M = np.array([[M[2,1]], \
                      [M[0,2]], \
                      [M[1,0]]])

    return np.array(vee_M)


def get_current_b1(R):
    e1 = np.array([1., 0., 0.]) 
    b1 = R.dot(e1)
    theta = atan2(b1[1], b1[0])
    return np.array([cos(theta), sin(theta), 0.])


def get_current_Rd(R):
    # Get the unit vector along the first axis
    e1 = np.array([1., 0., 0.])
    b1 = R.dot(e1)
    theta = atan2(b1[1], b1[0])
    b1_proj = np.array([cos(theta), sin(theta), 0.])

    # Get the unit vector along the second axis
    b2_proj = np.array([-sin(theta), cos(theta), 0.])
    
    # Construct the projected rotation matrix
    R_proj = np.zeros((3, 3))
    R_proj[:, 0] = b1_proj
    R_proj[:, 1] = b2_proj
    R_proj[:, 2] = np.array([0., 0., 1.])

    return R_proj


def ensure_SO3(R, tolerance=1e-6):
    """ Make sure the given input array is in SO(3).

    Args:
        x: (3x3 numpy array) matrix
        tolerance: Tolerance level for considering the magnitude as 1

    Returns:
        True if the input array is in SO(3). Raises an exception otherwise.
    """
    # Calculate the magnitude (norm) of the matrix
    magnitude = np.linalg.det(R)

    # R matrix should satisfy R^T@R = I and det(R) = 1:
    if np.allclose(R.T@R,np.eye(3),rtol=tolerance) and np.isclose(magnitude,1.,rtol=tolerance):
        return R
    else: 
        U, s, VT = psvd(R)
        R = U @ VT.T # Re-orthonormalized R
        return R
    

def ang_btw_two_vectors(vec1, vec2):
    # Compute the dot product:
    unit_vector_1 = vec1 / norm(vec1)
    unit_vector_2 = vec2 / norm(vec2)
    dot_product = clip(dot(unit_vector_1, unit_vector_2), -1., 1.)

    # Compute the angle using arccosine:
    angle = acos(dot_product)
    angle = 0. if angle < 1e-6 else angle
    return angle


def norm_ang_btw_two_vectors(desired_vec, current_vec):
    # Compute the dot product:
    desired_unit_vec = desired_vec / norm(desired_vec)
    current_unit_vec = current_vec / norm(current_vec)
    dot_product = clip(dot(desired_unit_vec, current_unit_vec), -1., 1.)

    # Compute the angle using arccosine:
    angle_radians = acos(dot_product) # [0, pi)

    # Determine the direction of rotation:
    cross_product = np.cross(desired_unit_vec, current_unit_vec)
    z_component_sign = np.sign(cross_product[2])

    # The angle to the range [-pi, pi):
    if z_component_sign < 0:
        angle_radians = -angle_radians

    # Normalize to the range [-1, 1):
    norm_angle = angle_radians/np.pi

    return norm_angle
    

def eulerAnglesToRotationMatrix(theta) :
    # Calculates Rotation Matrix given euler angles.
    R_x = np.array([[1.,             0.,              0.],
                    [0.,  cos(theta[0]),  -sin(theta[0])],
                    [0.,  sin(theta[0]),   cos(theta[0])]])

    R_y = np.array([[ cos(theta[1]),   0.,  sin(theta[1])],
                    [            0.,   1.,             0.],
                    [-sin(theta[1]),   0.,  cos(theta[1])]])

    R_z = np.array([[cos(theta[2]),  -sin(theta[2]),  0.],
                    [sin(theta[2]),   cos(theta[2]),  0.],
                    [           0.,              0.,  1.]])

    R = dot(R_z, dot( R_y, R_x ))

    return R


def isRotationMatrix(R):
    # Checks if a matrix is a valid rotation matrix.
    Rt = transpose(R)
    shouldBeIdentity = dot(Rt, R)
    I = identity(3, dtype = R.dtype)
    n = norm(I - shouldBeIdentity)
    return n < 1e-6


def rotationMatrixToEulerAngles(R):
    # Calculates rotation matrix to euler angles.
    assert(isRotationMatrix(R))

    sy = sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular:
        x = atan2(R[2,1] , R[2,2])
        y = atan2(-R[2,0], sy)
        z = atan2(R[1,0], R[0,0])
    else:
        x = atan2(-R[1,2], R[1,1])
        y = atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])


def psvd(A):
    assert A.shape == (3,3)
    U, s, VT = svd(A)
    detU = det(U)
    detV = det(VT)
    U[:,2] = U[:,2]*detU
    VT[2,:] = VT[2,:]*detV
    s[2] = s[2]*detU*detV
    # assert norm(A-U@np.diag(s)@VT) < 1e-7
    return U, s, VT.T

# def psvd(A):
#     assert A.shape == (3,3)
#     U, s, VT = svd(A)
#     det = np.det(U @ VT)
#     U[:,2] *= np.sign(det)
#     VT[2,:] *= np.sign(det)
#     s[2] *= np.sign(det)
#     assert np.allclose(A, U @ np.diag(s) @ VT)
#     return U, s, VT.T