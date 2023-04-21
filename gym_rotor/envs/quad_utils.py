import numpy as np
from numpy import linalg
from numpy.linalg import norm
from math import cos, sin, atan2, sqrt, acos, degrees

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


class LowPassFilter:
    def __init__(self, prev_filtered_act, cutoff_freq, dt):
        self.dt = dt
        self.cutoff_freq = cutoff_freq
        self.prev_filtered_act = prev_filtered_act
        self.tau = self.calc_filter_coef() 
        
    def calc_filter_coef(self):
        w_cut = 2*np.pi*self.cutoff_freq
        return 1/w_cut
        
    def filter(self, act):
        filtered_act = (self.tau * self.prev_filtered_act + self.dt * act) / (self.tau + self.dt)
        self.prev_filtered_act = filtered_act
        return filtered_act
        

class OUNoise:
    # Ornstein–Uhlenbeck process
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.3):
        """
        @param: mu: mean of noise
        @param: theta: stabilization coeff (i.e. noise return to mean)
        @param: sigma: noise scale coeff
        """
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
        

def deg2rad(x):
    """Converts degrees to radians."""
    return x*np.pi/180


def rad2deg(x):
    """Converts radians to degrees."""
    return x*180/np.pi


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
    e1 = np.array([1.0, 0.0, 0.0]) 
    b1 = R.dot(e1)
    theta = np.arctan2(b1[1], b1[0])
    return np.array([np.cos(theta), np.sin(theta), 0.0])

def get_current_Rd(R):
    # Get the unit vector along the first axis
    e1 = np.array([1.0, 0.0, 0.0])
    b1 = R.dot(e1)
    theta = np.arctan2(b1[1], b1[0])
    b1_proj = np.array([np.cos(theta), np.sin(theta), 0.0])

    # Get the unit vector along the second axis
    b2_proj = np.array([-sin(theta), cos(theta), 0.0])
    '''
    e2 = np.array([0.0, 1.0, 0.0])
    b2 = R.dot(e2) #np.cross(R[:, 2], b1)
    # Compute the projection of the second axis b2 onto the plane perpendicular to the first axis b1_proj
    b2_proj = np.cross(b1_proj, b2) 
    theta = np.arctan2(b2_proj[1], b2_proj[0])
    b2_proj = np.array([np.cos(theta), np.sin(theta), 0.0])
    '''
    
    # Construct the projected rotation matrix
    R_proj = np.zeros((3, 3))
    R_proj[:, 0] = b1_proj
    R_proj[:, 1] = b2_proj
    R_proj[:, 2] = np.array([0.0, 0.0, 1.0])

    return R_proj

# Decomposing state vectors
def state_decomposition(state):
    x, v, R_vec, W = state[0:3], state[3:6], state[6:15], state[15:18]
    R = R_vec.reshape(3, 3, order='F')
    # Re-orthonormalize:
    if not isRotationMatrix(R):
        U, s, VT = psvd(R)
        R = U @ VT.T
        R_vec = R.reshape(9, 1, order='F').flatten()

    return x, v, R, W


# Normalization state vectors: [max, min] -> [-1, 1]
def state_normalization(state, x_lim, v_lim, W_lim):
    x_norm = state[0:3]/x_lim 
    v_norm = state[3:6]/v_lim 
    W_norm = state[15:18]/W_lim
    R_vec = state[6:15]
    R = R_vec.reshape(3, 3, order='F')
    # Re-orthonormalize:
    if not isRotationMatrix(R):
        U, s, VT = psvd(R)
        R = U @ VT.T
        R_vec = R.reshape(9, 1, order='F').flatten()

    return x_norm, v_norm, R, W_norm


# De-normalization state vectors: [-1, 1] -> [max, min]
def state_de_normalization(state, x_lim, v_lim, W_lim):
    x = state[0:3]*x_lim # [m]
    v = state[3:6]*v_lim # [m/s]
    W = state[15:18]*W_lim # [rad/s]
    R_vec = state[6:15]
    R = R_vec.reshape(3, 3, order='F')
    # Re-orthonormalize:
    if not isRotationMatrix(R):
        U, s, VT = psvd(R)
        R = U @ VT.T
        R_vec = R.reshape(9, 1, order='F').flatten()

    return x, v, R, W


def ang_btw_two_vectors(vec1, vec2):
    unit_vector_1 = vec1 / linalg.norm(vec1)
    unit_vector_2 = vec2 / linalg.norm(vec2)
    dot_product = np.clip(np.dot(unit_vector_1, unit_vector_2), -1., 1.)
    angle = np.arccos(dot_product)
    angle = 0. if angle < 1e-6 else angle
    return angle
    

def eulerAnglesToRotationMatrix(theta) :
    # Calculates Rotation Matrix given euler angles.
    R_x = np.array([[1,              0,               0],
                    [0,  cos(theta[0]),  -sin(theta[0])],
                    [0,  sin(theta[0]),   cos(theta[0])]])

    R_y = np.array([[ cos(theta[1]),   0,  sin(theta[1])],
                    [             0,   1,              0],
                    [-sin(theta[1]),   0,  cos(theta[1])]])

    R_z = np.array([[cos(theta[2]),  -sin(theta[2]),  0],
                    [sin(theta[2]),   cos(theta[2]),  0],
                    [            0,               0,  1]])

    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R


def isRotationMatrix(R):
    # Checks if a matrix is a valid rotation matrix.
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
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
    U, s, VT = linalg.svd(A)
    detU = linalg.det(U)
    detV = linalg.det(VT)
    U[:,2] = U[:,2]*detU
    VT[2,:] = VT[2,:]*detV
    s[2] = s[2]*detU*detV
    # assert linalg.norm(A-U@np.diag(s)@VT) < 1e-7
    return U, s, VT.T

# def psvd(A):
#     assert A.shape == (3,3)
#     U, s, VT = linalg.svd(A)
#     det = np.linalg.det(U @ VT)
#     U[:,2] *= np.sign(det)
#     VT[2,:] *= np.sign(det)
#     s[2] *= np.sign(det)
#     assert np.allclose(A, U @ np.diag(s) @ VT)
#     return U, s, VT.T