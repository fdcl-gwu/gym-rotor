import numpy as np
from numpy import linalg
from numpy.linalg import norm
from math import cos, sin, atan2, sqrt, acos, degrees


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


# Rotation on e3 axis
def R_e3(theta):
    return np.array([[cos(theta), -sin(theta), 0.0],
                     [sin(theta),  cos(theta), 0.0],
                     [       0.0,         0.0, 1.0]])

# Equivariant b1d
def rot_b1d(x):
    # Compute theta:
    theta_x = np.arctan2(x[1], x[0]) 

    # Real desired heading direction:
    b1d = np.array([1.0, 0.0, 0.0])  

    # Imaginary desired heading direction:
    b1d_equiv = R_e3(-theta_x) @ b1d

    return b1d_equiv

def angle_of_vectors(vec1, vec2):
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