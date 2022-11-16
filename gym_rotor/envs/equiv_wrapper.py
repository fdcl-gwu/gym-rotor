import numpy as np
from numpy import linalg
from numpy.linalg import inv
from math import cos, sin, atan2, sqrt, pi
from scipy.integrate import odeint, solve_ivp

from gym_rotor.envs.quad import QuadEnv

class EquivWrapper(QuadEnv):

    def __init__(self): 
        super().__init__()
