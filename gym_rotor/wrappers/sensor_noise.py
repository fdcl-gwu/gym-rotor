"""
Source: A.Molchanov, T. Chen, W. Hönig, J. A.Preiss, N. Ayanian, G. S. Sukhatme, 
        University of Southern California
        Sim-to-(Multi)-Real: Transfer of Low-Level Robust Control Policies to Multiple Quadrotors
"""
from math import exp
import numpy as np
from numpy.random import normal 
from numpy.random import uniform
from transforms3d.euler import euler2mat, mat2euler
from gym_rotor.envs.quad_utils import *

class SensorNoise:
    def __init__(
            self,
            pos_norm_std=0.002, pos_unif_range=0.001,
            vel_norm_std=0.01, vel_unif_range=0.,
            euler_norm_std=deg2rad(0.5), euler_unif_range=0.,
            #quat_norm_std=deg2rad(0.1), quat_unif_range=deg2rad(0.05),
            gyro_noise_density=0.000175, gyro_random_walk=0.0105,
            gyro_bias_correlation_time=1000., gyro_turn_on_bias_sigma=deg2rad(5),
            bypass=False
    ):
        """
        Args:
            pos_norm_std (float): std of pos gaus noise component
            pos_unif_range (float): range of pos unif noise component
            vel_norm_std (float): std of linear vel gaus noise component
            vel_unif_range (float): range of linear vel unif noise component
            quat_norm_std (float): std of rotational quaternion noisy angle gaus component
            quat_unif_range (float): range of rotational quaternion noisy angle gaus component
            gyro_gyro_noise_density: gyroscope noise, MPU-9250 spec
            gyro_random_walk: gyroscope noise, MPU-9250 spec
            gyro_bias_correlation_time: gyroscope noise, MPU-9250 spec
            gyro_turn_on_bias_sigma: gyroscope noise, MPU-9250 spec (val 0.09)
            bypass: no noise
        """

        self.pos_norm_std = pos_norm_std
        self.pos_unif_range = pos_unif_range

        self.vel_norm_std = vel_norm_std
        self.vel_unif_range = vel_unif_range

        # self.quat_norm_std = quat_norm_std
        # self.quat_unif_range = quat_unif_range
        self.euler_norm_std = euler_norm_std
        self.euler_unif_range = euler_unif_range

        self.gyro_noise_density = gyro_noise_density
        self.gyro_random_walk = gyro_random_walk
        self.gyro_bias_correlation_time = gyro_bias_correlation_time
        self.gyro_turn_on_bias_sigma = gyro_turn_on_bias_sigma
        self.gyro_bias = np.zeros(3)

        self.bypass = bypass
        

    def add_noise(self, pos, vel, rot, omega, dt):
        """
        # Args: 
        #     pos: ground truth of the position in world frame
        #     vel: ground truth if the linear velocity in world frame
        #     rot: ground truth of the orientation in rotational matrix / quaterions / euler angles
        #     omega: ground truth of the angular velocity in body frame
        #     dt: integration step
        """
        assert pos.shape == (3,)
        assert vel.shape == (3,)
        assert rot.shape == (3,3)
        assert omega.shape == (3,)

        if self.bypass:
            return pos, vel, rot, omega

        # Add noise to position measurement:
        pos_noise = normal(loc=0.,scale=self.pos_norm_std,size=3) + \
                    uniform(low=-self.pos_unif_range,high=self.pos_unif_range,size=3)
        noisy_pos = pos + pos_noise

        # Add noise to linear velocity:
        vel_noise = normal(loc=0.,scale=self.vel_norm_std,size=3) + \
                    uniform(low=-self.vel_unif_range,high=self.vel_unif_range,size=3)
        noisy_vel = vel + vel_noise

        # Add noise to attitude:
        euler_noise = normal(loc=0.,scale=self.euler_norm_std,size=3) + \
                      uniform(low=-self.euler_unif_range,high=self.euler_unif_range,size=3)
        noisy_roll, noisy_pitch, noisy_yaw = mat2euler(rot) + euler_noise
        noisy_rot = euler2mat(noisy_roll, noisy_pitch, noisy_yaw) 
        '''
        theta = normal(loc=0.,scale=self.quat_norm_std,size=3) + \
                uniform(low=-self.quat_unif_range,high=self.quat_unif_range,size=3)
        quat_theta = self.quat_from_small_angle(theta)
        quat = self.rot2quat(rot)
        noisy_quat = self.quatXquat(quat, quat_theta)
        noisy_rot = self.quat2R(noisy_quat[0], noisy_quat[1], noisy_quat[2], noisy_quat[3])
        '''

        # Add noise to omega:
        noisy_omega = self.add_noise_to_omega(omega, dt)

        return noisy_pos, noisy_vel, noisy_rot, noisy_omega

    # copied from rotorS imu plugin
    def add_noise_to_omega(self, omega, dt):
        assert omega.shape == (3,)

        sigma_g_d = self.gyro_noise_density / (dt ** 0.5)
        sigma_b_g_d = (-(sigma_g_d ** 2) * (self.gyro_bias_correlation_time / 2) * \
                        (exp(-2 * dt / self.gyro_bias_correlation_time) - 1)) ** 0.5
        pi_g_d = exp(-dt / self.gyro_bias_correlation_time)

        self.gyro_bias = pi_g_d * self.gyro_bias + sigma_b_g_d * normal(0, 1, 3)
        return omega + self.gyro_bias \
               + self.gyro_random_walk * normal(0, 1, 3) \
               + self.gyro_turn_on_bias_sigma * normal(0, 1, 3)


    def quat_from_small_angle(self, theta):
        assert theta.shape == (3,)

        q_squared = np.linalg.norm(theta)**2 / 4.0
        if q_squared < 1:
            q_theta = np.array([(1 - q_squared)**0.5, theta[0] * 0.5, theta[1] * 0.5, theta[2] * 0.5])
        else:
            w = 1.0 / (1 + q_squared)**0.5
            f = 0.5 * w
            q_theta = np.array([w, theta[0] * f, theta[1] * f, theta[2] * f])

        q_theta = q_theta / np.linalg.norm(q_theta)
        return q_theta


    '''
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    '''
    def rot2quat(self, rot):
        assert rot.shape == (3, 3)

        trace = np.trace(rot)
        if trace > 0:
            S = (trace + 1.0)**0.5 * 2
            qw = 0.25 * S
            qx = (rot[2][1] - rot[1][2]) / S 
            qy = (rot[0][2] - rot[2][0]) / S
            qz = (rot[1][0] - rot[0][1]) / S
        elif rot[0][0] > rot[1][1] and rot[0][0] > rot[2][2]:
            S = (1.0 + rot[0][0] - rot[1][1] - rot[2][2])**0.5 * 2
            qw = (rot[2][1] - rot[1][2]) / S
            qx = 0.25 * S 
            qy = (rot[0][1] + rot[1][0]) / S
            qz = (rot[0][2] + rot[2][0]) / S
        elif rot[1][1] > rot[2][2]:
            S = (1.0 + rot[1][1] - rot[0][0] - rot[2][2])**0.5 * 2
            qw = (rot[0][2] - rot[2][0]) / S 
            qx = (rot[0][1] + rot[1][0]) / S
            qy = 0.25 * S
            qz = (rot[1][2] + rot[2][1]) / S
        else:
            S = (1.0 + rot[2][2] - rot[0][0] - rot[1][1])**0.5 * 2
            qw = (rot[1][0] - rot[0][1]) / S
            qx = (rot[0][2] + rot[2][0]) / S 
            qy = (rot[1][2] + rot[2][1]) / S
            qz = 0.25 * S

        return np.array([qw, qx, qy, qz])


    def quat2R(self, qw, qx, qy, qz):
        R = \
        [[1.0 - 2*qy**2 - 2*qz**2,         2*qx*qy - 2*qz*qw,         2*qx*qz + 2*qy*qw],
        [      2*qx*qy + 2*qz*qw,   1.0 - 2*qx**2 - 2*qz**2,         2*qy*qz - 2*qx*qw],
        [      2*qx*qz - 2*qy*qw,         2*qy*qz + 2*qx*qw,   1.0 - 2*qx**2 - 2*qy**2]]
        return np.array(R)


    def quatXquat(self, quat, quat_theta):
        ## quat * quat_theta
        noisy_quat = np.zeros(4)
        noisy_quat[0] = quat[0] * quat_theta[0] - quat[1] * quat_theta[1] - quat[2] * quat_theta[2] - quat[3] * quat_theta[3] 
        noisy_quat[1] = quat[0] * quat_theta[1] + quat[1] * quat_theta[0] - quat[2] * quat_theta[3] + quat[3] * quat_theta[2] 
        noisy_quat[2] = quat[0] * quat_theta[2] + quat[1] * quat_theta[3] + quat[2] * quat_theta[0] - quat[3] * quat_theta[1] 
        noisy_quat[3] = quat[0] * quat_theta[3] - quat[1] * quat_theta[2] + quat[2] * quat_theta[1] + quat[3] * quat_theta[0]
        return noisy_quat
