import numpy as np

EPS = 1e-6 #small constant to avoid divisions by 0 and log(0)

thrust_cmds = np.ones(4)
dt = 0.005
motor_damp_time_up = 0.15, # See: [4] for details on motor damping. Note: these are rotational velocity damp params.
motor_damp_time_down = 0.15 #2.0

thrust_cmds_damp = np.zeros([4])
thrust_rot_damp = np.zeros([4])

###################################
thrust_cmds = np.clip(thrust_cmds, a_min=0., a_max=1.)

###################################
## Filtering the thruster and adding noise
# I use the multiplier 4, since 4*T ~ time for a step response to finish, where
# T is a time constant of the first-order filter
motor_tau_up = 4*dt/(motor_damp_time_up + EPS)
motor_tau_down = 4*dt/(motor_damp_time_down + EPS)
motor_tau = motor_tau_up * np.ones([4,])
motor_tau[thrust_cmds < thrust_cmds_damp] = motor_tau_down 
motor_tau[motor_tau > 1.] = 1.

## Since NN commands thrusts we need to convert to rot vel and back
# WARNING: Unfortunately if the linearity != 1 then filtering using square root is not quite correct
# since it likely means that you are using rotational velocities as an input instead of the thrust and hence
# you are filtering square roots of angular velocities
thrust_rot = thrust_cmds**0.5
thrust_rot_damp = motor_tau * (thrust_rot - thrust_rot_damp) + thrust_rot_damp       
thrust_cmds_damp = thrust_rot_damp**2

## Adding noise
thrust_noise = thrust_cmds * thrust_noise.noise()
thrust_cmds_damp = np.clip(thrust_cmds_damp + thrust_noise, 0.0, 1.0)        

thrusts = thrust_max * angvel2thrust(thrust_cmds_damp, linearity=motor_linearity)
#Prop crossproduct give torque directions
torques = prop_crossproducts * thrusts[:,None] # (4,3)=(props, xyz)

# additional torques along z-axis caused by propeller rotations
torques[:, 2] += torque_max * prop_ccw * thrust_cmds_damp 

# net torque: sum over propellers
thrust_torque = np.sum(torques, axis=0) 