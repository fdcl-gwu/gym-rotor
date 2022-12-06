## Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import gym
import gym_rotor

# data load and indexing
is_SAVE = True
log_date = np.loadtxt('log_12-06-2022_16-29-43.dat')
start_index = 3
end_index = len(log_date)
time_now = datetime.now().strftime("%H-%M-%S-%m-%d-%Y")
env = gym.make("Quad-v0")

# Loading Saved Data
load_act  = log_date[:, 0:4] # automatically discards the headers
load_obs  = log_date[:, 4:] 
act = load_act[start_index-2: end_index-2]
act = env.scale_act * act + env.avrg_act
obs  = load_obs[start_index-2: end_index-2]

# pre_processing
# Actions
f1 = act[:, 0]
f2 = act[:, 1]
f3 = act[:, 2]
f4 = act[:, 3]

# States
x1 = obs[:, 0] * env.x_lim
x2 = obs[:, 1] * env.x_lim
x3 = obs[:, 2] * env.x_lim
v1 = obs[:, 3] * env.v_lim
v2 = obs[:, 4] * env.v_lim
v3 = obs[:, 5] * env.v_lim
R1 = obs[:, 6] 
R2 = obs[:, 7] 
R3 = obs[:, 8] 
R4 = obs[:, 9] 
R5 = obs[:, 10] 
R6 = obs[:, 11] 
R7 = obs[:, 12] 
R8 = obs[:, 13] 
R9 = obs[:, 14] 
W1 = obs[:, 15] * env.W_lim
W2 = obs[:, 16] * env.W_lim
W3 = obs[:, 17] * env.W_lim

x = np.arange(end_index - start_index)
dt = 0.005
t = x*dt

# Plot Forces
fig, axs = plt.subplots(4, figsize=(15, 8))
fig.suptitle('Actions vs Forces')
axs[0].plot(t, f1)
axs[1].plot(t, f2) 
axs[2].plot(t, f3)
axs[3].plot(t, f4) 
axs[1].set_xlabel('Time [s]')
axs[0].set_ylabel('Forces 1 [N]')
axs[1].set_ylabel('Forces 2 [N]')
axs[2].set_ylabel('Forces 3 [N]')
axs[3].set_ylabel('Forces 4 [N]')
axs[0].grid(True)
axs[1].grid(True)
axs[2].grid(True)
axs[3].grid(True)
if is_SAVE:
    plt.savefig(time_now+'_plots_forces'+'.jpg') 

# Plot States t, v, and W
fig, axs = plt.subplots(3, 3, figsize=(19, 12))

# axs[0, 0].plot(t, xd1, 'tab:red', label='desired')
axs[0, 0].plot(t, x1, label='response')
axs[0, 0].set_title('$x_1$')
axs[0, 0].set_ylabel('$x_1$ [m]', size=15)
axs[0, 0].grid(True)
axs[0, 0].legend()

# axs[0, 1].plot(t, xd2, 'tab:red', label='desired')
axs[0, 1].plot(t, x2, label='response')
axs[0, 1].set_title('$x_2$')
axs[0, 1].set_ylabel('$x_2$ [m]', size=15)
axs[0, 1].grid(True)
axs[0, 1].legend()

# axs[0, 2].plot(t, xd3, 'tab:red', label='desired')
axs[0, 2].plot(t, x3, label='response')
axs[0, 2].set_title('$x_3$')
axs[0, 2].set_ylabel('$x_3$ [m]', size=15)
axs[0, 2].grid(True)
axs[0, 2].legend()

# axs[1, 0].plot(t, vd1, 'tab:red', label='desired')
axs[1, 0].plot(t, v1, label='response')
axs[1, 0].set_title('$v_1$')
axs[1, 0].set_ylabel('$v_1$ [m/s]', size=15)
axs[1, 0].grid(True)
axs[1, 0].legend()

# axs[1, 1].plot(t, vd2, 'tab:red', label='desired')
axs[1, 1].plot(t, v2, label='response')
axs[1, 1].set_title('$v_2$')
axs[1, 1].set_ylabel('$v_2$ [m/s]', size=15)
axs[1, 1].grid(True)
axs[1, 1].legend()

# axs[1, 2].plot(t, vd3, 'tab:red', label='desired')
axs[1, 2].plot(t, v3, label='response')
axs[1, 2].set_title('$v_3$')
axs[1, 2].set_ylabel('$v_3$ [m/s]', size=15)
axs[1, 2].grid(True)
axs[1, 2].legend()

# axs[2, 0].plot(t, Wd1, 'tab:red', label='desired')
axs[2, 0].plot(t, W1, label='response')
axs[2, 0].set_title('$\Omega_1$')
axs[2, 0].set_xlabel('Time [s]', size=15)
axs[2, 0].set_ylabel('$\Omega_1$ [rad/s]', size=15)
axs[2, 0].grid(True)
axs[2, 0].legend()

# axs[2, 1].plot(t, Wd2, 'tab:red', label='desired')
axs[2, 1].plot(t, W2, label='response')
axs[2, 1].set_title('$\Omega_2$')
axs[2, 1].set_xlabel('Time [s]', size=15)
axs[2, 1].set_ylabel('$\Omega_2$ [rad/s]', size=15)
axs[2, 1].grid(True)
axs[2, 1].legend()

# axs[2, 2].plot(t, Wd3, 'tab:red', label='desired')
axs[2, 2].plot(t, W3, label='response')
axs[2, 2].set_title('$\Omega_3$')
axs[2, 2].set_xlabel('Time [s]', size=15)
axs[2, 2].set_ylabel('$\Omega_3$ [rad/s]', size=15)
axs[2, 2].grid(True)
axs[2, 2].legend()
if is_SAVE:
    plt.savefig(time_now+'_plots_x_v_W_states'+'.jpg')   

# Plot States R
fig, axs = plt.subplots(3, 3, figsize=(19, 12))

# axs[0, 0].plot(t, Rd1, 'tab:red', label='desired')
axs[0, 0].plot(t, R1, label='response')
axs[0, 0].set_title('$R_1$')
axs[0, 0].set_ylabel('$R_1$ [rad]', size=15)
axs[0, 0].grid(True)
axs[0, 0].legend()

# axs[0, 1].plot(t, Rd2, 'tab:red', label='desired')
axs[0, 1].plot(t, R2, label='response')
axs[0, 1].set_title('$R_2$')
axs[0, 1].set_ylabel('$R_2$ [rad]', size=15)
axs[0, 1].grid(True)
axs[0, 1].legend()

# axs[0, 2].plot(t, Rd3, 'tab:red', label='desired')
axs[0, 2].plot(t, R3, label='response')
axs[0, 2].set_title('$R_3$')
axs[0, 2].set_ylabel('$R_3$ [rad]', size=15)
axs[0, 2].grid(True)
axs[0, 2].legend()

# axs[1, 0].plot(t, Rd4, 'tab:red', label='desired')
axs[1, 0].plot(t, R4, label='response')
axs[1, 0].set_title('$R_4$')
axs[1, 0].set_ylabel('$R_4$ [rad]', size=15)
axs[1, 0].grid(True)
axs[1, 0].legend()

# axs[1, 1].plot(t, Rd5, 'tab:red', label='desired')
axs[1, 1].plot(t, R5, label='response')
axs[1, 1].set_title('$R_5$')
axs[1, 1].set_ylabel('$R_5$ [rad]', size=15)
axs[1, 1].grid(True)
axs[1, 1].legend()

# axs[1, 2].plot(t, Rd6, 'tab:red', label='desired')
axs[1, 2].plot(t, R6, label='response')
axs[1, 2].set_title('$R_6$')
axs[1, 2].set_ylabel('$R_6$ [rad]', size=15)
axs[1, 2].grid(True)
axs[1, 2].legend()

# axs[2, 0].plot(t, Rd7, 'tab:red', label='desired')
axs[2, 0].plot(t, R7, label='response')
axs[2, 0].set_title('$R_7$')
axs[2, 0].set_xlabel('Time [s]', size=15)
axs[2, 0].set_ylabel('$R_7$ [rad]', size=15)
axs[2, 0].grid(True)
axs[2, 0].legend()

# axs[2, 1].plot(t, Rd8, 'tab:red', label='desired')
axs[2, 1].plot(t, R8, label='response')
axs[2, 1].set_title('$R_8$')
axs[2, 1].set_xlabel('Time [s]', size=15)
axs[2, 1].set_ylabel('$R_8$ [rad]', size=15)
axs[2, 1].grid(True)
axs[2, 1].legend()

# axs[2, 2].plot(t, Rd9, 'tab:red', label='desired')
axs[2, 2].plot(t, R9, label='response')
axs[2, 2].set_title('$R_9$')
axs[2, 2].set_xlabel('Time [s]', size=15)
axs[2, 2].set_ylabel('$R_9$ [rad]', size=15)
axs[2, 2].grid(True)
axs[2, 2].legend()
if is_SAVE:
    plt.savefig(time_now+'_plots_R_states'+'.jpg')   
#plt.show()