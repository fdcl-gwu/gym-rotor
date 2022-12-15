## Importing Libraries
import gym
import gym_rotor
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
# https://www.geeksforgeeks.org/style-plots-using-matplotlib/
# https://www.dunderdata.com/blog/view-all-available-matplotlib-styles
# https://matplotlib.org/stable/gallery/color/named_colors.html
plt.style.use('seaborn')
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['font.size'] = 18
fontsize = 32

# Data load and indexing:
log_date = np.loadtxt('circle.dat') 
start_index = 3
end_index = 3000#len(log_date)
is_SAVE = False

# Pre-processing:
env = gym.make("Quad-v0")
t = np.arange(end_index - start_index)*env.dt # time [sec]
time_now = datetime.now().strftime("%H-%M-%S-%m-%d")

load_act  = log_date[:, 0:4] # automatically discards the headers
load_obs  = log_date[:, 4:22] 
load_cmd  = log_date[:, 22:] 
act = load_act[start_index-2: end_index-2]
act = env.scale_act * act + env.avrg_act # scaling
obs = load_obs[start_index-2: end_index-2]
cmd = load_cmd[start_index-2: end_index-2]

# Actions
f1, f2, f3, f4 = act[:, 0], act[:, 1], act[:, 2], act[:, 3]

# States
x1, x2, x3 = obs[:, 0]*env.x_lim, obs[:, 1]*env.x_lim, obs[:, 2]*env.x_lim
v1, v2, v3 = obs[:, 3]*env.v_lim, obs[:, 4]*env.v_lim, obs[:, 5]*env.v_lim
R11, R21, R31 = obs[:, 6],  obs[:, 7],  obs[:, 8] 
R12, R22, R32 = obs[:, 9],  obs[:, 10], obs[:, 11]
R13, R23, R33 = obs[:, 12], obs[:, 13], obs[:, 14]  
W1, W2, W3 = obs[:, 15]*env.W_lim, obs[:, 16]*env.W_lim, obs[:, 17]*env.W_lim

# Commands
xd1, xd2, xd3 = cmd[:, 0]*env.x_lim, cmd[:, 1]*env.x_lim, cmd[:, 2]*env.x_lim
vd1, vd2, vd3 = cmd[:, 3]*env.v_lim, cmd[:, 4]*env.v_lim, cmd[:, 5]*env.v_lim
b1d1, b1d2, b1d3 = cmd[:, 6], cmd[:, 7], cmd[:, 8]
Wd1, Wd2, Wd3 = cmd[:, 9]*env.W_lim, cmd[:, 10]*env.W_lim, cmd[:, 11]*env.W_lim

#######################################################################
############################# Plot Forces #############################
#######################################################################
fig, axs = plt.subplots(4, figsize=(25, 12))
axs[0].plot(t, f1, linewidth=3)
axs[0].set_ylabel('$T_1$ [N]', size=fontsize)

axs[1].plot(t, f2, linewidth=3) 
axs[1].set_ylabel('$T_2$ [N]', size=fontsize)

axs[2].plot(t, f3, linewidth=3)
axs[2].set_ylabel('$T_3$ [N]', size=fontsize)

axs[3].plot(t, f4, linewidth=3) 
axs[3].set_ylabel('$T_4$ [N]', size=fontsize)
axs[3].set_xlabel('Time [s]', size=fontsize)

for i in range(4):
    axs[i].set_xlim([0., t[-1]])
    axs[i].set_ylim([env.min_force-0.3, env.max_force+0.3])
    axs[i].grid(True, color='white', linestyle='-', linewidth=1.0)
    axs[i].locator_params(axis='y', nbins=4)
for label in (axs[0].get_xticklabels() + axs[0].get_yticklabels()):
	label.set_fontsize(fontsize)
for label in (axs[1].get_xticklabels() + axs[1].get_yticklabels()):
    label.set_fontsize(fontsize)
for label in (axs[2].get_xticklabels() + axs[2].get_yticklabels()):
	label.set_fontsize(fontsize)
for label in (axs[3].get_xticklabels() + axs[3].get_yticklabels()):
	label.set_fontsize(fontsize)
if is_SAVE:
    plt.savefig(time_now+'_plots_forces'+'.png', bbox_inches='tight')


###################################################################################
############################# Plot States x, v, and W #############################
###################################################################################
fig, axs = plt.subplots(3, 3, figsize=(25, 12))

axs[0, 0].plot(t, xd1, 'tab:red', linewidth=3, label='$x_{d_1}$')
axs[0, 0].plot(t, x1, linewidth=3, label='$x_1$')
axs[0, 0].set_ylabel('$x_1$ [m]', size=fontsize)
# axs[0, 0].set_title('$x_1$')

axs[0, 1].plot(t, xd2, 'tab:red', linewidth=3, label='$x_{d_2}$')
axs[0, 1].plot(t, x2, linewidth=3, label='$x_2$')
axs[0, 1].set_ylabel('$x_2$ [m]', size=fontsize)
# axs[0, 1].set_title('$x_2$')

axs[0, 2].plot(t, xd3, 'tab:red', linewidth=3, label='$x_{d_3}$')
axs[0, 2].plot(t, x3, linewidth=3, label='$x_3$')
axs[0, 2].set_ylabel('$x_3$ [m]', size=fontsize)
# axs[0, 2].set_title('$x_3$')

axs[1, 0].plot(t, vd1, 'tab:red', linewidth=3, label='$v_{d_1}$')
axs[1, 0].plot(t, v1, linewidth=3, label='$v_1$')
axs[1, 0].set_ylabel('$v_1$ [m/s]', size=fontsize)
# axs[1, 0].set_title('$v_1$')

axs[1, 1].plot(t, vd2, 'tab:red', linewidth=3, label='$v_{d_2}$')
axs[1, 1].plot(t, v2, linewidth=3, label='$v_2$')
axs[1, 1].set_ylabel('$v_2$ [m/s]', size=fontsize)
# axs[1, 1].set_title('$v_2$')

axs[1, 2].plot(t, vd3, 'tab:red', linewidth=3, label='$v_{d_3}$')
axs[1, 2].plot(t, v3, linewidth=3, label='$v_3$')
axs[1, 2].set_ylabel('$v_3$ [m/s]', size=fontsize)
# axs[1, 2].set_title('$v_3$')

axs[2, 0].plot(t, Wd1, 'tab:red', linewidth=3, label='$\Omega_{d_1}$')
axs[2, 0].plot(t, W1, linewidth=3, label='$\Omega_1$')
axs[2, 0].set_xlabel('Time [s]', size=fontsize)
axs[2, 0].set_ylabel('$\Omega_1$ [rad/s]', size=fontsize)
# axs[2, 0].set_title('$\Omega_1$')

axs[2, 1].plot(t, Wd2, 'tab:red', linewidth=3, label='$\Omega_{d_2}$')
axs[2, 1].plot(t, W2, linewidth=3, label='$\Omega_2$')
axs[2, 1].set_xlabel('Time [s]', size=fontsize)
axs[2, 1].set_ylabel('$\Omega_2$ [rad/s]', size=fontsize)
# axs[2, 1].set_title('$\Omega_2$')

axs[2, 2].plot(t, Wd3, 'tab:red', linewidth=3, label='$\Omega_{d_3}$')
axs[2, 2].plot(t, W3, linewidth=3, label='$\Omega_3$')
axs[2, 2].set_xlabel('Time [s]', size=fontsize)
axs[2, 2].set_ylabel('$\Omega_3$ [rad/s]', size=fontsize)
# axs[2, 2].set_title('$\Omega_3$')

for i in range(3):
    for j in range(3):
        axs[i, j].set_xlim([0., t[-1]])
        axs[i, j].grid(True, color='white', linestyle='-', linewidth=1.0)
        axs[i, j].legend(ncol=1, prop={'size': fontsize}, loc='lower right')
        axs[i, j].locator_params(axis='y', nbins=4)
for label in (axs[0, 0].get_xticklabels() + axs[0, 0].get_yticklabels()):
	label.set_fontsize(fontsize)
for label in (axs[0, 1].get_xticklabels() + axs[0, 1].get_yticklabels()):
	label.set_fontsize(fontsize)
for label in (axs[0, 2].get_xticklabels() + axs[0, 2].get_yticklabels()):
	label.set_fontsize(fontsize)
for label in (axs[1, 0].get_xticklabels() + axs[1, 0].get_yticklabels()):
	label.set_fontsize(fontsize)
for label in (axs[1, 1].get_xticklabels() + axs[1, 1].get_yticklabels()):
	label.set_fontsize(fontsize)
for label in (axs[1, 2].get_xticklabels() + axs[1, 2].get_yticklabels()):
	label.set_fontsize(fontsize)
for label in (axs[2, 0].get_xticklabels() + axs[2, 0].get_yticklabels()):
	label.set_fontsize(fontsize)
for label in (axs[2, 1].get_xticklabels() + axs[2, 1].get_yticklabels()):
	label.set_fontsize(fontsize)
for label in (axs[2, 2].get_xticklabels() + axs[2, 2].get_yticklabels()):
	label.set_fontsize(fontsize)
if is_SAVE:
    plt.savefig(time_now+'_plots_x_v_W_states'+'.png', bbox_inches='tight') 


#########################################################################
############################# Plot States R #############################
#########################################################################
fig, axs = plt.subplots(3, 3, figsize=(25, 12))

axs[0, 0].plot(t, b1d1, 'tab:red', linewidth=3, label='$b_{1d_1}$')
axs[0, 0].plot(t, R11, linewidth=3, label='response')
axs[0, 0].set_ylabel('$R_{11}$', size=fontsize)
axs[0, 0].grid(True, color='white', linestyle='-', linewidth=1.0)
# axs[0, 0].set_yticks(np.arange(0.97, 1.0, 0.01))

axs[1, 0].plot(t, b1d2, 'tab:red', linewidth=3, label='$b_{1d_2}$')
axs[1, 0].plot(t, R21, linewidth=3)
axs[1, 0].set_ylabel('$R_{21}$', size=fontsize)
# axs[1, 0].set_title('$R_{21}$')

axs[2, 0].plot(t, b1d3, 'tab:red', linewidth=3, label='$b_{1d_3}$')
axs[2, 0].plot(t, R31, linewidth=3)
axs[2, 0].set_ylabel('$R_{31}$', size=fontsize)
axs[2, 0].set_xlabel('Time [s]', size=fontsize)
# axs[2, 0].set_title('$R_{31}$')
# axs[2, 0].set_yticks(np.arange(-0.05, 0.08, 0.04))

axs[0, 1].plot(t, R12, linewidth=3)
axs[0, 1].set_ylabel('$R_{12}$', size=fontsize)
# axs[0, 1].set_title('$R_{12}$')

axs[1, 1].plot(t, R22, linewidth=3)
axs[1, 1].set_ylabel('$R_{22}$', size=fontsize)
# axs[1, 1].set_title('$R_{22}$')
# axs[1, 1].set_yticks(np.arange(0.97, 1.0, 0.01))

axs[2, 1].plot(t, R32, linewidth=3)
axs[2, 1].set_ylabel('$R_{32}$', size=fontsize)
axs[2, 1].set_xlabel('Time [s]', size=fontsize)
# axs[2, 1].set_title('$R_{32}$')

axs[0, 2].plot(t, R13, linewidth=3)
axs[0, 2].set_ylabel('$R_{13}$', size=fontsize)
# axs[0, 2].set_title('$R_{13}$')
# axs[0, 2].set_yticks(np.arange(-0.10, 0.05, 0.05))

axs[1, 2].plot(t, R23, linewidth=3)
axs[1, 2].set_ylabel('$R_{23}$', size=fontsize)
# axs[1, 2].set_title('$R_{23}$')

axs[2, 2].plot(t, R33, linewidth=3)
axs[2, 2].set_ylabel('$R_{33}$', size=fontsize)
axs[2, 2].set_xlabel('Time [s]', size=fontsize)
# axs[2, 2].set_title('$R_{33}$')
# axs[2, 2].set_yticks(np.arange(0.990, 1.0, 0.004))

for i in range(3):
    for j in range(3):
        axs[i, j].set_xlim([0., t[-1]])
        axs[i, j].grid(True, color='white', linestyle='-', linewidth=1.0)
        axs[i, j].legend(ncol=1, prop={'size': fontsize}, loc='best')
        axs[i, j].locator_params(axis='y', nbins=4)
for label in (axs[0, 0].get_xticklabels() + axs[0, 0].get_yticklabels()):
	label.set_fontsize(fontsize)
for label in (axs[0, 1].get_xticklabels() + axs[0, 1].get_yticklabels()):
	label.set_fontsize(fontsize)
for label in (axs[0, 2].get_xticklabels() + axs[0, 2].get_yticklabels()):
	label.set_fontsize(fontsize)
for label in (axs[1, 0].get_xticklabels() + axs[1, 0].get_yticklabels()):
	label.set_fontsize(fontsize)
for label in (axs[1, 1].get_xticklabels() + axs[1, 1].get_yticklabels()):
	label.set_fontsize(fontsize)
for label in (axs[1, 2].get_xticklabels() + axs[1, 2].get_yticklabels()):
	label.set_fontsize(fontsize)
for label in (axs[2, 0].get_xticklabels() + axs[2, 0].get_yticklabels()):
	label.set_fontsize(fontsize)
for label in (axs[2, 1].get_xticklabels() + axs[2, 1].get_yticklabels()):
	label.set_fontsize(fontsize)
for label in (axs[2, 2].get_xticklabels() + axs[2, 2].get_yticklabels()):
	label.set_fontsize(fontsize)
if is_SAVE:
    plt.savefig(time_now+'_plots_R_states'+'.png', bbox_inches='tight')

plt.show()