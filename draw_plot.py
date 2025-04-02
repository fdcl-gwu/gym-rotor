## Importing Libraries
import os, gym_rotor, args_parse
import gymnasium as gym
from gym_rotor.envs.quad_utils import *
from gym_rotor.wrappers.decoupled_yaw_wrapper import DecoupledWrapper
from gym_rotor.wrappers.coupled_yaw_wrapper import CoupledWrapper
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['font.size'] = 18
fontsize = 25

# Data load and indexing:
file_name = 'MONO_log_20250303_112829'
log_date = np.loadtxt(os.path.join('./results', file_name + '.dat')) 
start_index = 3
end_index = len(log_date)
is_SAVE = True

# Pre-processing:
parser = args_parse.create_parser()
args = parser.parse_args()
if args.framework == "MODUL":
    env = DecoupledWrapper()
    load_act  = log_date[:, 0:5] # automatically discards the headers
    load_obs  = log_date[:, 5:28] 
    load_cmd  = log_date[:, 28:] 
elif args.framework == "MONO":
    env = CoupledWrapper()
    load_act  = log_date[:, 0:4] # automatically discards the headers
    load_obs  = log_date[:, 4:27] 
    load_cmd  = log_date[:, 27:] 
act = load_act[start_index-2: end_index-2]
obs = load_obs[start_index-2: end_index-2]
cmd = load_cmd[start_index-2: end_index-2]
t = np.arange(end_index - start_index)*env.dt # sim time [sec]

# States:
x1, x2, x3 = obs[:, 0], obs[:, 1], obs[:, 2]
v1, v2, v3 = obs[:, 3], obs[:, 4], obs[:, 5]
R11, R21, R31 = obs[:, 6],  obs[:, 7],  obs[:, 8] 
R12, R22, R32 = obs[:, 9],  obs[:, 10], obs[:, 11]
R13, R23, R33 = obs[:, 12], obs[:, 13], obs[:, 14]  
W1, W2, W3 = obs[:, 15], obs[:, 16], obs[:, 17]
eIx1, eIx2, eIx3 = obs[:, 18], obs[:, 19], obs[:, 20]
eb1, eIb1 = obs[:, 21], obs[:, 22] # eb1 = b1 error, [-pi, pi)

# Actions:
fM = np.zeros((4, act.shape[0])) # Force-moment vector
f_total = (
        4 * (env.scale_act * act[:, 0] + env.avrg_act)
        ).clip(4*env.min_force, 4*env.max_force)
if args.framework == "MODUL":
    tau = act[:, 1:4]
    b1, b2 = obs[:, 6:9], obs[:, 9:12]
    fM[0] = f_total
    fM[1] = np.einsum('ij,ij->i', b1, tau) + env.J_nominal[2,2]*W3*W2 # M1
    fM[2] = np.einsum('ij,ij->i', b2, tau) - env.J_nominal[2,2]*W3*W1 # M2
    fM[3] = act[:, 4] # M3
    
    # FM matrix to thrust of each motor:
    forces = (env.fM_to_forces @ fM).clip(env.min_force, env.max_force)
    f1, f2, f3, f4 = forces[0], forces[1], forces[2], forces[3]
elif args.framework == "MONO":
    fM[0] = f_total
    fM[1], fM[2], fM[3] = act[:, 1], act[:, 2], act[:, 3] 
	
    # FM matrix to thrust of each motor:
    forces = (env.fM_to_forces @ fM).clip(env.min_force, env.max_force)
    f1, f2, f3, f4 = forces[0], forces[1], forces[2], forces[3]

# Commands:
xd1, xd2, xd3 = cmd[:, 0], cmd[:, 1], cmd[:, 2]
vd1, vd2, vd3 = cmd[:, 3], cmd[:, 4], cmd[:, 5]
b1d1, b1d2, b1d3 = cmd[:, 6], cmd[:, 7], cmd[:, 8]
Wd1, Wd2, Wd3 = cmd[:, 9], cmd[:, 10], cmd[:, 11]
Rd = np.eye(3) # arbitrary desired attitude

#######################################################################
############################ Plot f and M #############################
#######################################################################
fig, axs = plt.subplots(4, figsize=(30, 12))
axs[0].plot(t, fM[0], linewidth=3)
axs[0].set_ylabel('$f$ [N]', size=fontsize)

axs[1].plot(t, fM[1], linewidth=3) 
axs[1].set_ylabel('$M_1$ [Nm]', size=fontsize)

axs[2].plot(t, fM[2], linewidth=3)
axs[2].set_ylabel('$M_2$ [Nm]', size=fontsize)

axs[3].plot(t, fM[3], linewidth=3) 
axs[3].set_ylabel('$M_3$ [Nm]', size=fontsize)
axs[3].set_xlabel('Time [s]', size=fontsize)

for i in range(4):
    axs[i].set_xlim([0., t[-1]])
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
    if args.framework == "MODUL":
        plt.savefig(os.path.join('./results', file_name[:5]+file_name[9:]+'_fM'+'.png'), bbox_inches='tight')
    elif args.framework == "MONO":
        plt.savefig(os.path.join('./results', file_name[:4]+file_name[8:]+'_fM'+'.png'), bbox_inches='tight')

#######################################################################
############################# Plot Forces #############################
#######################################################################
fig, axs = plt.subplots(4, figsize=(30, 12))
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
    # axs[i].set_ylim([env.min_force-0.3, env.max_force+0.3])
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
    if args.framework == "MODUL":
        plt.savefig(os.path.join('./results', file_name[:5]+file_name[9:]+'_T'+'.png'), bbox_inches='tight')
    elif args.framework == "MONO":
        plt.savefig(os.path.join('./results', file_name[:4]+file_name[8:]+'_T'+'.png'), bbox_inches='tight')

###################################################################################
############################# Plot States x, v, and W #############################
###################################################################################
fig, axs = plt.subplots(3, 3, figsize=(30, 12))

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
    if args.framework == "MODUL":
        plt.savefig(os.path.join('./results', file_name[:5]+file_name[9:]+'_x_v_W'+'.png'), bbox_inches='tight')
    elif args.framework == "MONO":
        plt.savefig(os.path.join('./results', file_name[:4]+file_name[8:]+'_x_v_W'+'.png'), bbox_inches='tight')

#########################################################################
############################# Plot States R #############################
#########################################################################
fig, axs = plt.subplots(3, 3, figsize=(30, 12))

axs[0, 0].plot(t, b1d1, 'tab:red', linewidth=3, label='$b_{1c_1}$')
axs[0, 0].plot(t, R11, linewidth=3, label='response')
axs[0, 0].set_ylabel('$R_{11}$', size=fontsize)
axs[0, 0].grid(True, color='white', linestyle='-', linewidth=1.0)
axs[0, 0].legend(ncol=1, prop={'size': fontsize}, loc='best')
# axs[0, 0].set_yticks(np.arange(0.97, 1.0, 0.01))

axs[1, 0].plot(t, b1d2, 'tab:red', linewidth=3, label='$b_{1c_2}$')
axs[1, 0].plot(t, R21, linewidth=3)
axs[1, 0].set_ylabel('$R_{21}$', size=fontsize)
axs[1, 0].legend(ncol=1, prop={'size': fontsize}, loc='best')
# axs[1, 0].set_title('$R_{21}$')

axs[2, 0].plot(t, b1d3, 'tab:red', linewidth=3, label='$b_{1c_3}$')
axs[2, 0].plot(t, R31, linewidth=3)
axs[2, 0].set_ylabel('$R_{31}$', size=fontsize)
axs[2, 0].set_xlabel('Time [s]', size=fontsize)
axs[2, 0].legend(ncol=1, prop={'size': fontsize}, loc='best')
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
# axs[0, 2].legend(ncol=1, prop={'size': fontsize}, loc='best')
# axs[0, 2].set_title('$R_{13}$')
# axs[0, 2].set_yticks(np.arange(-0.10, 0.05, 0.05))

axs[1, 2].plot(t, R23, linewidth=3)
axs[1, 2].set_ylabel('$R_{23}$', size=fontsize)
# axs[1, 2].legend(ncol=1, prop={'size': fontsize}, loc='best')
# axs[1, 2].set_title('$R_{23}$')

axs[2, 2].plot(t, R33, linewidth=3)
axs[2, 2].set_ylabel('$R_{33}$', size=fontsize)
axs[2, 2].set_xlabel('Time [s]', size=fontsize)
# axs[2, 2].legend(ncol=1, prop={'size': fontsize}, loc='best')
# axs[2, 2].set_title('$R_{33}$')
# axs[2, 2].set_yticks(np.arange(0.990, 1.0, 0.004))

for i in range(3):
    for j in range(3):
        axs[i, j].set_xlim([0., t[-1]])
        axs[i, j].grid(True, color='white', linestyle='-', linewidth=1.0)
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
    if args.framework == "MODUL":
        plt.savefig(os.path.join('./results', file_name[:5]+file_name[9:]+'_R'+'.png'), bbox_inches='tight')
    elif args.framework == "MONO":
        plt.savefig(os.path.join('./results', file_name[:4]+file_name[8:]+'_R'+'.png'), bbox_inches='tight')

##########################################################################
#################### Plot ex, eIx, eb1, and  eIb1R #######################
##########################################################################
ex1, ex2, ex3 = x1 - xd1, x2 - xd2, x3 - xd3
ev1, ev2, ev3 = v1 - vd1, v2 - vd2, v3 - vd3
eW1, eW2, eW3 = W1 - Wd1, W2 - Wd2, W3 - Wd3
mse_ex1, mse_ex2, mse_ex3, mse_yaw = np.mean(ex1**2), np.mean(ex2**2), np.mean(ex3**2), np.mean(eb1**2)
mse_ev1, mse_ev2, mse_ev3 = np.mean(ev1**2), np.mean(ev2**2), np.mean(ev3**2)
mse_eW1, mse_eW2, mse_eW3 = np.mean(eW1**2), np.mean(eW2**2), np.mean(eW3**2)
rmse_ex1, rmse_ex2, rmse_ex3, rmse_yaw = np.sqrt(mse_ex1), np.sqrt(mse_ex2), np.sqrt(mse_ex3), np.sqrt(mse_yaw)
rmse_ev1, rmse_ev2, rmse_ev3 = np.sqrt(mse_ev1), np.sqrt(mse_ev2), np.sqrt(mse_ev3)
rmse_eW1, rmse_eW2, rmse_eW3 = np.sqrt(mse_eW1), np.sqrt(mse_eW2), np.sqrt(mse_eW3)
rmse_f, rmse_M1, rmse_M2, rmse_M3 = np.sqrt(np.mean(fM[0]**2)), np.sqrt(np.mean(fM[1]**2)), np.sqrt(np.mean(fM[2]**2)), np.sqrt(np.mean(fM[3]**2))
max_f, max_M1, max_M2, max_M3 = max(fM[0]), max(abs(fM[1])), max(abs(fM[2])), max(abs(fM[3]))
print('========================================================================')
# print(f"rmse_ex1 [m]: {rmse_ex1:.2f}, rmse_ex2 [m]: {rmse_ex2:.2f}, rmse_ex3 [m]: {rmse_ex3:.2f}")
print(f"rmse_ex1 [cm]: {rmse_ex1*100:.2f}, rmse_ex2 [cm]: {rmse_ex2*100:.2f}, rmse_ex3 [cm]: {rmse_ex3*100:.2f}")
# print(f"rmse_ev1 [m/s]: {rmse_ev1:.2f}, rmse_ev2 [m/s]: {rmse_ev2:.2f}, rmse_ev3 [m/s]: {rmse_ev3:.2f}")
print(f"rmse_ev1 [cm/s]: {rmse_ev1*100:.2f}, rmse_ev2 [cm/s]: {rmse_ev2*100:.2f}, rmse_ev3 [cm/s]: {rmse_ev3*100:.2f}")
# print(f"rmse_eW1 [rad/s]: {rmse_eW1:.2f}, rmse_eW2 [rad/s]: {rmse_eW2:.2f}, rmse_eW3 [rad/s]: {rmse_eW3:.2f}")
print(f"rmse_eW1 [deg/s]: {rmse_eW1*180/np.pi:.2f}, rmse_eW2 [deg/s]: {rmse_eW2*180/np.pi:.2f}, rmse_eW3 [deg/s]: {rmse_eW3*180/np.pi:.2f}")
print('========================================================================')
print(f"rmse_ex [cm]: {(rmse_ex1+rmse_ex2+rmse_ex3)*100/3:.2f}, rmse_ev [cm/s]: {(rmse_ev1+rmse_ev2+rmse_ev3)*100/3:.2f}, rmse_eW [deg/s]: {(rmse_eW1+rmse_eW2+rmse_eW3)*180/np.pi/3:.2f}")
print(f"rmse_eW_12 [deg/s]: {(rmse_eW1+rmse_eW2)*180/np.pi/2:.2f}, rmse_eW_3 [deg/s]: {rmse_eW3*180/np.pi:.2f}, rmse_yaw [deg]: {rmse_yaw*180/np.pi:.2f}")
print(f"rmse_f [N]: {rmse_f:.2f}, max_f [N]: {max_f:.2f}, abs_max_M3 [Nm]: {max_M3:.3f}")
print('========================================================================')

fig, axs = plt.subplots(3, 3, figsize=(30, 12))
axs[0, 0].plot(t, ex1, linewidth=3, label='$e_{x_1}$')
axs[0, 0].set_ylabel('$e_{x_1}$ [m]', size=fontsize)

axs[0, 1].plot(t, ex2, linewidth=3, label='$e_{x_2}$')
axs[0, 1].set_ylabel('$e_{x_2}$ [m]', size=fontsize)

axs[0, 2].plot(t, ex3, linewidth=3, label='$e_{x_3}$')
axs[0, 2].set_ylabel('$e_{x_3}$ [m]', size=fontsize)

axs[1, 0].plot(t, eIx1, linewidth=3, label='$eI_{x_1}$')
axs[1, 0].set_ylabel('$e_{I_{x_1}}$', size=fontsize)

axs[1, 1].plot(t, eIx2, linewidth=3, label='$eI_{x_2}$')
axs[1, 1].set_ylabel('$e_{I_{x_2}}$', size=fontsize)

axs[1, 2].plot(t, eIx3, linewidth=3, label='$eI_{x_3}$')
axs[1, 2].set_ylabel('$e_{I_{x_3}}$', size=fontsize)

axs[2, 0].plot(t, eb1, linewidth=3, label='$e_{R_1}$')
axs[2, 0].set_ylabel('$e_{b_1}$', size=fontsize)

axs[2, 1].plot(t, eIb1, linewidth=3, label='$e_{R_2}$')
axs[2, 1].set_ylabel('$e_{I_{b_1}}$', size=fontsize)

for i in range(3):
    for j in range(3):
        axs[i, j].set_xlim([0., t[-1]])
        axs[i, j].grid(True, color='white', linestyle='-', linewidth=1.0)
        # axs[i, j].legend(ncol=1, prop={'size': fontsize}, loc='lower right')
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
if is_SAVE:
    if args.framework == "MODUL":
        plt.savefig(os.path.join('./results', file_name[:5]+file_name[9:]+'_eIx_eIb1'+'.png'), bbox_inches='tight')
    elif args.framework == "MONO":
        plt.savefig(os.path.join('./results', file_name[:4]+file_name[8:]+'_eIx_eIb1'+'.png'), bbox_inches='tight')
else:
    plt.show()