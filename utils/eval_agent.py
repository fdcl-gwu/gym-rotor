import os
import torch
import numpy as np
from datetime import datetime

import gym
import gym_rotor
from gym_rotor.wrappers.s2r_wrapper import Sim2RealWrapper
from gym_rotor.wrappers.equiv_wrapper import EquivWrapper
from gym_rotor.envs.quad_utils import *
from gym_rotor.wrappers.equiv_utils import *

# Runs policy for n episodes and returns average reward.
def eval_agent(policy, args, i_eval, file_name):
    # Make OpenAI Gym environment:
    if args.wrapper_id == "Sim2RealWrapper":
        eval_env = Sim2RealWrapper()
    elif args.aux_id == "EquivWrapper":
        eval_env = EquivWrapper()
    else:
        eval_env = gym.make(args.env_id)

    # Fixed seed is used for the eval environment.
    seed = 12345
    eval_env.seed(seed)
    eval_env.action_space.seed(seed)
    eval_env.observation_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Save solved model:
    success_count = [] if args.save_model else None

    episode_eval = 5
    avg_reward = 0.
    for _ in range(episode_eval):
        # State:
        state, done = eval_env.reset(env_type='eval'), False

        # Decomposing state vectors:
        x, v, R, W = state_decomposition(state)
        '''
        x_equiv, _, R_equiv, _ = equiv_state_decomposition(state)
        b1d_equiv = np.append(x_equiv[:2], 0.) / np.linalg.norm(x_equiv[:2])
        '''

        # Goal state:
        xd = np.array([0.0, 0.0, 0.0])/eval_env.x_lim 
        xd_dot = np.array([0.0, 0.0, 0.0])/eval_env.v_lim  
        Wd = np.array([0.0, 0.0, 0.0])/eval_env.W_lim 
        b1d = np.array([1.0, 0.0, 0.0]) # desired heading direction
        # b1d = get_current_b1(R) # desired heading direction
        Rd = np.eye(3)
        #Rd = get_current_Rd(R)

        # Data save:
        act_list, obs_list, cmd_list = [], [], [] if args.save_log else None

        episode_timesteps = 0
        while not done:
            episode_timesteps += 1

            if args.aux_id == "EquivWrapper":
                # state_equiv = equiv_state(state, b1d)
                state_equiv = equiv_state(state)
                x, v, R, W = state_decomposition(state)
                b1d_equiv = get_equiv_b1d(x, b1d)
                # _, _, R_equiv, _ = equiv_state_decomposition(state)
                x_equiv, v_equiv, R_equiv, W = equiv_state_decomposition(state)
                eR = ang_btw_two_vectors(get_current_b1(R_equiv), b1d_equiv) # heading error [rad]
                '''
                x, _, _, _ = state_decomposition(state)
                b1d = get_actual_b1d(x, b1d_equiv) # actual heading cmd
                '''
            elif args.aux_id == "eRWrapper":
                x, v, R, W = state_decomposition(state)
                RdT_R = Rd.T @ R
                eR = 0.5 * vee(RdT_R - RdT_R.T).flatten()
                R_vec = R.reshape(9, 1, order='F').flatten()
                state = np.concatenate((x, v, R_vec, eR, W), axis=0)
            else:
                _, _, R, _ = state_decomposition(state)
                eR = ang_btw_two_vectors(get_current_b1(R), b1d) # heading error [rad]

            # Select action according to policy:
            if args.aux_id == "EquivWrapper":
                action = policy.select_action(np.array(state_equiv))
            else:
                action = policy.select_action(np.array(state))

            # Perform action
            state, reward, done, _ = eval_env.step(action)
            eX = np.round(state[0:3]*eval_env.x_lim, 5) # position error [m]
            
            # 3D visualization:
            #eval_env.render() 
            
            # Cumulative rewards:
            avg_reward += reward

            # Save data:
            if args.save_log:
                act_list.append(np.concatenate((action), axis=None))
                obs_list.append(np.concatenate((state), axis=None))
                cmd_list.append(np.concatenate((xd, xd_dot, b1d, Wd), axis=None))

            # Episode termination:
            if episode_timesteps == args.max_steps:
                done = True 
            if done == True:
                success = True if (abs(eX) <= 0.1).all() else False
                success_count.append(success)

        # Save data:
        if args.save_log:
            min_len = min(len(act_list), len(obs_list), len(cmd_list))
            log_data = np.column_stack((act_list[-min_len:], obs_list[-min_len:], cmd_list[-min_len:]))
            header = "Actions and States\n"
            header += "action[0], ..., state[0], ..., command[0], ..." 
            time_now = datetime.now().strftime("%m%d%Y_%H%M%S") 
            fpath = os.path.join('./results', 'log_' + time_now + '.dat')
            np.savetxt(fpath, log_data, header=header, fmt='%.10f') 

    # Average reward:
    avg_reward /= episode_eval

    # Save solved model:
    if all(i == True for i in success_count) and args.save_model == True: # Problem is solved!
        policy.save(f"./models/{file_name+ '_solved_' + str(i_eval)}") 

    print("------------------------------------------------------------------------------------------")
    print(f"Evaluation over {episode_eval}, average reward: {avg_reward:.3f}, eX: {eX}, eR: {np.round(eR, 5)}")
    print("------------------------------------------------------------------------------------------")

    return avg_reward