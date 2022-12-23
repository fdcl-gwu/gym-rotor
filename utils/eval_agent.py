import os
import torch
import numpy as np
from numpy import linalg
from datetime import datetime

import gym
import gym_rotor
from gym_rotor.envs.s2r_wrapper import Sim2RealWrapper
from gym_rotor.envs.quad_utils import *
from utils.ctrl_utils import *

# Runs policy for n episodes and returns average reward.
def eval_agent(policy, args, i_eval, file_name):
    # Make OpenAI Gym environment:
    if args.wrapper_id == "Sim2RealWrapper":
        eval_env = Sim2RealWrapper()
    else:
        eval_env = gym.make(args.env_id)

    # Fixed seed is used for the eval environment.
    eval_env.seed(args.seed)
    eval_env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    episode_eval = 5
    avg_reward = 0.0
    for _ in range(episode_eval):
        # Data save:
        if args.save_log:
            act_list, obs_list, cmd_list = [], [], []

        # State:
        state, done = eval_env.reset(env_type='eval'), False

        # Decomposing state vectors:
        x, v, R, W = state_decomposition(state)

        # Goal state:
        xd = np.array([0.0, 0.0, 0.0])/eval_env.x_lim 
        xd_dot = np.array([0.0, 0.0, 0.0])/eval_env.v_lim 
        Wd = np.array([0.0, 0.0, 0.0])/eval_env.W_lim 
        b1d = np.array([1.0, 0.0, 0.0]) # desired heading direction
        # b1d = get_current_b1(R) # desired heading direction

        episode_timesteps = 0
        while not done:
            episode_timesteps += 1

            if args.aux_id == "EquivWrapper":
                state_equiv = rot_e3(state)

            # Select action according to policy:
            if args.aux_id == "EquivWrapper":
                action = policy.select_action(np.array(state_equiv))
            else:
                action = policy.select_action(np.array(state))
            # Control input saturation:
            eX = np.round(state[0:3]*eval_env.x_lim, 5) # position error [m]
            if args.aux_id == "CtrlSatWrapper":
                action = ctrl_sat(action, eX, -1., +1., eval_env)

            # Perform action
            state, reward, done, _ = eval_env.step(action)

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

        # Save data:
        if args.save_log:
            min_len = min(len(act_list), len(obs_list), len(cmd_list))
            log_data = np.column_stack((act_list[-min_len:], obs_list[-min_len:], cmd_list[-min_len:]))
            header = "Actions and States\n"
            header += "action[0], ..., state[0], ..., command[0], ..." 
            time_now = datetime.now().strftime("%Hh_%Mm_%Ss")
            fpath = os.path.join('./results', 'log_' + time_now + '.dat')
            np.savetxt(fpath, log_data, header=header, fmt='%.10f') 

    # Average reward:
    avg_reward /= episode_eval

    # Save solved model:
    if (abs(eX) <= 0.005).all() and args.save_model == True: # Problem is solved!
        policy.save(f"./models/{file_name+ '_solved_' + str(i_eval)}") 

    print("---------------------------------------")
    print(f"Evaluation over {episode_eval}, average reward: {avg_reward:.3f}, eX: {eX}")
    print("---------------------------------------")


    return avg_reward