import torch
import numpy as np
from datetime import datetime

import gym
import gym_rotor
from utils.ctrl_utils import *

# Runs policy for n episodes and returns average reward.
def eval_agent(policy, avrg_act, args):
    # Make OpenAI Gym environment:
    eval_env = gym.make(args.env_id)

    # Fixed seed is used for the eval environment.
    eval_env.seed(args.seed)
    eval_env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    episode_eval = 5
    avg_reward = 0.0
    for _ in range(episode_eval):
        state, done = eval_env.reset(env_type='eval'), False
        episode_timesteps = 0
        action = avrg_act * np.ones(4) 

        if args.save_log:
            action_list = []
            state_list  = []

        while not done:
            episode_timesteps += 1

            if args.wrapper_id == "EquivWrapper":
                state_equiv = rot_e3(state)

            # Keep previous action:
            prev_action = action 
            # Select action according to policy:
            if args.wrapper_id == "EquivWrapper":
                action = policy.select_action(np.array(state_equiv))
            else:
                action = policy.select_action(np.array(state))
            # Control input saturation:
            eX = np.round(state[0:3]*eval_env.x_lim, 5) # position error [m]
            action = ctrl_sat(action, eX, -1., +1., eval_env)
            # Concatenate `action` and `prev_action:
            action_step = np.concatenate((action, prev_action), axis=None)

            # Perform action
            state, reward, done, _ = eval_env.step(action_step)

            # eval_env.render() # 3D visualization:
            
            # Cumulative rewards:
            avg_reward += reward

            # Save data:
            if args.save_log:
                action_list.append(np.concatenate((action), axis=None))
                state_list.append(state)

            if episode_timesteps == args.max_steps:
                done = True

        # Save data
        if args.save_log:
            min_len = min(len(action_list), len(state_list))
            log_data = np.column_stack((action_list[-min_len:], state_list[-min_len:]))
            header = "Actions and States\n"
            header += "action[0], ..., action[4], state[0], ..., state[17]" 
            time_now = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
            np.savetxt('log_'+time_now+'.dat', log_data, header=header, fmt='%.10f') 

    avg_reward /= episode_eval

    print("---------------------------------------")
    print(f"Evaluation over {episode_eval}, average reward: {avg_reward:.3f}, eX: {eX}")
    print("---------------------------------------")

    return avg_reward