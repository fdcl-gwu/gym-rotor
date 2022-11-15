import torch
import numpy as np
from datetime import datetime

import gym
import gym_rotor
from gym_rotor.envs.ctrl_wrapper import CtrlWrapper

# Runs policy for n episodes and returns average reward.
def eval_agent(policy, env_id, wrapper_id, save_log, max_steps, avrg_act, seed):
    # Make OpenAI Gym environment:
    if wrapper_id == "CtrlWrapper":
        eval_env = CtrlWrapper()
    else:
        eval_env = gym.make(env_id)

    # Fixed seed is used for the eval environment.
    eval_env.seed(seed)
    eval_env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    episode_eval = 5
    avg_reward = 0.0
    for _ in range(episode_eval):
        state, done = eval_env.reset(), False
        episode_timesteps = 0
        action = avrg_act * np.ones(4) 

        if save_log:
            action_list = []
            state_list  = []

        while not done:
            episode_timesteps += 1

            prev_action = action # keep previous action
            # Select action according to policy
            action = policy.select_action(np.array(state))

            # Perform action
            state, reward, done, _ = eval_env.step(action, prev_action)
            
            # Cumulative rewards
            avg_reward += reward

            # Save data:
            if save_log:
                action_list.append(np.concatenate((action), axis=None))
                state_list.append(state)

            if episode_timesteps == max_steps:
                done = True

        # Save data
        if save_log:
            min_len = min(len(action_list), len(state_list))
            log_data = np.column_stack((action_list[-min_len:], state_list[-min_len:]))
            header = "Actions and States\n"
            header += "action[0], ..., action[4], state[0], ..., state[17]" 
            time_now = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
            np.savetxt('log_'+time_now+'.dat', log_data, header=header, fmt='%.10f') 

    avg_reward /= episode_eval

    print("---------------------------------------")
    print(f"Evaluation over {episode_eval}, average reward: {avg_reward:.3f}") 
    print("---------------------------------------")

    return avg_reward