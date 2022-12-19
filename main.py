import os
import sys
import torch
import argparse
import numpy as np
from numpy import linalg
from datetime import datetime

import gym
import gym_rotor

import algos.DDPG as DDPG
import algos.TD3 as TD3
import algos.TD3_CAPS as TD3_CAPS
import utils.replay_buffer as replay
from utils.eval_agent import eval_agent
from utils.ctrl_utils import *

if __name__ == "__main__":
	
    # Hyperparameters:
    parser = argparse.ArgumentParser(description='Reinforcement Learning for Quadrotor UAV')
    parser.add_argument("--save_model", default=True, action="store_true",
                    help='Save models and optimizer parameters (default: True)')
    parser.add_argument("--load_model", default=True, type=bool,
                    help='Load and test trained models (default: False)')   
    parser.add_argument("--save_log", default=True, type=bool,
                    help='Load trained models and save log(default: False)')      
    parser.add_argument("--eval_freq", default=1e4, type=int,
                    help='How often (time steps) evaluate our trained model')       
    parser.add_argument('--seed', default=1234, type=int, metavar='N',
                    help='Random seed of Gym, PyTorch and Numpy (default: 123)')      
    # Args of Environment:
    parser.add_argument('--env_id', default="Quad-v0",
                    help='Name of OpenAI Gym environment (default: Quad-v0)')
    parser.add_argument('--wrapper_id', default="",
                    help='Name of wrapper: Sim2RealWrapper')    
    parser.add_argument('--aux_id', default="",
                    help='Name of auxiliary technique: EquivWrapper, CtrlSatWrapper')    
    parser.add_argument('--max_steps', default=2000, type=int,
                    help='Maximum number of steps in each episode (default: 3000)')
    parser.add_argument('--max_timesteps', default=int(1e8), type=int,
                    help='Number of total timesteps (default: 1e8)')
    parser.add_argument('--render', default=False, type=bool,
                    help='Simulation visualization (default: False)')
    # Args of Agents:
    parser.add_argument("--policy", default="TD3_CAPS",
                    help='Which algorithms? DDPG or TD3 or TD3_CAPS(default: TD3)')
    parser.add_argument("--hidden_dim", default=64, type=int, 
                    help='Number of nodes in hidden layers (default: 256)')
    parser.add_argument('--discount', default=0.99, type=float, metavar='G',
                        help='discount factor, gamma (default: 0.99)')
    parser.add_argument('--lr', default=6e-4, type=float, metavar='G',
                        help='learning rate, alpha (default: 1e-5)')
    parser.add_argument("--start_timesteps", default=int(1e4), type=int, 
                    help='Number of steps for uniform-random action selection (default: 25e3)')
    # DDPG:
    parser.add_argument('--tau', default=0.005, type=float, metavar='G',
                    help='Target network update rate (default: 0.005)')
    # TD3:
    parser.add_argument("--act_noise", default=0.1, type=float,
                    help='Stddev for Gaussian exploration noise (default: 0.1)')
    parser.add_argument("--target_noise", default=0.2, type=float,
                    help='Stddev for smoothing noise added to target policy (default: 0.2)')
    parser.add_argument("--noise_clip", default=0.5, type=float,
                    help='Clipping range of target policy smoothing noise (default: 0.5)')
    parser.add_argument('--policy_update_freq', default=2, type=int, metavar='N',
                        help='Frequency of “Delayed” policy updates (default: 2)')
    # Args of Replay buffer:
    parser.add_argument('--batch_size', default=256, type=int, metavar='N',
                        help='Batch size of actor and critic networks (default: 256)')
    parser.add_argument('--replay_buffer_size', default=int(1e6), type=int, metavar='N',
                        help='Maximum size of replay buffer (default: 1e6)')
    args = parser.parse_args()

    # Show information:
    print("-----------------------------------------")
    print(f"Env: {args.env_id}, Policy: {args.policy}, Seed: {args.seed}")
    print("-----------------------------------------")

    # Make OpenAI Gym environment:
    env = gym.make(args.env_id)

    # Set seed for random number generators:
    if args.seed:
        env.seed(args.seed)
        env.action_space.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Initialize policy:
    state_dim  = env.observation_space.shape[0]
    if args.wrapper_id == "EquivWrapper":
        state_dim -= 1 # (_x1, _x3, _v1, _v2, _v3)
    action_dim = env.action_space.shape[0] 
    min_act = env.action_space.low[0]
    max_act = env.action_space.high[0]
    avrg_act = (min_act+max_act)/2.0 
    scale_act = max_act-avrg_act # actor scaling

    kwargs = {
        "state_dim" : state_dim,
        "action_dim": action_dim,
        "hidden_dim": args.hidden_dim,
        "min_act": min_act,
        "max_act": max_act,
        "avrg_act": avrg_act,
        "scale_act": scale_act,
        "discount": args.discount,
        "lr": args.lr,
        "tau": args.tau,
    }

    if args.policy == "DDPG":
        policy = DDPG.DDPG(**kwargs)
    else:
        kwargs["target_noise"] = args.target_noise 
        kwargs["noise_clip"] = args.noise_clip 
        kwargs["policy_update_freq"] = args.policy_update_freq
        if args.policy == "TD3":
            policy = TD3.TD3(**kwargs)
        elif args.policy == "TD3_CAPS":
            policy = TD3_CAPS.TD3_CAPS(**kwargs)
     
    # Set experience replay buffer:
    replay_buffer = replay.ReplayBuffer(state_dim, action_dim, args.replay_buffer_size)

    # Load trained models and optimizer parameters:
    file_name = f"{args.policy}_{args.env_id}"
    if args.load_model == True:
        policy.load(f"./models/{file_name + '_best'}") # '_solved' or '_best'

    # Evaluate policy
    eval_policy = [eval_agent(policy, avrg_act, args)]
    if args.load_model == True:
        sys.exit("The trained model has been test!")

    # Save models and optimizer parameters:
    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")
    max_total_reward = -1e9 # to save best models

    # Setup loggers:
    if not os.path.exists("./results"):
        os.makedirs("./results")
    log_epi_path  = os.path.join("./results", "log_epi.txt") 
    #log_step_path = os.path.join("./results", "log_step.txt")   
    log_eval_path = os.path.join("./results", "log_eval.txt")   
    log_epi  = open(log_epi_path, "w+")  # no. episode vs. Total reward vs. Each timesteps
    #log_step = open(log_step_path, "w+") # Total timesteps vs. Total reward
    log_eval = open(log_eval_path,"w+")  # Total timesteps vs. Evaluated average reward

    # Initialize environment:
    state, done = env.reset(env_type='train'), False
    action = avrg_act * np.ones(4)
    i_episode, episode_timesteps, episode_reward = 0, 0, 0
    i_eval = 1  

    # Training loop:
    for total_timesteps in range(int(args.max_timesteps)):
        episode_timesteps += 1

        if args.wrapper_id == "EquivWrapper":
            state_equiv = rot_e3(state)

        # Keep previous action:
        prev_action = action 
        # Select action randomly or from policy:
        if total_timesteps < args.start_timesteps:
            action = env.action_space.sample() 
        else:
            if args.wrapper_id == "EquivWrapper":
                action = policy.select_action(np.array(state_equiv))
            else:
                action = policy.select_action(np.array(state))
            action = (
                action + np.random.normal(0, avrg_act * args.act_noise, size=action_dim)
            ).clip(min_act, max_act)
        # Control input saturation:
        eX = np.round(state[0:3]*env.x_lim, 5) # position error [m]
        if args.wrapper_id == "CtrlSatWrapper":
            action = ctrl_sat(action, eX, min_act, max_act, env)
        # Concatenate `action` and `prev_action:
        action_step = np.concatenate((action, prev_action), axis=None)

        # Perform action:
        next_state, reward, done, _ = env.step(action_step)
        if done: # Out of boundry
            reward = -1.0

        if args.wrapper_id == "EquivWrapper":
            next_state_equiv = rot_e3(next_state)

        # 3D visualization:
        if args.render == True:
            env.render()

        # Episode termination:
        if episode_timesteps == args.max_steps:
            done = True
            if (abs(eX) <= 0.005).all(): # problem is solved!
                policy.save(f"./models/{file_name+ '_solved_' + str(total_timesteps)}") # save solved model
        done_bool = float(done) if episode_timesteps < args.max_steps else 0

        # Store a set of transitions in replay buffer
        if args.wrapper_id == "EquivWrapper":
            replay_buffer.add(state_equiv, action, next_state_equiv, reward, done_bool)
        else:
            replay_buffer.add(state, action, next_state, reward, done_bool)

        # Train agent after collecting sufficient data:
        if total_timesteps >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        state = next_state
        episode_reward += reward

        if done: 
            print(f"Total-timestpes: {total_timesteps+1}, #Episode: {i_episode+1}, timestpes: {episode_timesteps}, Reward: {episode_reward:.3f}, eX: {eX}")
                        
            # Save best model:
            if episode_reward > max_total_reward:
                print("#---------------- Best! ----------------#")
                best_model = 'Best Model!'
                max_total_reward = episode_reward
                policy.save(f"./models/{file_name + '_best'}")
            else:
                best_model = ''

            # Log data:
            log_epi.write('{}\t {}\t {}\t {}\n'.format(i_episode+1, episode_reward, episode_timesteps, best_model))
            log_epi.flush()
            '''
            if total_timesteps >= args.start_timesteps:
                log_step.write('{}\t {}\n'.format(total_timesteps+1, episode_reward))
                log_step.flush()'''

            # Reset environment:
            state, done = env.reset(env_type='train'), False
            action = avrg_act * np.ones(4)  
            episode_timesteps, episode_reward = 0, 0
            i_episode += 1 

        # Evaluate episode
        if (total_timesteps+1) % args.eval_freq == 0:
            if total_timesteps >= args.start_timesteps:
                eval_policy.append(eval_agent(policy, avrg_act, args))
                # Logging updates:
                log_eval.write('{}\t {}\n'.format(total_timesteps+1, eval_policy[i_eval]))
                log_eval.flush()
                i_eval += 1