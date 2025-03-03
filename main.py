import os
import sys
import copy
import torch
import numpy as np
import gymnasium as gym
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-c340a596-fde8-5944-a3bf-5a40017ea818"

import gym_rotor
from gym_rotor.envs.quad_utils import *
from gym_rotor.wrappers.decoupled_yaw_wrapper import DecoupledWrapper
from gym_rotor.wrappers.coupled_yaw_wrapper import CoupledWrapper
from utils.utils import *
from utils.trajectory_generator import TrajectoryGenerator
from algos.replay_buffer import ReplayBuffer
from algos.td3.td3 import TD3
from algos.sac.sac import SAC
from algos.ppo.ppo import PPO
import args_parse

# Create directories
os.makedirs("./models") if not os.path.exists("./models") else None 
os.makedirs("./results") if not os.path.exists("./results") else None

# Running device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA found.")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS found.")
else:
    device = torch.device("cpu")


class Learner:
    def __init__(self, args):
        # Make a new OpenAI Gym environment
        args.device = device
        if args.framework == "MODUL":
            self.env = DecoupledWrapper()
            """----------------------------------------------------------------------------------------------
            | Agents   | Observations            | obs_dim | Actions        | act_dim | Rewards              |
            | module#1 | {ex, eIx, ev, b3, ew12} | 15      | {f_total, tau} | 4       | f(ex, eIx, ev, ew12) |
            | module#2 | {eb1, eIb1, eW3}        | 3       | {M3}           | 1       | f(eb1, eIb1, eW3)    |
            ----------------------------------------------------------------------------------------------"""
            args.N = 2  # num of agents
            args.obs_dim_n = [15, 3]
            args.action_dim_n = [4, 1]
        elif args.framework == "MONO":
            self.env = CoupledWrapper()
            """-------------------------------------------------------------------------------------------------------------
            | Agents  | Observations                    | obs_dim | Actions      | act_dim | Rewards                       |
            | single  | {ex, eIx, ev, R, eb1, eIb1, eW} | 23      | {f_total, M} | 4       | f(ex, eIx, ev, eb1, eIb1, eW) |
            -------------------------------------------------------------------------------------------------------------"""
            args.N = 1  # num of agents
            args.obs_dim_n = [23]
            args.action_dim_n = [4]

        # Convert args to dictionary, e.g., self.rl_algo = args.rl_algo
        self.__dict__.update(vars(args))  
        
        # Set seed for random number generators
        set_seed(self.env, self.seed)  # set seed for random number generators

        # Limits of each state
        self.x_lim, self.v_lim = self.env.x_lim, self.env.v_lim
        self.eIx_lim, self.eIb1_lim = self.env.eIx_lim, self.env.eIb1_lim

        # Initialize the training loop
        self.total_timesteps = 0  # total num of timesteps
        self.eval_max_steps = self.eval_max_steps / self.env.dt  # max num of steps during evaluation; [sec] -> [timestep]
        if self.rl_algo == "TD3" and self.use_explor_noise_decay:
            explor_noise_decay_steps = self.max_timesteps  # how many steps before the explor_noise_std decays to the minimum
            self.noise_std_decay = (self.explor_noise_std_init - self.explor_noise_std_min) / explor_noise_decay_steps
            self.explor_noise_std = self.explor_noise_std_init  # initialize explor_noise_std
        
        # Initialize the trajectory generator
        self.trajectory_generator = TrajectoryGenerator(self.env)  
        self.train_traj_mode = 0  # set mode for generating curtain trajectories 

        # Initialize N agents
        if self.rl_algo == "TD3":
            self.agent_n = [TD3(args, agent_id) for agent_id in range(self.N)]  
        elif self.rl_algo == "SAC":
            self.agent_n = [SAC(args, agent_id) for agent_id in range(self.N)]
        elif self.rl_algo == "PPO":
            self.agent_n = [PPO(args, agent_id) for agent_id in range(self.N)]
            self.start_timesteps = 0

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(args)

        # Tesnorboard
        if args.save_tensorboard:
            self.writer = SummaryWriter('./results/tesnorboard/{}_{}_{}_{}_{}'.format(datetime.now().strftime("%Y%m%d_%H%M%S"), \
                         self.rl_algo, self.seed, self.framework, self.module_training if self.framework == "MODUL" else ''))
            
        # Load trained models for evaluation
        if self.test_model:
            if self.framework == "MODUL":
                total_steps, agent_id = 564_000, 0  # edit 'total_steps' accordingly
                self.agent_n[agent_id].load(self.rl_algo, self.framework, total_steps, agent_id, self.seed)  # test best models
                # self.agent_n[agent_id].load_solved_model(self.rl_algo, self.framework, total_steps, agent_id, self.seed)  # test solved models
                total_steps, agent_id = 850_000, 1  # edit 'total_steps' accordingly
                self.agent_n[agent_id].load(self.rl_algo, self.framework, total_steps, agent_id, self.seed)  # test best models 
                # self.agent_n[agent_id].load_solved_model(self.rl_algo, self.framework, total_steps, agent_id, self.seed)  # test solved models
            elif self.framework == "MONO":
                total_steps, agent_id = 700_000, 0  # edit 'total_steps' accordingly
                self.agent_n[agent_id].load(self.rl_algo, self.framework, total_steps, agent_id, self.seed)
                # self.agent_n[agent_id].load_solved_model(self.rl_algo, self.framework, total_steps, agent_id, self.seed)  # test solved models


    def train_policy(self):
        # Evaluate policies at the beginning before training
        self.eval_policy()

        # Setup loggers
        log_step_path = os.path.join("./results", "log_step_seed_"+str(self.seed)+".txt")   
        log_eval_path = os.path.join("./results", "log_eval_seed_"+str(self.seed)+".txt")
        log_step = open(log_step_path,"w+")  # total reward during training w.r.t. total timesteps
        log_eval = open(log_eval_path,"w+")  # total reward during evaluation w.r.t. total timesteps

        # Initialize the environment
        state, done_episode = self.env.reset(env_type='train', seed=self.seed), False
        xd, vd, b1d, b1d_dot, Wd = self.trajectory_generator.get_desired(state, self.train_traj_mode)
        self.env.set_goal_state(xd, vd, b1d, b1d_dot, Wd)
        obs_n = self.env.get_norm_error_state(self.framework)

        # Initialize reward variables
        max_total_reward = [0.85*self.eval_max_steps, 0.85*self.eval_max_steps]  # start saving best models after agents achieve 85% of the total reward for each episode
        if self.framework == "MODUL":
            episode_reward = [0.,0.]
        elif self.framework == "MONO":
            episode_reward = [0.]
        episode_timesteps = 0

        # Training loop
        for self.total_timesteps in range(int(self.max_timesteps)):
            self.total_timesteps += 1
            episode_timesteps += 1

            # Generate trajectories for training
            state = self.env.get_current_state()
            xd, vd, b1d, b1d_dot, Wd = self.trajectory_generator.get_desired(state, self.train_traj_mode)
            self.env.set_goal_state(xd, vd, b1d, b1d_dot, Wd)

            # Each agent selects actions based on its own local observations (with exploration noise)
            if self.rl_algo == "PPO":
                act_n, logprob_a_n = map(list, zip(*(agent.choose_action(obs, is_eval=False) \
                                                     for agent, obs in zip(self.agent_n, obs_n)))) # use stochastic when training
            else:
                if self.total_timesteps < self.start_timesteps:  # select actions randomly
                    act_n = [np.random.rand(action_dim_n) * 2 - 1 for action_dim_n in self.action_dim_n]  # random actions between -1 and 1
                else:  # select actions from trained policies
                    if self.rl_algo == "TD3":
                        act_n = [agent.choose_action(obs, explor_noise_std=self.explor_noise_std) for agent, obs in zip(self.agent_n, obs_n)]
                    elif self.rl_algo == "SAC":
                        act_n = [agent.choose_action(obs, is_eval=False) for agent, obs in zip(self.agent_n, obs_n)]
            action = np.concatenate((act_n), axis=None)

            # Perform actions in the environment
            obs_next_n, rwd_n, done_n, _, _ = self.env.step(copy.deepcopy(action))
            state_next = self.env.get_current_state()
            ex, _, _, eb1, _ = get_error_state(obs_next_n, self.x_lim, self.v_lim, self.eIx_lim, self.eIb1_lim, args)

            # Episode termination
            if episode_timesteps == self.max_steps:  # episode terminated!
                done_episode = True
                done_n[0] = True if (abs(ex) <= 0.03).all() and rwd_n[0] != -1. else False  # problem is solved! when ex < 0.03m
                if self.framework == "MODUL":
                    done_n[1] = True if abs(eb1) <= 0.03 and rwd_n[1] != -1. else False  # problem is solved! when eb1 < 0.03rad

            # Store a set of transitions in buffer
            if self.rl_algo == "PPO":
                self.replay_buffer.store_transition(obs_n, act_n, rwd_n, obs_next_n, done_n, logprob_a_n)
            else:
                self.replay_buffer.store_transition(obs_n, act_n, rwd_n, obs_next_n, done_n)
            episode_reward = [float('{:.4f}'.format(episode_reward[agent_id]+r)) for agent_id, r in zip(range(self.N), rwd_n)]
            obs_n = obs_next_n

            # Train agent after collecting sufficient data
            if self.rl_algo == "PPO":
                if self.total_timesteps % args.T_horizon == 0:
                    for agent_id in range(self.N):
                        critic_loss, actor_loss = self.agent_n[agent_id].train(self.replay_buffer,self.agent_n, self.env)
                        if args.save_tensorboard:
                            self.writer.add_scalar('loss/critic_loss', critic_loss, self.total_timesteps)
                            self.writer.add_scalar('loss/actor_loss', actor_loss, self.total_timesteps)
            else:
                if self.total_timesteps > self.start_timesteps:
                    # Train each agent individually:
                    for agent_id in range(self.N):
                        if self.rl_algo == "TD3":
                            if args.save_tensorboard:
                                critic_loss, actor_loss = self.agent_n[agent_id].train(self.replay_buffer, self.agent_n, self.env)
                                self.writer.add_scalar('loss/critic_loss', critic_loss, self.total_timesteps)
                                if actor_loss is not None:
                                    self.writer.add_scalar('loss/actor_loss', actor_loss, self.total_timesteps)
                            else:
                                self.agent_n[agent_id].train(self.replay_buffer, self.agent_n, self.env)
                        elif self.rl_algo == "SAC":
                            critic_loss, actor_loss, alpha_loss, alpha = self.agent_n[agent_id].train(self.replay_buffer, self.agent_n, self.env)
                            if args.save_tensorboard:
                                self.writer.add_scalar('loss/critic_loss', critic_loss, self.total_timesteps)
                                self.writer.add_scalar('loss/actor_loss', actor_loss, self.total_timesteps)
                                self.writer.add_scalar('loss/entropy_loss', alpha_loss, self.total_timesteps)
                                self.writer.add_scalar('entropy/alpha', alpha, self.total_timesteps)
                    
            # When done_episode
            if any(done_n) == True or done_episode == True:

                # Print training updates
                print(f"total_timestpes: {self.total_timesteps+1}, time_stpes: {episode_timesteps}, reward: {episode_reward}, ex: {ex}, eb1: {eb1:.3f}")

                # Log data
                if self.total_timesteps >= self.start_timesteps:
                    if self.framework == "MODUL":
                        log_step.write('{}\t {}\n'.format(self.total_timesteps, episode_reward))
                    elif self.framework == "MONO":
                        log_step.write('{}\t {}\n'.format(self.total_timesteps, episode_reward))
                    log_step.flush()
                
                # Reset environment
                state, done_episode = self.env.reset(env_type='train', seed=self.seed), False
                self.trajectory_generator.mark_traj_start(state) # reset trajectories
                xd, vd, b1d, b1d_dot, Wd = self.trajectory_generator.get_desired(state, self.train_traj_mode)
                self.env.set_goal_state(xd, vd, b1d, b1d_dot, Wd)
                obs_n = self.env.get_norm_error_state(self.framework)
                if self.framework == "MODUL":
                    episode_reward = [0.,0.]
                elif self.framework == "MONO":
                    episode_reward = [0.]
                episode_timesteps = 0

            # Decay explor_noise_std
            if self.rl_algo == "TD3" and self.use_explor_noise_decay:
                self.explor_noise_std = self.explor_noise_std - self.noise_std_decay if self.explor_noise_std > self.explor_noise_std_min else self.explor_noise_std_min

            # Evaluate policy
            if ((self.rl_algo == "PPO" and self.total_timesteps % args.T_horizon == 0) or
                (self.rl_algo in ["TD3", "SAC"] and self.total_timesteps % self.eval_freq == 0 and self.total_timesteps > self.start_timesteps)):
                eval_reward, benchmark_reward = self.eval_policy()

                # Logging updates
                if self.framework == "MODUL":
                    log_eval.write('{}\t {}\t {}\n'.format(self.total_timesteps, benchmark_reward, eval_reward))
                    if args.save_tensorboard: 
                        self.writer.add_scalar('reward/benchmark_reward', benchmark_reward, self.total_timesteps)
                        self.writer.add_scalar('reward/eval_reward1', eval_reward[0], self.total_timesteps)
                        self.writer.add_scalar('reward/eval_reward2', eval_reward[1], self.total_timesteps)
                elif self.framework == "MONO":
                    log_eval.write('{}\t {}\t {}\n'.format(self.total_timesteps, benchmark_reward, eval_reward))
                    if args.save_tensorboard: 
                        self.writer.add_scalar('reward/benchmark_reward', benchmark_reward, self.total_timesteps)
                        self.writer.add_scalar('reward/eval_reward', eval_reward[0], self.total_timesteps)
                log_eval.flush()

                # Save best model
                for agent_id in range(self.N):
                    if eval_reward[agent_id] > max_total_reward[agent_id] and self.save_model == True:
                        max_total_reward[agent_id] = eval_reward[agent_id]
                        self.agent_n[agent_id].save(self.rl_algo, self.framework, self.total_timesteps, agent_id, self.seed)
            
        # Close environment
        self.env.close()


    def eval_policy(self):
        # Make OpenAI Gym environment for evaluation
        if self.framework == "MODUL":
            eval_env = DecoupledWrapper()
        elif self.framework == "MONO":
            eval_env = CoupledWrapper()

        # Initialize the trajectory generator for evaluation
        eval_trajectory_generator = TrajectoryGenerator(eval_env)  

        # Fixed seed is used for the eval environment
        eval_seed = 1992
        set_seed(eval_env, eval_seed)  # set seed for random number generators

        # Save rewards and models
        success_count = []
        if self.framework == "MODUL":
            success, eval_reward = [False,False], [0.,0.]
        elif self.framework == "MONO":
            success, eval_reward = [False], [0.]
        benchmark_reward = 0. # Reward for benchmark

        print("--------------------------------------------------------------------------------------------------------------------------------")
        for num_eval in range(self.num_eval):
            # Set mode for generating trajectories
            eval_traj_mode = self.train_traj_mode
            """----------------------------------------------------------
            eval_traj_mode == 0   # manual mode (idle and warm-up)
            eval_traj_mode == 1:  # hovering
            eval_traj_mode == 2:  # take-off
            eval_traj_mode == 3:  # landing
            eval_traj_mode == 4:  # stay (maintain current position)
            eval_traj_mode == 5:  # circle
            eval_traj_mode == 6:  # eight-shaped curve
            ----------------------------------------------------------"""

            # Data save
            act_list, obs_list, cmd_list = [], [], [] if args.save_log else None

            # Initialize the environment
            state = eval_env.reset(env_type='eval', seed=eval_seed)
            eval_trajectory_generator.mark_traj_start(state) # reset trajectories
            xd, vd, b1d, b1d_dot, Wd = eval_trajectory_generator.get_desired(state, eval_traj_mode)
            eval_env.set_goal_state(xd, vd, b1d, b1d_dot, Wd)
            obs_n = eval_env.get_norm_error_state(self.framework)

            # Initialize reward variables
            if self.framework == "MODUL":
                episode_reward = [0.,0.]
            elif self.framework == "MONO":
                episode_reward = [0.]
            episode_timesteps = 0
            episode_benchmark_reward = 0.

            # Evaluation loop
            for _ in range(int(self.eval_max_steps)):
                episode_timesteps += 1

                # Generate trajectories for evaluation
                state = eval_env.get_current_state()
                xd, vd, b1d, b1d_dot, Wd = eval_trajectory_generator.get_desired(state, eval_traj_mode)
                eval_env.set_goal_state(xd, vd, b1d, b1d_dot, Wd)

                # Actions without exploration noise
                if self.rl_algo == "TD3":
                    act_n = [agent.choose_action(obs, explor_noise_std=0.) for agent, obs in zip(self.agent_n, obs_n)]
                elif self.rl_algo == "SAC":
                    act_n = [agent.choose_action(obs, is_eval=True) for agent, obs in zip(self.agent_n, obs_n)]
                elif self.rl_algo == "PPO":
                    act_n, _ = map(list, zip(*(agent.choose_action(obs, is_eval=True) \
                                               for agent, obs in zip(self.agent_n, obs_n)))) # Take deterministic actions when evaluation
                action = np.concatenate((act_n), axis=None)

                # Save data
                if self.save_log:
                    _, eIx, _, eb1, eIb1 = get_error_state(obs_n, self.x_lim, self.v_lim, self.eIx_lim, self.eIb1_lim, args)
                    obs_list.append(np.concatenate((state, eIx, eb1, eIb1), axis=None))
                    # Compute b1c
                    R = state[6:15].reshape(3,3,order='F')
                    b3 = R@np.array([0.,0.,1.])
                    b1c = b1d - np.dot(b1d, b3) * b3
                    cmd_list.append(np.concatenate((xd, vd, b1c, Wd), axis=None))
                    act_list.append(action)

                # Perform actions
                obs_next_n, rwd_n, done_n, _, _ = eval_env.step(copy.deepcopy(action))
                eval_env.render() if self.render == True else None
                state_next = eval_env.get_current_state()
                ex, eIx, ev, eb1, eIb1 = get_error_state(obs_next_n, self.x_lim, self.v_lim, self.eIx_lim, self.eIb1_lim, args)

                # Cumulative rewards
                episode_reward = [float('{:.4f}'.format(episode_reward[agent_id]+r)) for agent_id, r in zip(range(self.N), rwd_n)]
                episode_benchmark_reward += benchmark_reward_func(ex, eb1)

                # Episode termination
                if any(done_n) or episode_timesteps == self.eval_max_steps:
                    print(f"eval_iter: {num_eval+1}, time_stpes: {episode_timesteps}, episode_reward: {episode_reward}, episode_benchmark_reward: {episode_benchmark_reward:.3f}, ex: {ex}, eb1: {eb1:.3f}")
                    if episode_timesteps == self.eval_max_steps:
                        if self.framework == "MODUL":
                            success[0] = True if (abs(ex) <= 0.01).all() else False
                            success[1] = True if abs(eb1) <= 0.01 else False
                        elif self.framework == "MONO":
                            success[0] = True if (abs(ex) <= 0.01).all() else False
                    success_count.append(success)
                    break
                obs_n = obs_next_n

            # Compute total evaluation rewards
            eval_reward = [eval_reward[agent_id]+epi_r for agent_id, epi_r in zip(range(self.N), episode_reward)]
            benchmark_reward += episode_benchmark_reward

            # Save data
            if self.save_log:
                min_len = min(len(act_list), len(obs_list), len(cmd_list))
                log_data = np.column_stack((act_list[-min_len:], obs_list[-min_len:], cmd_list[-min_len:]))
                header = "Actions and States\n"
                header += "action[0], ..., state[0], ..., command[0], ..." 
                time_now = datetime.now().strftime("%Y%m%d_%H%M%S") 
                fpath = os.path.join('./results', self.framework+'_log_'+time_now+'.dat')
                np.savetxt(fpath, log_data, header=header, fmt='%.10f') 

        # Average reward
        eval_reward = [float('{:.4f}'.format(eval_r/self.num_eval)) for eval_r in eval_reward]
        benchmark_reward = float('{:.4f}'.format(benchmark_reward/self.num_eval))
        print("--------------------------------------------------------------------------------------------------------------------------------")
        print(f"total_timesteps: {self.total_timesteps} \t eval_reward: {eval_reward} \t benchmark_reward: {benchmark_reward}")
        print("--------------------------------------------------------------------------------------------------------------------------------")
        sys.exit("The trained agent has been test!") if self.test_model == True else None

        # Save solved model
        for agent_id in range(self.N): 
            if all(i[agent_id] == True for i in success_count) and self.save_model == True: # Problem is solved
                self.agent_n[agent_id].save_solved_model(self.rl_algo, self.framework, self.total_timesteps, agent_id, self.seed)

        return eval_reward, benchmark_reward
        
        
if __name__ == '__main__':
    # Hyperparameters
    parser = args_parse.create_parser()
    args = parser.parse_args()

    # Show information
    print("--------------------------------------------------------------------------------------------------------------------------------")
    print("Framework:", args.framework, "| Equivariant RL:", args.use_equiv, "| RL algorithm:", args.rl_algo, "| Seed:", args.seed)
    print("gamma:", args.discount, "| lr_a:", args.lr_a, "| lr_c:", args.lr_c, 
          "| Actor hidden dim:", args.actor_hidden_dim, 
          "| Critic hidden dim:", args.critic_hidden_dim)
    print("--------------------------------------------------------------------------------------------------------------------------------")

    learner = Learner(args)
    learner.train_policy()