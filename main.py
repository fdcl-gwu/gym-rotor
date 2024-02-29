import os
import sys
import copy
import torch
import numpy as np
import gymnasium as gym
from datetime import datetime

import gym_rotor
from gym_rotor.envs.quad_utils import *
from gym_rotor.wrappers.decoupled_yaw_wrapper import DecoupledWrapper
from gym_rotor.wrappers.coupled_yaw_wrapper import CoupledWrapper
from utils.utils import *
from utils.trajectory_generation import TrajectoryGeneration
from algos.replay_buffer import ReplayBuffer
from algos.td3.matd3 import MATD3
from algos.td3.td3 import TD3
import args_parse

# Create directories:    
os.makedirs("./models") if not os.path.exists("./models") else None 
os.makedirs("./results") if not os.path.exists("./results") else None

class Learner:
    def __init__(self, args):
        # Make a new OpenAI Gym environment:
        self.args = args
        self.framework = self.args.framework
        if self.framework in ("CTDE","DTDE"):
            """--------------------------------------------------------------------------------------------------
            | Agents  | Observations           | obs_dim | Actions:       | act_dim | Rewards                   |
            | #agent1 | {ex, ev, b3, w12, eIx} | 15      | {f_total, tau} | 4       | f(ex, ev, eb3, ew12, eIx) |
            | #agent2 | {b1, eb1, W3, eIb1}    | 6       | {M3}           | 1       | f(eb1, eW3, eIb1)         |
            --------------------------------------------------------------------------------------------------"""
            self.env = DecoupledWrapper()
            self.args.N = 2  # num of agents
            self.args.obs_dim_n = [15, 6]
            self.args.action_dim_n = [4, 1]
        elif self.framework == "SARL":
            """------------------------------------------------------------------------------------------------------------------
            | Agents  | Observations                    | obs_dim | Actions:     | act_dim | Rewards                            |
            | #agent1 | {ex, ev, R, eW, eIx, eb1, eIb1} | 23      | {f_total, M} | 4       | f(ex, ev, eb1, eb3, eW, eIx, eIb1) |
            ------------------------------------------------------------------------------------------------------------------"""
            self.env = CoupledWrapper()
            self.args.N = 1  # num of agents
            self.args.obs_dim_n = [23]
            self.args.action_dim_n = [4]
        
        # Set seed for random number generators:
        self.seed = self.args.seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.env.action_space.seed(self.seed)
        self.env.observation_space.seed(self.seed)

        # Initialize the training loop:
        self.total_timesteps = 0  # total num of timesteps
        self.eval_max_steps = self.args.eval_max_steps / self.env.dt  # max num of steps during evaluation; [sec] -> [timestep]
        self.trajectory = TrajectoryGeneration(self.env)  # generate trajectories for evaluation
        if self.args.use_explor_noise_decay:
            self.noise_std_decay = (self.args.explor_noise_std_init - self.args.explor_noise_std_min) / self.args.explor_noise_decay_steps
            self.explor_noise_std = self.args.explor_noise_std_init  # initialize explor_noise_std

        # Initialize N agents:
        if self.framework == "CTDE":
            self.agent_n = [MATD3(args, agent_id) for agent_id in range(self.args.N)]
        elif self.framework in ("SARL", "DTDE"):
            self.agent_n = [TD3(args, agent_id) for agent_id in range(self.args.N)]

        # Initialize replay buffer:
        self.replay_buffer = ReplayBuffer(self.args)
        
        # Load trained models for evaluation:
        if self.args.test_model:
            if self.framework == "CTDE":
                total_steps, agent_id = 2180_000, 0  # edit 'total_steps' accordingly
                # self.agent_n[agent_id].load(self.framework, total_steps, agent_id, self.seed)  # test best models
                self.agent_n[agent_id].load_solved_model(self.framework, total_steps, agent_id, self.seed)  # test solved models
                total_steps, agent_id = 1440_000, 1  # edit 'total_steps' accordingly
                # self.agent_n[agent_id].load(self.framework, total_steps, agent_id, self.seed)  # test best models 
                self.agent_n[agent_id].load_solved_model(self.framework, total_steps, agent_id, self.seed)  # test solved models
            if self.framework == "DTDE":
                total_steps, agent_id = 1710_000, 0  # edit 'total_steps' accordingly
                # self.agent_n[agent_id].load(self.framework, total_steps, agent_id, self.seed)
                self.agent_n[agent_id].load_solved_model(self.framework, total_steps, agent_id, self.seed)
                total_steps, agent_id = 1480_000, 1  # edit 'total_steps' accordingly
                # self.agent_n[agent_id].load(self.framework, total_steps, agent_id, self.seed)
                self.agent_n[agent_id].load_solved_model(self.framework, total_steps, agent_id, self.seed)
            elif self.framework == "SARL":
                total_steps, agent_id = 1580_000, 0  # edit 'total_steps' accordingly
                # self.agent_n[agent_id].load(self.framework, total_steps, agent_id, self.seed)
                self.agent_n[agent_id].load_solved_model(self.framework, total_steps, agent_id, self.seed)


    def train_policy(self):
        # Evaluate policies at the beginning before training:
        self.eval_policy()

        # Setup loggers:
        log_step_path = os.path.join("./results", "log_step_seed_"+str(self.seed)+".txt")   
        log_eval_path = os.path.join("./results", "log_eval_seed_"+str(self.seed)+".txt")
        log_step = open(log_step_path,"w+")  # total reward during training w.r.t. total timesteps
        log_eval = open(log_eval_path,"w+")  # total reward during evaluation w.r.t. total timesteps

        # Initialize the environment:
        obs_n, done_episode, b1d = self.env.reset(env_type='train', seed=self.seed), False, self.env.b1d
        max_total_reward = [0.8 * self.eval_max_steps, 0.8 * self.eval_max_steps]  # starte saving best models after agents achieve 80% of the total reward for each episode
        if self.framework in ("CTDE","DTDE"):
            episode_reward = [0.,0.]
        elif self.framework == "SARL":
            episode_reward = [0.]
        episode_timesteps = 0

        # Training loop:
        for self.total_timesteps in range(int(self.args.max_timesteps)):
            self.total_timesteps += 1
            episode_timesteps += 1

            # Each agent selects actions based on its own local observations with exploration noise:
            if self.total_timesteps < self.args.start_timesteps:  # select actions randomly
                act_n = [np.random.rand(action_dim_n) * 2 - 1 for action_dim_n in self.args.action_dim_n]  # random actions between -1 and 1
            else:  # select actions from trained policies
                act_n = [agent.choose_action(obs, explor_noise_std=self.explor_noise_std) for agent, obs in zip(self.agent_n, obs_n)]
            action = np.concatenate((act_n), axis=None)

            # Perform actions in the environment:
            obs_next_n, r_n, done_n, _, _ = self.env.step(copy.deepcopy(action))

            # Episode termination:
            state = self.env.get_current_state()
            ex_m = np.round(state[0:3]*self.env.x_lim, 5)  # position error in [m]
            eb1_rad = ang_btw_two_vectors(b1d, state[6:9])  # heading error in [rad]
            if episode_timesteps == self.args.max_steps:  # episode terminated!
                done_episode = True
                done_n[0] = True if (abs(ex_m) <= 0.05).all() else False  # problem is solved! when ex < 0.05m
                if self.framework in ("CTDE","DTDE"):
                    done_n[1] = True if abs(eb1_rad) <= 0.03 else False  # problem is solved! when eb1 < 0.03rad
        
            # Store a set of transitions in replay buffer:
            self.replay_buffer.store_transition(obs_n, act_n, r_n, obs_next_n, done_n)
            episode_reward = [float('{:.4f}'.format(episode_reward[agent_id]+r)) for agent_id, r in zip(range(self.args.N), r_n)]
            obs_n = obs_next_n

            # Train agent after collecting sufficient data:
            if self.total_timesteps > self.args.start_timesteps:
                # Train each agent individually:
                for agent_id in range(self.args.N):
                    self.agent_n[agent_id].train(self.replay_buffer, self.agent_n, self.env)

            # If done_episode:
            if any(done_n) == True or done_episode == True:
                print(f"total_timestpes: {self.total_timesteps+1}, time_stpes: {episode_timesteps}, reward: {episode_reward}, ex: {ex_m}, eb1: {eb1_rad:.3f}")

                # Log data:
                if self.total_timesteps >= self.args.start_timesteps:
                    if self.framework in ("CTDE","DTDE"):
                        log_step.write('{}\t {}\n'.format(self.total_timesteps, episode_reward))
                    elif self.framework == "SARL":
                        log_step.write('{}\t {}\n'.format(self.total_timesteps, episode_reward))
                    log_step.flush()
                
                # Reset environment:
                obs_n, done_episode, b1d = self.env.reset(env_type='train', seed=self.seed), False, self.env.b1d
                if self.framework in ("CTDE","DTDE"):
                    episode_reward = [0.,0.]
                elif self.framework == "SARL":
                    episode_reward = [0.]
                episode_timesteps = 0

            # Decay explor_noise_std:
            if self.args.use_explor_noise_decay:
                self.explor_noise_std = self.explor_noise_std - self.noise_std_decay if self.explor_noise_std > self.args.explor_noise_std_min else self.args.explor_noise_std_min

            # Evaluate policy:
            if self.total_timesteps % self.args.eval_freq == 0 and self.total_timesteps > self.args.start_timesteps:
                eval_reward, benchmark_reward = self.eval_policy()

                # Logging updates:
                if self.framework in ("CTDE","DTDE"):
                    log_eval.write('{}\t {}\t {}\n'.format(self.total_timesteps, benchmark_reward, eval_reward))
                elif self.framework == "SARL":
                    log_eval.write('{}\t {}\t {}\n'.format(self.total_timesteps, benchmark_reward, eval_reward))
                log_eval.flush()

                # Save best model:
                for agent_id in range(self.args.N):
                    if eval_reward[agent_id] > max_total_reward[agent_id]:
                        max_total_reward[agent_id] = eval_reward[agent_id]
                        self.agent_n[agent_id].save_model(self.framework, self.total_timesteps, agent_id, self.seed)

        # Close environment:
        self.env.close()


    def eval_policy(self):
        # Make OpenAI Gym environment for evaluation:
        if self.framework in ("CTDE","DTDE"):
            eval_env = DecoupledWrapper()
        elif self.framework == "SARL":
            eval_env = CoupledWrapper()

        # Fixed seed is used for the eval environment:
        seed = 123
        eval_env.action_space.seed(seed)
        eval_env.observation_space.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Save rewards and models:
        success_count = []
        if self.framework in ("CTDE","DTDE"):
            success, eval_reward = [False,False], [0.,0.]
        elif self.framework == "SARL":
            success, eval_reward = [False], [0.]
        benchmark_reward = 0. # Reward for benchmark

        print("---------------------------------------------------------------------------------------------------------------------")
        for num_eval in range(self.args.num_eval):
            # Set mode for generating trajectory:
            mode = 0
            """ Mode List -----------------------------------------------
            0 or 1: idle and warm-up (approach to xd = [0,0,0])
            2: take-off
            3: landing
            4: stay (hovering)
            5: circle
            ----------------------------------------------------------"""
            self.trajectory.mark_traj_start() # reset trajectory

            # Data save:
            act_list, obs_list, cmd_list = [], [], [] if args.save_log else None

            # Reset envs, timesteps, and reward:
            obs_n = eval_env.reset(env_type='eval', seed=self.seed)
            if self.framework in ("CTDE","DTDE"):
                episode_reward = [0.,0.]
            elif self.framework == "SARL":
                episode_reward = [0.]
            episode_timesteps = 0
            episode_benchmark_reward = 0.

            # Evaluation loop:
            for _ in range(int(self.eval_max_steps)):
                episode_timesteps += 1

                # Generate trajectory:
                state = eval_env.get_current_state()
                xd, vd, b1d, b3d, Wd = self.trajectory.get_desired(state, mode)
                eval_env.set_goal_pos(xd, b1d)
                error_obs_n, error_state = self.trajectory.get_error_state(self.framework)

                # Actions without exploration noise:
                act_n = [agent.choose_action(obs, explor_noise_std=0.) for agent, obs in zip(self.agent_n, error_obs_n)] # obs_n
                action = np.concatenate((act_n), axis=None)

                # Perform actions:
                obs_next_n, r_n, done_n, _, _ = eval_env.step(copy.deepcopy(action))
                state_next = eval_env.get_current_state()
                eval_env.render() if self.args.render == True else None

                # Cumulative rewards:
                episode_reward = [float('{:.4f}'.format(episode_reward[agent_id]+r)) for agent_id, r in zip(range(self.args.N), r_n)]
                episode_benchmark_reward += benchmark_reward_func(error_state, args)

                # Save data:
                if self.args.save_log:
                    eIx, eb1, eIb1 = error_state[3], error_state[4], error_state[5]
                    act_list.append(action)
                    obs_list.append(np.concatenate((state, eIx, eb1, eIb1), axis=None))
                    cmd_list.append(np.concatenate((xd, vd, b1d, b3d, Wd), axis=None))

                # Episode termination:
                if any(done_n) or episode_timesteps == self.eval_max_steps:
                    ex_m = np.round(state[0:3]*self.env.x_lim, 5)  # position error [m]
                    eb1_rad = ang_btw_two_vectors(b1d, state[6:9]) # heading error [rad]
                    if self.framework in ("CTDE","DTDE"):
                        success[0] = True if (abs(ex_m) <= 0.05).all() else False
                        success[1] = True if abs(eb1_rad) <= 0.02 else False
                    elif self.framework == "SARL":
                        success[0] = True if (abs(ex_m) <= 0.05).all() else False
                    print(f"eval_iter: {num_eval+1}, time_stpes: {episode_timesteps}, episode_reward: {episode_reward}, episode_benchmark_reward: {episode_benchmark_reward:.3f}, ex: {ex_m}, eb1: {eb1_rad:.3f}")
                    success_count.append(success)
                    break

            # Compute total evaluation rewards:
            eval_reward = [eval_reward[agent_id]+epi_r for agent_id, epi_r in zip(range(self.args.N), episode_reward)]
            benchmark_reward += episode_benchmark_reward

            # Save data:
            if self.args.save_log:
                min_len = min(len(act_list), len(obs_list), len(cmd_list))
                log_data = np.column_stack((act_list[-min_len:], obs_list[-min_len:], cmd_list[-min_len:]))
                header = "Actions and States\n"
                header += "action[0], ..., state[0], ..., command[0], ..." 
                time_now = datetime.now().strftime("%m%d%Y_%H%M%S") 
                fpath = os.path.join('./results', self.framework+'_log_'+time_now+'.dat')
                np.savetxt(fpath, log_data, header=header, fmt='%.10f') 

        # Average reward:
        eval_reward = [float('{:.4f}'.format(eval_r/self.args.num_eval)) for eval_r in eval_reward]
        benchmark_reward = float('{:.4f}'.format(benchmark_reward/self.args.num_eval))
        print("--------------------------------------------------------------------------------------------------------------------------------")
        print(f"total_timesteps: {self.total_timesteps} \t eval_reward: {eval_reward} \t benchmark_reward: {benchmark_reward} \t explor_noise_std: {self.explor_noise_std}")
        print("--------------------------------------------------------------------------------------------------------------------------------")
        sys.exit("The trained agent has been test!") if self.args.test_model == True else None

        # Save solved model:
        for agent_id in range(self.args.N): 
            if all(i[agent_id] == True for i in success_count) and self.args.save_model == True: # Problem is solved
                self.agent_n[agent_id].save_solved_model(self.framework, self.total_timesteps, agent_id, self.seed)

        return eval_reward, benchmark_reward
        
        
if __name__ == '__main__':
    # Hyperparameters:
    parser = args_parse.create_parser()
    args = parser.parse_args()

    # Show information:
    print("---------------------------------------------------------------------------------------------------------------------")
    print("Framework:", args.framework, "| Seed:", args.seed, "| Batch size:", args.batch_size)
    print("gamma:", args.discount, "| lr_a:", args.lr_a, "| lr_c:", args.lr_c, 
          "| Actor hidden dim:", args.actor_hidden_dim, 
          "| Critic hidden dim:", args.critic_hidden_dim)
    print("---------------------------------------------------------------------------------------------------------------------")

    learner = Learner(args)
    learner.train_policy()