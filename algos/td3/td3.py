import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from algos.td3.td3_mlp import MLP_Actor_TD3, MLP_Critic, MLP_Critic_CTDE
from algos.td3.td3_emlp import EMLP_MONO_Actor_TD3, EMLP_MODUL1_Actor_TD3, EMLP_MODUL2_Actor_TD3, \
    EMLP_MONO_Critic, EMLP_MODUL1_CTDE_Critic, EMLP_MODUL2_CTDE_Critic, EMLP_MODUL1_DTDE_Critic, EMLP_MODUL2_DTDE_Critic
from algos.policy_regularization import policy_regularization

class TD3(object):
    def __init__(self, args, agent_id):
        """
        Twin Delayed Deep Deterministic Policy Gradient (TD3) class for training RL agents.
        
        Args:
            args: Namespace containing hyperparameters and configurations.
            agent_id: Unique identifier for the agent.
        """
        self.__dict__.update(vars(args))  # Convert args into attributes (e.g., self.discount = args.discount)
        self.args = args
        self.agent_id = agent_id
        self.obs_dim = args.obs_dim_n[agent_id]  # Observation space dimension for each agent
        self.action_dim = args.action_dim_n[agent_id]  # Action space dimension for each agent
        self.actor_hidden_dim = args.actor_hidden_dim[agent_id]  # Actor network hidden layer dimension
        self.lr_a = args.lr_a[agent_id]  # Learning rate for the actor networks
        self.lr_c = args.lr_c[agent_id]  # Learning rate for the critic networks
        self.is_MODUL_CTDE = (self.framework == "MODUL" and self.module_training == "CTDE")  # True if the training scheme is a modular approach using CTDE
        self.is_MODUL_DTDE = (self.framework == "MODUL" and self.module_training == "DTDE")  # True if the training scheme is a modular approach using DTDE
        self.total_it = 0  # Iteration counter

        # Create an instance of the actor networks
        if self.use_equiv:  # Using equivariant reinforcement learning
            if self.framework == "MONO":
                self.actor = EMLP_MONO_Actor_TD3(args, agent_id).to(self.device)
            elif self.framework == "MODUL":
                # Assign different modular actor networks based on agent_id
                if self.agent_id == 0:
                    self.actor = EMLP_MODUL1_Actor_TD3(args, agent_id).to(self.device) 
                elif self.agent_id == 1:
                    self.actor = EMLP_MODUL2_Actor_TD3(args, agent_id).to(self.device)
        else:  # Use standard MLP-based actor network
            self.actor = MLP_Actor_TD3(args, agent_id).to(self.device)

        # If the model is in training mode
        if not self.test_model:
            # Create an instance of the critic networks
            if self.use_equiv:  # Using equivariant reinforcement learning
                if self.framework == "MONO":
                    self.critic = EMLP_MONO_Critic(args, agent_id).to(self.device)
                elif self.framework == "MODUL":
                    if self.module_training == "CTDE":  # Centralized Training, Decentralized Execution
                        if self.agent_id == 0:
                            self.critic = EMLP_MODUL1_CTDE_Critic(args, agent_id).to(self.device)
                        elif self.agent_id == 1:
                            self.critic = EMLP_MODUL2_CTDE_Critic(args, agent_id).to(self.device)
                    elif self.module_training == "DTDE":  # Decentralized Training, Decentralized Execution
                        if self.agent_id == 0:
                            self.critic = EMLP_MODUL1_DTDE_Critic(args, agent_id).to(self.device)
                        elif self.agent_id == 1:
                            self.critic = EMLP_MODUL2_DTDE_Critic(args, agent_id).to(self.device)
            else:  # Use standard MLP-based actor network
                if self.framework == "MONO" or self.is_MODUL_DTDE:
                    self.critic = MLP_Critic(args, agent_id).to(self.device)
                elif self.is_MODUL_CTDE:
                    self.critic = MLP_Critic_CTDE(args).to(self.device)  # Centralized Training, Decentralized Execution

            # Create target networks for stability in training
            self.actor_target = copy.deepcopy(self.actor)
            self.critic_target = copy.deepcopy(self.critic)

            # Initialize optimizers for actor and critic networks
            self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=self.lr_a)
            self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=self.lr_c)

            # Use CosineAnnealingWarmRestarts to adjust learning rate dynamically
            self.actor_scheduler = CosineAnnealingWarmRestarts(self.actor_optimizer, T_0=1_000_000, eta_min=1e-5)
            self.critic_scheduler = CosineAnnealingWarmRestarts(self.critic_optimizer, T_0=1_000_000, eta_min=1e-5)

    def choose_action(self, obs, explor_noise_std):
        """
        Select an action based on the current policy (actor network).
        
        Args:
            obs: The current observation (state) of the agent.
            is_eval: Flag indicating if the agent is in evaluation mode (no exploration).
            
        Returns:
            The selected action based on the current policy.
        """
        obs = torch.unsqueeze(torch.tensor(obs, dtype=torch.float), 0).to(self.device)
        action = self.actor(obs).cpu().data.numpy().flatten()
        explor_noise = np.random.normal(0, explor_noise_std, size=self.action_dim)  # exploration noise
        return (action + explor_noise).clip(-self.max_action, self.max_action) # Return the action as a numpy array

    def train(self, replay_buffer, agent_n, env):
        """
        Perform one training step of the TD3 algorithm.
        
        Args:
            replay_buffer: A buffer storing past experiences for off-policy learning.
            agent_n: List of agents in the environment.
            env: The environment in which the agents interact.
            
        Returns:
            critic_loss: Loss for the critic network.
            actor_loss: Loss for the actor network.
        """
        self.total_it += 1  # Track training iterations

        # Randomly sample a batch of transitions from an experience replay buffer
        batch_obs_n, batch_act_n, batch_rwd_n, batch_obs_next_n, batch_done_n = replay_buffer.sample()

        # Extract current agent's experiences
        batch_obs, batch_act, batch_rwd = batch_obs_n[self.agent_id], batch_act_n[self.agent_id], batch_rwd_n[self.agent_id]
        batch_obs_next, batch_done = batch_obs_next_n[self.agent_id], batch_done_n[self.agent_id]

        """
        Q-Learning side of TD3 with critic networks
        """
        with torch.no_grad():  # No gradient computation for target values
            if self.is_MODUL_CTDE:  # If using modular approach with centralized critic networks
                batch_act_next_n = []
                for i in range(self.N):
                    batch_act_next = agent_n[i].actor_target(batch_obs_next_n[i])

                    # Add clipped noise to target actions for 'target policy smoothing'
                    noise = (torch.randn_like(batch_act_next) * self.target_noise).clamp(-self.noise_clip, self.noise_clip)
                    
                    # Compute target actions from a target policy network
                    batch_act_next = (batch_act_next + noise).clamp(-self.max_action, self.max_action)
                    batch_act_next_n.append(batch_act_next)
                
                # Get target Q-values, Q_targ(s', a')
                target_Q1, target_Q2 = self.critic_target(batch_obs_next_n, batch_act_next_n)
            else:
                batch_act_next = self.actor_target(batch_obs_next)
                
                # Add clipped noise to target actions for 'target policy smoothing'
                noise = (torch.randn_like(batch_act_next) * self.target_noise).clamp(-self.noise_clip, self.noise_clip)
                
                # Compute target actions from a target policy network
                batch_act_next = (batch_act_next + noise).clamp(-self.max_action, self.max_action)
                
                # Get target Q-values, Q_targ(s', a')
                target_Q1, target_Q2 = self.critic_target(batch_obs_next, batch_act_next)

            # Use a smaller target Q-value
            target_Q = torch.min(target_Q1, target_Q2)

            # Compute targets, y(r, s', d)
            target_Q = batch_rwd + self.discount * (1 - batch_done) * target_Q  # Bellman update for Q-values

        # Get current Q-values, Q1(s, a) and Q2(s, a)
        current_Q1, current_Q2 = self.critic(batch_obs_n if self.is_MODUL_CTDE else batch_obs, 
                                             batch_act_n if self.is_MODUL_CTDE else batch_act)  # shape: (batch_size,1)

        # Compute critic loss using a mean-squared Bellman error (MSBE)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        if self.use_equiv:  # Apply spectral norm regularization
            critic_loss += 1e-8*self.critic.spectral_norm_regularization()

        # Update critic networks by gradient descent
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.use_clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_max_norm)
        self.critic_optimizer.step()
        self.critic_scheduler.step()

        """
        Policy learning side of TD3 with actor networks
        """
        # Update policy less frequently than Q-function for 'delayed policy updates'
        if self.total_it % self.policy_update_freq == 0:

            # Compute \pi(s)
            if self.is_MODUL_CTDE:
                batch_act_n = [agent_n[i].actor(batch_obs_n[i]).clamp(-self.max_action, self.max_action) for i in range(self.N)]
            else:
                batch_act = (self.actor(batch_obs)).clamp(-self.max_action, self.max_action)

            # Set actor loss s.t. Q(s,\pi(s)) approximates \max_a Q(s,a)
            if self.use_equiv:
                current_Q1, current_Q2 = self.critic(batch_obs_n if self.is_MODUL_CTDE else batch_obs, 
                                                     batch_act_n if self.is_MODUL_CTDE else batch_act)  # shape: (batch_size,1)
                actor_loss = -current_Q1.mean()  # Only use Q1
                actor_loss += 1e-5*self.actor.spectral_norm_regularization()
            else:
                actor_loss = -self.critic.Q1(batch_obs_n if self.is_MODUL_CTDE else batch_obs, 
                                             batch_act_n if self.is_MODUL_CTDE else batch_act).mean()  # Only use Q1

            # Regularizing action policies for smooth and efficient control
            actor_loss = policy_regularization(self.agent_id, self.actor, actor_loss, batch_obs, batch_obs_next, env, self.args)

            # Update policy by gradient ascent
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.use_clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_max_norm)
            self.actor_optimizer.step()
            self.actor_scheduler.step()

            # Softly update the target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        if self.save_tensorboard:
            return critic_loss.item(), actor_loss.item() if self.total_it % self.policy_update_freq == 0 else None

    def save(self, rl_algo, framework, total_steps, agent_id, seed):
        torch.save(self.actor.state_dict(), \
                   "./models/{}_{}_{}k_steps_agent_{}_{}.pth".format(rl_algo, framework, total_steps/1000, agent_id, seed))

    def save_solved_model(self, rl_algo, framework, total_steps, agent_id, seed):
        torch.save(self.actor.state_dict(), \
                   "./models/{}_{}_{}k_steps_agent_{}_solved_{}.pth".format(rl_algo, framework, total_steps/1000, agent_id, seed))

    def load(self, rl_algo, framework, total_steps, agent_id, seed):
        if self.device == torch.device("cuda"):
            self.actor.load_state_dict(torch.load( \
                "./models/{}_{}_{}k_steps_agent_{}_{}.pth".format(rl_algo, framework, total_steps/1000, agent_id, seed)))
        else:
            self.actor.load_state_dict(torch.load( \
                "./models/{}_{}_{}k_steps_agent_{}_{}.pth".format(rl_algo, framework, total_steps/1000, agent_id, seed), map_location=torch.device('cpu')))

    def load_solved_model(self, rl_algo, framework, total_steps, agent_id, seed):
        if self.device == torch.device("cuda"):
            self.actor.load_state_dict(torch.load( \
                "./models/{}_{}_{}k_steps_agent_{}_solved_{}.pth".format(rl_algo, framework, total_steps/1000, agent_id, seed)))
        else:
            self.actor.load_state_dict(torch.load( \
                "./models/{}_{}_{}k_steps_agent_{}_solved_{}.pth".format(rl_algo, framework, total_steps/1000, agent_id, seed), map_location=torch.device('cpu')))