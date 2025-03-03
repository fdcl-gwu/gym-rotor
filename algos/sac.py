import os
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from algos.networks.mlp import MLP_Actor_SAC, MLP_Critic
from algos.networks.emlp import EMLP_Actor_SAC, EMLP_Critic
from algos.utils.policy_regularization import policy_regularization

class SAC(object):
    def __init__(self, args, agent_id):
        """
        Soft Actor-Critic (SAC) class for training RL agents.
        
        Args:
            args: Namespace containing hyperparameters and configurations.
            agent_id: Unique identifier for the agent.
        """
        self.__dict__.update(vars(args))  # convert args to dictionary, e.g., self.discount = args.discount
        self.args = args
        self.agent_id = agent_id
        self.obs_dim = args.obs_dim_n[agent_id]  # Observation space dimension for each agent
        self.action_dim = args.action_dim_n[agent_id]  # Action space dimension for each agent
        self.actor_hidden_dim = args.actor_hidden_dim[agent_id]  # Actor network hidden layer dimension
        self.lr_a = args.lr_a[agent_id]  # Learning rate for the actor networks
        self.lr_c = args.lr_c[agent_id]  # Learning rate for the critic networks
        self.total_it = 0  # Iteration counter

        # Create an instance of the actor networks
        if self.use_equiv:  # Using equivariant reinforcement learning
            self.actor = EMLP_Actor_SAC(args, agent_id).to(self.device)
        else:  # Use standard MLP-based actor network
            self.actor = MLP_Actor_SAC(args, agent_id).to(self.device)

        # If the model is in training mode
        if not self.test_model:
            # Create an instance of the critic networks
            if self.use_equiv:  # Using equivariant reinforcement learning
                self.critic = EMLP_Critic(args, agent_id).to(self.device)
            else:  # Use standard MLP-based actor network
                self.critic = MLP_Critic(args, agent_id).to(self.device)

            # Create target networks for stability in training
            self.actor_target = copy.deepcopy(self.actor)
            self.critic_target = copy.deepcopy(self.critic)

            # Initialize optimizers for actor and critic networks
            self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=self.lr_a)
            self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=self.lr_c)

            # Use CosineAnnealingWarmRestarts to adjust learning rate dynamically
            self.actor_scheduler = CosineAnnealingWarmRestarts(self.actor_optimizer, T_0=1_000_000, eta_min=1e-5)
            self.critic_scheduler = CosineAnnealingWarmRestarts(self.critic_optimizer, T_0=1_000_000, eta_min=1e-5)

        # Automatic entropy tuning (optional)
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(self.action_dim).to(self.device)).item()  # Target Entropy = −dim(A) 
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.AdamW([self.log_alpha], lr=self.lr_a)

    def choose_action(self, obs, is_eval=False):
        """
        Select an action based on the current policy (actor network).
        
        Args:
            obs: The current observation (state) of the agent.
            is_eval: Flag indicating if the agent is in evaluation mode (no exploration).
            
        Returns:
            The selected action based on the current policy.
        """
        obs = torch.unsqueeze(torch.tensor(obs, dtype=torch.float), 0).to(self.device)
        if is_eval == False:
            action, _, _ = self.actor.sample(obs)  # Sample action using the actor (with exploration)
        else:
            _, _, action = self.actor.sample(obs)  # Evaluate the actor (no exploration)
        return action.detach().cpu().numpy()[0] # Return the action as a numpy array

    def train(self, replay_buffer, env):
        """
        Perform one training step of the SAC algorithm.
        
        Args:
            replay_buffer: A buffer storing past experiences for off-policy learning.
            env: The environment in which the agents interact.
            
        Returns:
            critic_loss: Loss for the critic network.
            actor_loss: Loss for the actor network.
            alpha_loss: Loss for entropy regularization (if automatic tuning is enabled).
            alpha_tlogs: The value of the entropy temperature parameter.
        """
        self.total_it += 1  # Track training iterations
        
        # Randomly sample a batch of transitions from an experience replay buffer
        batch_obs_n, batch_act_n, batch_rwd_n, batch_obs_next_n, batch_done_n = replay_buffer.sample()

        # Extract current agent's experiences
        batch_obs, batch_act, batch_rwd = batch_obs_n[self.agent_id], batch_act_n[self.agent_id], batch_rwd_n[self.agent_id]
        batch_obs_next, batch_done = batch_obs_next_n[self.agent_id], batch_done_n[self.agent_id]

        """
        Q-Learning side of SAC with critic networks
        """
        with torch.no_grad():  # No gradient computation for target values
            batch_act_next, batch_act_next_log, _ = self.actor.sample(batch_obs_next)
            
            # Get target Q-values, Q_targ(s', a')
            target_Q1, target_Q2 = self.critic_target(batch_obs_next, batch_act_next)
                
            # Compute targets for the Q functions
            target_Q = torch.min(target_Q1, target_Q2) - self.sac_alpha * batch_act_next_log
            target_Q = batch_rwd + self.discount * (1 - batch_done) * target_Q  # Bellman update for Q-values
        
        # Get current Q-values, Q1(s, a) and Q2(s, a)
        current_Q1, current_Q2 = self.critic(batch_obs, batch_act)  # Two Q-functions to mitigate positive bias in the policy improvement step

        # Compute critic loss using a mean-squared Bellman error (MSBE)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) 

        # Update critic networks by gradient descent
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.use_clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_max_norm)
        self.critic_optimizer.step()
        self.critic_scheduler.step()

        """
        Policy learning side of SAC with actor networks
        """
        batch_act, batch_act_log, _ = self.actor.sample(batch_obs)

        # Calculate Q-values
        current_Q1, current_Q2 = self.critic(batch_obs, batch_act)  # shape: (batch_size,1)
        min_current_Q = torch.min(current_Q1, current_Q2)
        
        # Compute the actor loss
        actor_loss = -(min_current_Q - (self.sac_alpha * batch_act_log)).mean() 

        # Regularizing action policies for smooth and efficient control
        actor_loss = policy_regularization(self.agent_id, self.actor, actor_loss, batch_obs, batch_obs_next, env, self.args)
        
        # Update policy by gradient ascent
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.use_clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_max_norm)
        self.actor_optimizer.step()
        self.actor_scheduler.step()

        # Automatic entropy tuning (optional)
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (self.target_entropy + batch_act_log).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.sac_alpha = self.log_alpha.exp()
            alpha_log = self.sac_alpha.clone() # For logging in Tensorboard
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_log = torch.tensor(self.sac_alpha) # For logging in Tensorboard

        if self.total_it % self.policy_update_freq == 0:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
    def save(self, framework, total_steps, agent_id, seed):
        torch.save(self.actor.state_dict(), "./models/{}_{}k_steps_agent_{}_{}.pth".format(framework, total_steps/1000, agent_id, seed))

    def save_solved_model(self, framework, total_steps, agent_id, seed):
        torch.save(self.actor.state_dict(), "./models/{}_{}k_steps_agent_{}_solved_{}.pth".format(framework, total_steps/1000, agent_id, seed))

    def load(self, framework, total_steps, agent_id, seed):
        if self.device == torch.device("cuda"):
            self.actor.load_state_dict(torch.load("./models/{}_{}k_steps_agent_{}_{}.pth".format(framework, total_steps/1000, agent_id, seed)))
        else:
            self.actor.load_state_dict(torch.load("./models/{}_{}k_steps_agent_{}_{}.pth".format(framework, total_steps/1000, agent_id, seed), map_location=torch.device('cpu')))

    def load_solved_model(self, framework, total_steps, agent_id, seed):
        if self.device == torch.device("cuda"):
            self.actor.load_state_dict(torch.load("./models/{}_{}k_steps_agent_{}_solved_{}.pth".format(framework, total_steps/1000, agent_id, seed)))
        else:
            self.actor.load_state_dict(torch.load("./models/{}_{}k_steps_agent_{}_solved_{}.pth".format(framework, total_steps/1000, agent_id, seed), map_location=torch.device('cpu')))