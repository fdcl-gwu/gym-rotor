import copy
import math
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from algos.ppo.ppo_mlp import MLP_Actor_PPO, MLP_Critic, MLP_Critic_CTDE
from algos.ppo.ppo_emlp import EMLP_MONO_Actor_PPO, EMLP_MODUL1_Actor_PPO, EMLP_MODUL2_Actor_PPO, \
    EMLP_MONO_Critic_PPO, EMLP_MODUL1_CTDE_Critic_PPO, EMLP_MODUL2_CTDE_Critic_PPO, EMLP_MODUL1_DTDE_Critic_PPO, EMLP_MODUL2_DTDE_Critic_PPO
from algos.policy_regularization import policy_regularization


class PPO(object):
    def __init__(self, args, agent_id):
        """
        Proximal Policy Optimization (PPO) class for training RL agents.
        
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
                self.actor = EMLP_MONO_Actor_PPO(args, agent_id).to(self.device)
            elif self.framework == "MODUL":
                # Assign different modular actor networks based on agent_id
                if self.agent_id == 0:
                    self.actor = EMLP_MODUL1_Actor_PPO(args, agent_id).to(self.device) 
                elif self.agent_id == 1:
                    self.actor = EMLP_MODUL2_Actor_PPO(args, agent_id).to(self.device)
        else:  # Use standard MLP-based actor network
            self.actor = MLP_Actor_PPO(args, agent_id).to(self.device)

        # If the model is in training mode
        if not self.test_model:
            # Create an instance of the critic networks
            if self.use_equiv:  # Using equivariant reinforcement learning
                if self.framework == "MONO":
                    self.critic = EMLP_MONO_Critic_PPO(args, agent_id).to(self.device)
                elif self.framework == "MODUL":
                    if self.module_training == "CTDE":  # Centralized Training, Decentralized Execution
                        if self.agent_id == 0:
                            self.critic = EMLP_MODUL1_CTDE_Critic_PPO(args, agent_id).to(self.device)
                        elif self.agent_id == 1:
                            self.critic = EMLP_MODUL2_CTDE_Critic_PPO(args, agent_id).to(self.device)
                    elif self.module_training == "DTDE":  # Decentralized Training, Decentralized Execution
                        if self.agent_id == 0:
                            self.critic = EMLP_MODUL1_DTDE_Critic_PPO(args, agent_id).to(self.device)
                        elif self.agent_id == 1:
                            self.critic = EMLP_MODUL2_DTDE_Critic_PPO(args, agent_id).to(self.device)
            else:  # Use standard MLP-based actor network
                if self.framework == "MONO" or self.is_MODUL_DTDE:
                    self.critic = MLP_Critic(args, agent_id).to(self.device)
                elif self.is_MODUL_CTDE:
                    self.critic = MLP_Critic_CTDE(args, agent_id).to(self.device)  # Centralized Training, Decentralized Execution

            # Create target networks for stability in training
            self.actor_target = copy.deepcopy(self.actor)
            self.critic_target = copy.deepcopy(self.critic)

            # Initialize optimizers for actor and critic networks
            self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=self.lr_a)
            self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=self.lr_c)

            # Use CosineAnnealingWarmRestarts to adjust learning rate dynamically
            self.actor_scheduler = CosineAnnealingWarmRestarts(self.actor_optimizer, T_0=1_000_000, eta_min=1e-5)
            self.critic_scheduler = CosineAnnealingWarmRestarts(self.critic_optimizer, T_0=1_000_000, eta_min=1e-5)

    def choose_action(self, obs, is_eval):
        """
        Select an action based on the current policy (actor network).
        
        Args:
            obs: The current observation (state) of the agent.
            is_eval: Flag indicating if the agent is in evaluation mode (no exploration).
            
        Returns:
            The selected action based on the current policy.
        """
        obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        if is_eval == False:
            action_distribution = self.actor.get_dist(obs)
            action = action_distribution.sample()  # Sample action using the actor (with exploration)
            action = torch.clamp(action, -self.max_action, self.max_action)
            logprob = action_distribution.log_prob(action).detach().cpu().numpy().flatten()
            return action.cpu().numpy()[0], logprob
        else:
            action = self.actor(obs).clamp(-self.max_action, self.max_action) # Evaluate the actor (no exploration)
            return action.detach().cpu().numpy()[0], None
        
    def train(self, replay_buffer, agent_n, env):
        """
        Perform one training step of the PPO algorithm.
        
        Args:
            replay_buffer: A buffer storing past experiences for off-policy learning.
            agent_n: List of agents in the environment.
            env: The environment in which the agents interact.
            
        Returns:
            critic_loss: Loss for the critic network.
            actor_loss: Loss for the actor network.
        """
        # Randomly sample a batch of transitions from an experience buffer
        batch_obs_n, batch_act_n, batch_rwd_n, batch_obs_next_n, batch_done_n, batch_logprob_n = replay_buffer.sample()
        
        # Extract current agent's experiences
        batch_obs, batch_act, batch_rwd = batch_obs_n[self.agent_id], batch_act_n[self.agent_id], batch_rwd_n[self.agent_id]
        batch_obs_next, batch_done, batch_logprob = batch_obs_next_n[self.agent_id], batch_done_n[self.agent_id], batch_logprob_n[self.agent_id]

        # Compute Advantage and Temporal Difference (TD) target using Generalized Advantage Estimation (GAE)
        with torch.no_grad():
            if self.is_MODUL_CTDE:  # If using modular approach with centralized critic networks
                current_V = self.critic(batch_obs_n)  # Value estimates for current obs_n for centralized critic networks
                next_V = self.critic(batch_obs_next_n)  # Value estimates for next obs_n for centralized critic networks
            else:
                current_V = self.critic(batch_obs)  # Value estimates for current obs
                next_V = self.critic(batch_obs_next)  # Value estimates for next obs

            # Compute Temporal Difference (TD) error (aka delta), which measures how much the critic's prediction differs from actual rewards
            td_errors = batch_rwd + self.discount * next_V * (1 - batch_done) - current_V
            td_errors = td_errors.cpu().flatten().numpy()
            advantages = [0]  # Initialize advantage list for advantage estimation

            # Compute Generalized Advantage Estimation (GAE)
            for delta, done in zip(td_errors[::-1], batch_done.cpu().flatten().numpy()[::-1]):
                advantage = delta + self.discount * (1 - done) * self.GAE_lambda * advantages[-1]
                advantages.append(advantage)
            advantages.reverse()
            advantages = copy.deepcopy(advantages[:-1])  # Remove last element as it's extra
            advantages = torch.tensor(advantages).unsqueeze(1).float().to(self.device)
            td_targets = advantages + current_V  # Compute final TD target for critic learning
            advantages = (advantages - advantages.mean()) / ((advantages.std() + 1e-4))  # Normalize advantages for stable training

        # Decaying entropy coefficient over time to balance exploration and exploitation
        self.entropy_coef*=self.entropy_coef_decay

        # Mini-batch training using PPO updates to optimize actor and critic
        num_actor_update_steps = int(math.ceil(batch_obs.shape[0] / self.actor_batch_size))
        num_critic_update_steps = int(math.ceil(batch_obs.shape[0] / self.critic_batch_size))
        for i in range(self.K_epochs):

            # Shuffle experience batch
            indices = np.arange(batch_obs.shape[0])
            np.random.shuffle(indices)
            indices = torch.LongTensor(indices).to(self.device)
            batch_obs, batch_act, batch_obs_next, td_targets, advantages, batch_logprob = \
                batch_obs[indices].clone(), batch_act[indices].clone(), batch_obs_next[indices].clone(), \
                td_targets[indices].clone(), advantages[indices].clone(), batch_logprob[indices].clone()
            if self.is_MODUL_CTDE:  # If using modular approach with centralized critic networks
                batch_obs_n = [batch_obs[indices].clone() for batch_obs in batch_obs_n]
            
            # Update the actor networks using PPO clipped objective
            for i in range(num_actor_update_steps):
                index_slice = slice(i * self.actor_batch_size, min((i + 1) * self.actor_batch_size, batch_obs.shape[0]))
                action_distribution = self.actor.get_dist(batch_obs[index_slice])
                entropy = action_distribution.entropy().sum(1, keepdim=True)  # Entropy for encouraging exploration
                log_probs = action_distribution.log_prob(batch_act[index_slice])
                ratio = torch.exp(log_probs.sum(1,keepdim=True) - batch_logprob[index_slice].sum(1,keepdim=True))  # batch_act/b == exp(log(batch_act)-log(b))

                # Clipped surrogate loss for PPO to prevent overly large updates
                surrogate_loss1 = ratio * advantages[index_slice]
                surrogate_loss2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * advantages[index_slice]
                actor_loss = -(torch.min(surrogate_loss1, surrogate_loss2) + self.entropy_coef * entropy).mean()
                if self.use_equiv:  # Apply spectral norm regularization
                    actor_loss += 1e-5*self.actor.spectral_norm_regularization()

                # Regularizing action policies for smooth and efficient control
                actor_loss = policy_regularization(self.agent_id, self.actor, actor_loss, batch_obs[index_slice], batch_obs_next[index_slice], env, self.args)

                # Update policy by gradient ascent
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                if self.use_clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_max_norm)
                self.actor_optimizer.step()
                self.actor_scheduler.step()

            # Update critic networks by minimizing TD error
            for i in range(num_critic_update_steps):
                index_slice = slice(i * self.critic_batch_size, min((i + 1) * self.critic_batch_size, batch_obs.shape[0]))
                if self.is_MODUL_CTDE:  # If using modular approach with centralized critic networks
                    batch_obs_n_subset = [batch_obs[index_slice] for batch_obs in batch_obs_n]
                    critic_loss = (self.critic(batch_obs_n_subset) - td_targets[index_slice]).pow(2).mean()
                else:
                    critic_loss = (self.critic(batch_obs[index_slice]) - td_targets[index_slice]).pow(2).mean()
                
                # L2 regularization for critic networks
                for name, param in self.critic.named_parameters():
                    if 'weight' in name:
                        critic_loss += param.pow(2).sum() * self.l2_reg
                if self.use_equiv:  # Apply spectral norm regularization
                    critic_loss += 1e-10*self.critic.spectral_norm_regularization()

                # Update critic networks by gradient descent
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                if self.use_clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_max_norm)
                self.critic_optimizer.step()
                self.critic_scheduler.step()

        return critic_loss.item(), actor_loss.item()

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