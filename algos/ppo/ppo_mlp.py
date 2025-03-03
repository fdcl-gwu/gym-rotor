import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class MLP_Actor_PPO(nn.Module):
    def __init__(self, args, agent_id, log_std=0):
        """
        Actor network for PPO (Proximal Policy Optimization).
        This network outputs a mean action and a log standard deviation for a Gaussian policy.
        
        Args:
            args: Namespace containing hyperparameters and configurations.
            agent_id: Unique identifier for the agent.
            log_std: Initial log standard deviation for the action distribution.
        """
        super(MLP_Actor_PPO, self).__init__()

        self.fc1 = nn.Linear(args.obs_dim_n[agent_id], args.actor_hidden_dim[agent_id])  # Input layer
        self.fc2 = nn.Linear(args.actor_hidden_dim[agent_id], args.actor_hidden_dim[agent_id])  # Hidden layer

        # Output layers for mean and log standard deviation of the action distribution
        self.mean_linear = nn.Linear(args.actor_hidden_dim[agent_id], args.action_dim_n[agent_id])  # Output layer
        self.log_std = nn.Parameter(torch.ones(1, args.action_dim_n[agent_id]) * log_std)

        # Apply weight initialization
        self.mean_linear.weight.data.mul_(0.1)
        self.mean_linear.bias.data.mul_(0.0)

    def forward(self, x):
        """
        Forward pass through the actor network to compute the mean action.
        
        Args:
            x: Input state tensor.

        Returns:
            mean: Mean of the Gaussian action distribution.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = torch.tanh(self.mean_linear(x))

        return mean

    def get_dist(self, x):
        """
        Compute the Gaussian action distribution given the input state.
        
        Args:
            x: Input state tensor.

        Returns:
            normal: A Normal distribution representing the policy.
        """
        mean = self.forward(x)  # Compute mean action
        log_std = self.log_std.expand_as(mean)  # Expand log_std to match mean's shape
        std = torch.exp(log_std)  # Convert log standard deviation to standard deviation
        normal = Normal(mean, std)  # Create a normal distribution

        return normal


class MLP_Critic(nn.Module):
    def __init__(self, args, agent_id):
        """
        Critic network for single-agent PPO.
        This network estimates the value function V(s), which helps in advantage estimation.
        
        Args:
            args: Namespace containing hyperparameters and configurations.
            agent_id: Unique identifier for the agent.
        """
        super(MLP_Critic, self).__init__()

        self.fc1 = nn.Linear(args.obs_dim_n[agent_id], args.critic_hidden_dim)  # Input layer
        self.fc2 = nn.Linear(args.critic_hidden_dim, args.critic_hidden_dim)  # Hidden layer
        self.fc3 = nn.Linear(args.critic_hidden_dim, 1)  # Output layer (V-value)

    def forward(self, state):
        """
        Forward pass through the critic network to estimate the state value.
        
        Args:
            state: Input state tensor.

        Returns:
            v: Estimated value function V(s).
        """
        v = F.tanh(self.fc1(state))
        v = F.tanh(self.fc2(v))
        v = self.fc3(v)
        return v
    

class MLP_Critic_CTDE(nn.Module):
    def __init__(self, args, agent_id):
        """
        Centralized critic network for CTDE (Centralized Training with Decentralized Execution).
        This network estimates the value function V(s) using information from all agents.
        
        Args:
            args: Namespace containing hyperparameters and configurations.
            agent_id: Unique identifier for the agent.
        """
        super(MLP_Critic_CTDE, self).__init__()

        self.fc1 = nn.Linear(sum(args.obs_dim_n), args.critic_hidden_dim)  # Input layer
        self.fc2 = nn.Linear(args.critic_hidden_dim, args.critic_hidden_dim)  # Hidden layer
        self.fc3 = nn.Linear(args.critic_hidden_dim, 1)  # Output layer (V-value)

    def forward(self, state):
        """
        Forward pass through the centralized critic network to estimate the state value.
        
        Args:
            state: List of state tensors from all agents.
            
        Returns:
            v: Estimated value function V(s) using global state information.
        """
        state = torch.cat(state, dim=1)  # Concatenate state tensors from all agents
        v = F.tanh(self.fc1(state))
        v = F.tanh(self.fc2(v))
        v = self.fc3(v)
        return v