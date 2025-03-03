import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from algos.networks.emlp_utils import get_equiv_state, get_equiv_state_batch

LOG_SIG_MAX, LOG_SIG_MIN = 2, -20  # Define limits for log standard deviation
epsilon = 1e-6  # Small value to prevent numerical instability

# Apply Xavier initialization to linear layers
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class EMLP_Actor_TD3(nn.Module):
    def __init__(self, args, agent_id):
        """
        Equivariant Actor network for TD3.
        This network maps an agent's observations to actions while preserving rotational symmetry.

        Args:
            args: Namespace containing hyperparameters and configurations.
            agent_id: Unique identifier for the agent.
        """
        super(EMLP_Actor_TD3, self).__init__()
        self.fc1 = nn.Linear(args.obs_dim_n[agent_id]-1, args.actor_hidden_dim[agent_id])
        self.fc2 = nn.Linear(args.actor_hidden_dim[agent_id], args.actor_hidden_dim[agent_id])
        self.fc3 = nn.Linear(args.actor_hidden_dim[agent_id], args.action_dim_n[agent_id])

    def forward(self, state):
        """
        Forward pass of the TD3 actor network.

        Args:
            state (Tensor): The input state observation.

        Returns:
            Tensor: The action output in the range [-1, 1].
        """
        # Get (de-rotated) equivalent states
        if state.shape[0] == 1:  # Single-sample case
            state_equiv = get_equiv_state(state)
        else:  # Batch case
            state_equiv = get_equiv_state_batch(state)
        # Pass the transformed state through the network
        action_equiv = F.relu(self.fc1(state_equiv))
        action_equiv = F.relu(self.fc2(action_equiv))
        action_equiv = torch.tanh(self.fc3(action_equiv)) # [-1, 1]

        return action_equiv


class EMLP_Actor_SAC(nn.Module):
    def __init__(self, args, agent_id):
        """
        Equivariant Actor network for SAC.
        This network maps an agent's observations to actions while preserving rotational symmetry.

        Args:
            args: Namespace containing hyperparameters and configurations.
            agent_id: Unique identifier for the agent.
        """        
        super(EMLP_Actor_SAC, self).__init__()
        self.fc1 = nn.Linear(args.obs_dim_n[agent_id]-1, args.actor_hidden_dim[agent_id])
        self.fc2 = nn.Linear(args.actor_hidden_dim[agent_id], args.actor_hidden_dim[agent_id])
        self.mean_linear = nn.Linear(args.actor_hidden_dim[agent_id], args.action_dim_n[agent_id])
        self.log_std_linear = nn.Linear(args.actor_hidden_dim[agent_id], args.action_dim_n[agent_id])

        self.apply(weights_init_)

    def forward(self, state):
        """
        Forward pass of the SAC actor network.

        Args:
            state (Tensor): The input state observation.

        Returns:
            Tensor: The action output in the range [-1, 1].
        """
        # Get (de-rotated) equivalent states
        if state.shape[0] == 1:  # Single-sample case 
            state_equiv = get_equiv_state(state)
            state_equiv = state_equiv.view(1,-1)
        else:  # Batch case
            state_equiv = get_equiv_state_batch(state)
        # Pass the transformed state through the network
        action_equiv = F.relu(self.fc1(state_equiv))
        action_equiv = F.relu(self.fc2(action_equiv))

        mean = self.mean_linear(action_equiv)
        log_std = self.log_std_linear(action_equiv)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)  # Clip to avoid extreme values
        return mean, log_std

    def sample(self, x):
        """
        Samples an action using the reparameterization trick to allow backpropagation.
        
        Args:
            x: Input observation.

        Returns:
            Sampled action, its log probability, and the deterministic action (tanh(mean)).
        """
        mean, log_std = self.forward(x)
        std = log_std.exp()  # Convert log std to standard deviation
        normal = Normal(mean, std)  # Create normal distribution

        # rsample: sampling using reparameterization trick (mean + std * epsilon, where epsilon ~ N(0,1))
        x_t = normal.rsample()  # To prevent non-differentiability
        action = torch.tanh(x_t)  # Squash output using tanh to keep it within valid action range
        
        # Compute log probability of the action while accounting for squashing function
        log_prob = normal.log_prob(x_t)  # Compute the log probability density of x_t under the normal distribution
        log_prob -= torch.log((1 - action.pow(2)) + epsilon)  # Correct log prob for tanh transformation
        log_prob = log_prob.sum(1, keepdim=True)  # Sum over action dimensions
        mean = torch.tanh(mean)  # Deterministic action (used during evaluation)

        return action, log_prob, mean
    

class EMLP_Critic(nn.Module):
    def __init__(self, args, agent_id):
        """
        Equivariant Critic network, implementing Double-Q Learning.
        The critic estimates the state-action value function Q(s, a), using two separate Q-networks.

        Args:
            args: Namespace containing hyperparameters and configurations.
            agent_id: Unique identifier for the agent.
        """
        super(EMLP_Critic, self).__init__()
        # Q1 architecture:
        self.fc1 = nn.Linear(args.obs_dim_n[agent_id]-1+ args.action_dim_n[agent_id], args.critic_hidden_dim)
        self.fc2 = nn.Linear(args.critic_hidden_dim, args.critic_hidden_dim)
        self.fc3 = nn.Linear(args.critic_hidden_dim, 1)

        # Q2 architecture:
        self.fc4 = nn.Linear(args.obs_dim_n[agent_id]-1 + args.action_dim_n[agent_id], args.critic_hidden_dim)
        self.fc5 = nn.Linear(args.critic_hidden_dim, args.critic_hidden_dim)
        self.fc6 = nn.Linear(args.critic_hidden_dim, 1)

    def forward(self, state, action_equiv):
        """
        Forward pass of the critic network.

        Args:
            state: The input state observation.
            action_equiv: The equivalent action taken by the agent.

        Returns:
            The estimated Q-values from both Q-networks (Q1, Q2).
        """

        # Get (de-rotated) equivalent states
        state_equiv = get_equiv_state_batch(state)

        q1 = F.relu(self.fc1(torch.cat([state_equiv, action_equiv], dim=1)))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        q2 = F.relu(self.fc4(torch.cat([state_equiv, action_equiv], dim=1)))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        return q1, q2

    def Q1(self, state, action_equiv):
        """
        Compute only Q1-value (used in TD3).

        Args:
            state: The input state observation.
            action_equiv: The equivalent action taken by the agent.

        Returns:
            The estimated Q1-value.
        """

        # Get (de-rotated) equivalent states
        state_equiv = get_equiv_state_batch(state)

        q1 = F.relu(self.fc1(torch.cat([state_equiv, action_equiv], dim=1)))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        return q1