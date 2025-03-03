import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX, LOG_SIG_MIN = 2, -20  # Define limits for log standard deviation
epsilon = 1e-6  # Small value to prevent numerical instability

# Apply Xavier initialization to linear layers
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class MLP_Actor_SAC(nn.Module):
    def __init__(self, args, agent_id):
        """
        MLP-based actor network for SAC.
        Gaussian Policy Network that outputs a mean and log standard deviation of an action distribution.

        Args:
            args: Namespace containing hyperparameters and configurations.
            agent_id: Unique identifier for the agent.
        """
        super(MLP_Actor_SAC, self).__init__()
        self.fc1 = nn.Linear(args.obs_dim_n[agent_id], args.actor_hidden_dim[agent_id])  # Input layer
        self.fc2 = nn.Linear(args.actor_hidden_dim[agent_id], args.actor_hidden_dim[agent_id])  # Hidden layer
        # Output layers for mean and log standard deviation of the action distribution
        self.mean_linear = nn.Linear(args.actor_hidden_dim[agent_id], args.action_dim_n[agent_id])  # Output layer
        self.log_std_linear = nn.Linear(args.actor_hidden_dim[agent_id], args.action_dim_n[agent_id])  # Output layer

        # Apply weight initialization
        self.apply(weights_init_)

    def forward(self, x):
        """
        Forward pass computes the mean and log standard deviation of the action distribution.
        Uses ReLU activations for hidden layers.

        Args:
            x: Input observation.

        Returns:
            Mean and log standard deviation of the action distribution.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
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