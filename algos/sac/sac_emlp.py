import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from algos.emlp_torch.groups import *
from algos.emlp_torch.reps import *
from algos.emlp_torch.nn import EMLPBlock, Linear, uniform_rep
from algos.emlp_torch.groups import SO2eR3, Trivial
from algos.emlp_torch.reps import Vector, Scalar
from algos.spectral_norm_regularization import spectral_norm

LOG_SIG_MAX, LOG_SIG_MIN = 2, -20  # Define limits for log standard deviation
epsilon = 1e-6  # Small value to prevent numerical instability

# Apply Xavier initialization to linear layers
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

###################################################################################
############## Monolithic Architecture ############################################
###################################################################################
class EMLP_MONO_Actor_SAC(nn.Module):
    def __init__(self, args, agent_id, hidden_num=2):
        """
        Equivariant MLP-based monolithic actor network for TD3.

        Args:
            args: Namespace containing hyperparameters and configurations.
            agent_id: Unique identifier for the agent.
            hidden_num : Number of hidden layers.

        Group representation:
            Input: ρ_θ(g)ex, ρ_θ(g)eIx, ρ_θ(g)ev, ρ_θ(g)b1, ρ_θ(g)b2, ρ_θ(g)b3, ρ_e(g)eb1, ρ_e(g)eIb1, ρ_e(g)eΩ
                                                  <--------- ρ_θ(g)R ---------> 
            Output: ρ_e(g)f, ρ_e(g)M
        """
        super().__init__()
        self.device = args.device
        self.hidden_num = hidden_num

        # Define groups
        self.G_SO2eR3 = SO2eR3().to(self.device)  # ρ_θ(g) for 3D rotations
        self.G_trivialR1 = Trivial(1).to(self.device)  # ρ_e(g) for scalar values
        self.G_trivialR3 = Trivial(3).to(self.device)  # ρ_e(g) for 3D vector values
        
        # Input representation: ρ_θ(g)ex, ρ_θ(g)eIx, ρ_θ(g)ev, ρ_θ(g)b1, ρ_θ(g)b2, ρ_θ(g)b3, ρ_e(g)eb1, ρ_e(g)eIb1, ρ_e(g)eΩ
        self.rep_in  = Vector(self.G_SO2eR3)*6 + Scalar(self.G_trivialR1)*2 + Vector(self.G_trivialR3)
        # Output representation: ρ_e(g)f, ρ_e(g)M
        self.rep_out = Scalar(self.G_trivialR1) + Vector(self.G_trivialR3)
        
        # Define the hidden layers based on the number of hidden layers and size
        middle_layers = self.hidden_num*[uniform_rep(args.actor_hidden_dim[agent_id], self.G_SO2eR3)]
        reps = [self.rep_in]+middle_layers

        # Build the network as a sequence of EMLP blocks
        self.network = torch.nn.Sequential(
            *[EMLPBlock(rin,rout) for rin,rout in zip(reps,reps[1:])],
            Linear(reps[-1],self.rep_out)
        ).to(self.device)
        # Output layers log standard deviation of the action distribution
        self.log_std_linear = nn.Linear(args.actor_hidden_dim[agent_id], args.action_dim_n[agent_id])

        # Apply weight initialization
        self.apply(weights_init_)

    def forward(self, state):
        """
        Forward pass computes the mean and log standard deviation of the action distribution.

        Args:
            state: Input observation (ex, eIx, ev, R, eb1, eIb1, eΩ).

        Returns:
            action: The action (f, M) in body-fixed frame.
        """
        x = self.network[0](state)  
        for layer in range(1,self.hidden_num):  # Extract output of each hidden layer
            x = self.network[layer](x) 

        mean = self.network[self.hidden_num](x)  # mean = self.network(state)
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

    def spectral_norm_regularization(self):
        """
        Apply spectral normalization regularization to the network weights.
        """
        return spectral_norm(self.network, self.device)
    

###################################################################################
############## Modular Architecture ###############################################
###################################################################################
class EMLP_MODUL1_Actor_SAC(nn.Module):
    def __init__(self, args, agent_id, hidden_num=2):
        """
        Equivariant MLP-based 1st module's actor network for TD3.

        Args:
            args: Namespace containing hyperparameters and configurations.
            agent_id: Unique identifier for the agent.
            hidden_num : Number of hidden layers.

        Group representation:
            Input: ρ_θ(g)ex, ρ_θ(g)eIx, ρ_θ(g)ev, ρ_θ(g)b3, ρ_θ(g)eω12
            Output: ρ_e(g)f, ρ_θ(g)τ
        """
        super().__init__()
        self.device = args.device
        self.hidden_num = hidden_num

        # Define groups
        self.G_SO2eR3 = SO2eR3().to(self.device)  # ρ_θ(g) for 3D rotations
        self.G_trivialR1 = Trivial(1).to(self.device)  # ρ_e(g) for scalar values

        # Input representation: ρ_θ(g)ex, ρ_θ(g)eIx, ρ_θ(g)ev, ρ_θ(g)b3, ρ_θ(g)eω12
        self.rep_in  = Vector(self.G_SO2eR3)*5
        # Output representation: ρ_e(g)f, ρ_θ(g)τ
        self.rep_out = Scalar(self.G_trivialR1) + Vector(self.G_SO2eR3)
        
        # Define the hidden layers based on the number of hidden layers and size
        middle_layers = self.hidden_num*[uniform_rep(args.actor_hidden_dim[agent_id], self.G_SO2eR3)]
        reps = [self.rep_in]+middle_layers

        # Build the network as a sequence of EMLP blocks
        self.network = torch.nn.Sequential(
            *[EMLPBlock(rin,rout) for rin,rout in zip(reps,reps[1:])],
            Linear(reps[-1],self.rep_out)
        ).to(self.device)
        # Output layers log standard deviation of the action distribution
        self.log_std_linear = nn.Linear(args.actor_hidden_dim[agent_id], args.action_dim_n[agent_id])

        # Apply weight initialization
        self.apply(weights_init_)

    def forward(self, state):
        """
        Forward pass computes the mean and log standard deviation of the action distribution.

        Args:
            state: Input observation (ex, eIx, ev, b3, eω12).

        Returns:
            action: The action (f, τ).
        """
        x = self.network[0](state)  
        for layer in range(1,self.hidden_num):  # Extract output of each hidden layer
            x = self.network[layer](x) 

        mean = self.network[self.hidden_num](x)  # mean = self.network(state)
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
        
    def spectral_norm_regularization(self):
        """
        Apply spectral normalization regularization to the network weights.
        """
        return spectral_norm(self.network, self.device)


class EMLP_MODUL2_Actor_SAC(nn.Module):
    def __init__(self, args, agent_id, hidden_num=2):
        """
        Equivariant MLP-based 2nd module's actor network for TD3.

        Args:
            args: Namespace containing hyperparameters and configurations.
            agent_id: Unique identifier for the agent.
            hidden_num : Number of hidden layers.

        Group representation:
            Input: ρ_r(g)eb1, ρ_r(g)eIb1, ρ_r(g)eΩ3
            Output: ρ_r(g)M3
        """
        super().__init__()
        self.device = args.device
        self.hidden_num = hidden_num

        # Define groups
        self.G_reflection = Mirror(1).to(self.device)  # ρ_r(g), Mirror symmetry group 

        # Input representation: ρ_r(g)eb1, ρ_r(g)eIb1, ρ_r(g)eΩ3
        self.rep_in  = Vector(self.G_reflection)*3
        # Output representation: ρ_r(g)M3
        self.rep_out = Vector(self.G_reflection)
        
        # Define the hidden layers based on the number of hidden layers and size
        middle_layers = self.hidden_num*[uniform_rep(args.actor_hidden_dim[agent_id], self.G_reflection)]
        reps = [self.rep_in]+middle_layers

        # Build the network as a sequence of EMLP blocks
        self.network = torch.nn.Sequential(
            *[EMLPBlock(rin,rout) for rin,rout in zip(reps,reps[1:])],
            Linear(reps[-1],self.rep_out)
        ).to(self.device)
        # Output layers log standard deviation of the action distribution
        self.log_std_linear = nn.Linear(args.actor_hidden_dim[agent_id], args.action_dim_n[agent_id])

        # Apply weight initialization
        self.apply(weights_init_)

    def forward(self, state):
        """
        Forward pass computes the mean and log standard deviation of the action distribution.

        Args:
            state: Input observation (eb1, eIb1, eΩ3).

        Returns:
            action: The action (M3).
        """
        x = self.network[0](state)  
        for layer in range(1,self.hidden_num):  # Extract output of each hidden layer
            x = self.network[layer](x) 

        mean = self.network[self.hidden_num](x)  # mean = self.network(state)
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
        
    def spectral_norm_regularization(self):
        """
        Apply spectral normalization regularization to the network weights.
        """
        return spectral_norm(self.network, self.device)