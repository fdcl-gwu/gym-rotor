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

###################################################################################
############## Monolithic Architecture ############################################
###################################################################################
class EMLP_MONO_Actor_PPO(nn.Module):
    def __init__(self, args, agent_id, hidden_num=2, log_std=0):
        """
        Equivariant MLP-based monolithic actor network for PPO.

        Args:
            args: Namespace containing hyperparameters and configurations.
            agent_id: Unique identifier for the agent.
            hidden_num : Number of hidden layers.
            log_std: Initial log standard deviation for the action distribution.

        Group representation:
            Input: ρ_θ(g)ex, ρ_θ(g)eIx, ρ_θ(g)ev, ρ_θ(g)b1, ρ_θ(g)b2, ρ_θ(g)b3, ρ_e(g)eb1, ρ_e(g)eIb1, ρ_e(g)eΩ
                                                  <--------- ρ_θ(g)R ---------> 
            Output: ρ_e(g)f, ρ_e(g)M
        """
        super().__init__()
        self.device = args.device

        # Define groups
        self.G_SO2eR3 = SO2eR3().to(self.device)  # ρ_θ(g) for 3D rotations
        self.G_trivialR1 = Trivial(1).to(self.device)  # ρ_e(g) for scalar values
        self.G_trivialR3 = Trivial(3).to(self.device)  # ρ_e(g) for 3D vector values
        
        # Input representation: ρ_θ(g)ex, ρ_θ(g)eIx, ρ_θ(g)ev, ρ_θ(g)b1, ρ_θ(g)b2, ρ_θ(g)b3, ρ_e(g)eb1, ρ_e(g)eIb1, ρ_e(g)eΩ
        self.rep_in  = Vector(self.G_SO2eR3)*6 + Scalar(self.G_trivialR1)*2 + Vector(self.G_trivialR3)
        # Output representation: ρ_e(g)f, ρ_e(g)M
        self.rep_out = Scalar(self.G_trivialR1) + Vector(self.G_trivialR3)
        
        # Define the hidden layers based on the number of hidden layers and size
        middle_layers = hidden_num*[uniform_rep(args.actor_hidden_dim[agent_id], self.G_SO2eR3)]
        reps = [self.rep_in]+middle_layers

        # Build the network as a sequence of EMLP blocks
        self.network = torch.nn.Sequential(
            *[EMLPBlock(rin,rout) for rin,rout in zip(reps,reps[1:])],
            Linear(reps[-1],self.rep_out)
        ).to(self.device)
        # Output layers for log standard deviation of the action distribution
        self.log_std = nn.Parameter(torch.ones(1, args.action_dim_n[agent_id]) * log_std)

        # Apply weight initialization
        self.network[-1].weight.data.mul_(0.1)
        self.network[-1].bias.data.mul_(0.0)

    def forward(self, state):
        """
        Forward pass through the network.

        Args:
            state: Input observation (ex, eIx, ev, R, eb1, eIb1, eΩ).

        Returns:
            action: The action (f, M) in body-fixed frame.
        """
        mean = torch.tanh(self.network(state))
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

    def spectral_norm_regularization(self):
        """
        Apply spectral normalization regularization to the network weights.
        """
        return spectral_norm(self.network, self.device)
    

class EMLP_MONO_Critic_PPO(nn.Module):
    def __init__(self, args, agent_id, hidden_num=2):
        """
        Equivariant MLP-based monolithic critic network for PPO.

        Args:
            args: Namespace containing hyperparameters and configurations.
            agent_id: Unique identifier for the agent.
            hidden_num : Number of hidden layers.

        Group representation:
            Input: ρ_θ(g)ex, ρ_θ(g)eIx, ρ_θ(g)ev, ρ_θ(g)b1, ρ_θ(g)b2, ρ_θ(g)b3, ρ_e(g)eb1, ρ_e(g)eIb1, ρ_e(g)eΩ
                                                  <--------- ρ_θ(g)R ---------> 
            Output: ρ_e(g)V
        """
        super().__init__()
        self.device = args.device

        # Define groups
        self.G_SO2eR3 = SO2eR3().to(self.device)  # ρ_θ(g) for 3D rotations
        self.G_trivialR1 = Trivial(1).to(self.device)  # ρ_e(g) for scalar values
        self.G_trivialR3 = Trivial(3).to(self.device)  # ρ_e(g) for 3D vector values

        # Input representation: ρ_θ(g)ex, ρ_θ(g)eIx, ρ_θ(g)ev, ρ_θ(g)b1, ρ_θ(g)b2, ρ_θ(g)b3, ρ_e(g)eb1, ρ_e(g)eIb1, ρ_e(g)eΩ
        self.rep_in  = Vector(self.G_SO2eR3)*6 + Scalar(self.G_trivialR1)*2 + Vector(self.G_trivialR3)
        # Output representation: ρ_e(g)V(s)
        self.rep_out = Scalar(self.G_trivialR1) 
        
        # Define the hidden layers for the critic network
        middle_layers = hidden_num*[uniform_rep(args.critic_hidden_dim, self.G_SO2eR3)]
        reps = [self.rep_in]+middle_layers

        # Build the network as a sequence of EMLP blocks
        self.network = torch.nn.Sequential(
            *[EMLPBlock(rin,rout) for rin,rout in zip(reps,reps[1:])],
            Linear(reps[-1],self.rep_out)
        ).to(self.device)


    def forward(self, state):
        """
        Forward pass through the critic network to estimate the state value.
        
        Args:
            state: Input state tensor.

        Returns:
            Estimated value function V(s).
        """
        return self.network(state)
    
    def spectral_norm_regularization(self):
        """
        Apply spectral normalization regularization to the network weights.
        """
        return spectral_norm(self.network, self.device)
    

###################################################################################
############## Modular Architecture ###############################################
###################################################################################
class EMLP_MODUL1_Actor_PPO(nn.Module):
    def __init__(self, args, agent_id, hidden_num=2, log_std=0):
        """
        Equivariant MLP-based 1st module's actor network for PPO.

        Args:
            args: Namespace containing hyperparameters and configurations.
            agent_id: Unique identifier for the agent.
            hidden_num : Number of hidden layers.
            log_std: Initial log standard deviation for the action distribution.

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
        
        # Define the hidden layers for the critic network
        middle_layers = self.hidden_num*[uniform_rep(args.actor_hidden_dim[agent_id], self.G_SO2eR3)]
        reps = [self.rep_in]+middle_layers

        # Build the network as a sequence of EMLP blocks
        self.network = torch.nn.Sequential(
            *[EMLPBlock(rin,rout) for rin,rout in zip(reps,reps[1:])],
            Linear(reps[-1],self.rep_out)
        ).to(self.device)
        # Output layers for log standard deviation of the action distribution
        self.log_std = nn.Parameter(torch.ones(1, args.action_dim_n[agent_id]) * log_std)

        # Apply weight initialization
        self.network[-1].weight.data.mul_(0.1)
        self.network[-1].bias.data.mul_(0.0)

    def forward(self, state):
        """
        Forward pass through the network.

        Args:
            state: Input observation (ex, eIx, ev, b3, eω12).

        Returns:
            action: The action (f, τ).
        """
        mean = torch.tanh(self.network(state))
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
        
    def spectral_norm_regularization(self):
        """
        Apply spectral normalization regularization to the network weights.
        """
        return spectral_norm(self.network, self.device)


class EMLP_MODUL2_Actor_PPO(nn.Module):
    def __init__(self, args, agent_id, hidden_num=2, log_std=0):
        """
        Equivariant MLP-based 2nd module's actor network for PPO.

        Args:
            args: Namespace containing hyperparameters and configurations.
            agent_id: Unique identifier for the agent.
            hidden_num : Number of hidden layers.
            log_std: Initial log standard deviation for the action distribution.

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
        
        # Define the hidden layers for the critic network
        middle_layers = self.hidden_num*[uniform_rep(args.actor_hidden_dim[agent_id], self.G_reflection)]
        reps = [self.rep_in]+middle_layers

        # Build the network as a sequence of EMLP blocks
        self.network = torch.nn.Sequential(
            *[EMLPBlock(rin,rout) for rin,rout in zip(reps,reps[1:])],
            Linear(reps[-1],self.rep_out)
        ).to(self.device)
        # Output layers for log standard deviation of the action distribution
        self.log_std = nn.Parameter(torch.ones(1, args.action_dim_n[agent_id]) * log_std)

        # Apply weight initialization
        self.network[-1].weight.data.mul_(0.1)
        self.network[-1].bias.data.mul_(0.0)

    def forward(self, state):
        """
        Forward pass through the network.

        Args:
            state: Input observation (eb1, eIb1, eΩ3).

        Returns:
            action: The action (M3).
        """
        mean = torch.tanh(self.network(state))
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
        
    def spectral_norm_regularization(self):
        """
        Apply spectral normalization regularization to the network weights.
        """
        return spectral_norm(self.network, self.device)
    

class EMLP_MODUL1_CTDE_Critic_PPO(nn.Module):
    # Equivariant MLP for critic
    #   args:
    #       hidden_num : number of hidden layers
    #       hidden_dim : number of neurons at each hidden layer
    #   input: 𝜌(g)ex, 𝜌(g)eIx, 𝜌(g)ev, 𝜌(g)b3, 𝜌(g)ew12, 𝜌(h)eb1, 𝜌(h)eIb1, 𝜌(h)eW3
    #   output: V

    def __init__(self, args, agent_id, hidden_num=2):
        """
        Equivariant MLP-based 1st module's centralized critic network for TD3.

        Args:
            args: Namespace containing hyperparameters and configurations.
            agent_id: Unique identifier for the agent.
            hidden_num : Number of hidden layers.

        Group representation:
            Input: ρ_θ(g)ex, ρ_θ(g)eIx, ρ_θ(g)ev, ρ_θ(g)b3, ρ_θ(g)eω12, ρ_r(g)eb1, ρ_r(g)eIb1, ρ_r(g)eΩ3
            Output: ρ_e(g)V
        """
        super().__init__()
        self.device = args.device

        # Define groups
        self.G_SO2eR3 = SO2eR3().to(self.device)
        self.G_trivialR1 = Trivial(1).to(self.device)
        self.G_reflection = Mirror(1).to(self.device)

        # Input representation: ρ_θ(g)ex, ρ_θ(g)eIx, ρ_θ(g)ev, ρ_θ(g)b3, ρ_θ(g)eω12, ρ_r(g)eb1, ρ_r(g)eIb1, ρ_r(g)eΩ3
        self.rep_in = Vector(self.G_SO2eR3)*5 + Vector(self.G_reflection)*3 
        # Output representation: ρ_e(g)V(s)
        self.rep_out = Scalar(self.G_trivialR1) 
        
        # Define the hidden layers for the critic network
        middle_layers = hidden_num*[uniform_rep(args.critic_hidden_dim, self.G_SO2eR3)] 
        reps = [self.rep_in]+middle_layers

        # Build the network as a sequence of EMLP blocks
        self.network = torch.nn.Sequential(
            *[EMLPBlock(rin,rout) for rin,rout in zip(reps,reps[1:])],
            Linear(reps[-1],self.rep_out)
        ).to(self.device)

    def forward(self, state):
        """
        Forward pass through the critic network to estimate the state value.
        
        Args:
            state: Input state tensor.

        Returns:
            Estimated value function V(s).
        """
        state = torch.cat(state, dim=1)

        return self.network(state)
    
    def spectral_norm_regularization(self):
        """
        Apply spectral normalization regularization to the network weights.
        """
        return spectral_norm(self.network, self.device)


class EMLP_MODUL2_CTDE_Critic_PPO(nn.Module):
    # Equivariant MLP for critic
    #   args:
    #       hidden_num : number of hidden layers
    #       hidden_dim : number of neurons at each hidden layer
    #   input: 𝜌(g)ex, 𝜌(g)eIx, 𝜌(g)ev, 𝜌(g)b3, 𝜌(g)ew12, 𝜌(h)eb1, 𝜌(h)eIb1, 𝜌(h)eW3
    #   output: V

    def __init__(self, args, agent_id, hidden_num=2):
        """
        Equivariant MLP-based 2nd module's centralized critic network for PPO.

        Args:
            args: Namespace containing hyperparameters and configurations.
            agent_id: Unique identifier for the agent.
            hidden_num : Number of hidden layers.

        Group representation:
            Input: ρ_θ(g)ex, ρ_θ(g)eIx, ρ_θ(g)ev, ρ_θ(g)b3, ρ_θ(g)eω12, ρ_r(g)eb1, ρ_r(g)eIb1, ρ_r(g)eΩ3
            Output: ρ_e(g)V
        """
        super().__init__()
        self.device = args.device

        # Define groups
        self.G_SO2eR3 = SO2eR3().to(self.device)  # ρ_θ(g) for 3D rotations 
        self.G_trivialR1 = Trivial(1).to(self.device)  # ρ_e(g) for scalar values
        self.G_reflection = Mirror(1).to(self.device)  # ρ_r(g), Mirror symmetry group 

        # Input representation: ρ_θ(g)ex, ρ_θ(g)eIx, ρ_θ(g)ev, ρ_θ(g)b3, ρ_θ(g)eω12, ρ_r(g)eb1, ρ_r(g)eIb1, ρ_r(g)eΩ3
        self.rep_in = Vector(self.G_SO2eR3)*5 + Vector(self.G_reflection)*3 
        # Output representation: ρ_e(g)V(s)
        self.rep_out = Scalar(self.G_trivialR1) 
        
        # Define the hidden layers for the critic network
        middle_layers = hidden_num*[uniform_rep(args.critic_hidden_dim, self.G_reflection)] # self.G_SO2eR3
        reps = [self.rep_in]+middle_layers

        # Build the network as a sequence of EMLP blocks
        self.network = torch.nn.Sequential(
            *[EMLPBlock(rin,rout) for rin,rout in zip(reps,reps[1:])],
            Linear(reps[-1],self.rep_out)
        ).to(self.device)

    def forward(self, state):
        """
        Forward pass through the critic network to estimate the state value.
        
        Args:
            state: Input state tensor.

        Returns:
            Estimated value function V(s).
        """
        state = torch.cat(state, dim=1)

        return self.network(state)
    
    def spectral_norm_regularization(self):
        """
        Apply spectral normalization regularization to the network weights.
        """
        return spectral_norm(self.network, self.device)
    

class EMLP_MODUL1_DTDE_Critic_PPO(nn.Module):
    def __init__(self, args, agent_id, hidden_num=2):
        """
        Equivariant MLP-based 1st module's decentralized critic network for PPO.

        Args:
            args: Namespace containing hyperparameters and configurations.
            agent_id: Unique identifier for the agent.
            hidden_num : Number of hidden layers.

        Group representation:
            Input: ρ_θ(g)ex, ρ_θ(g)eIx, ρ_θ(g)ev, ρ_θ(g)b3, ρ_θ(g)eω12
            Output: ρ_e(g)V
        """
        super().__init__()
        self.device = args.device

        # Define groups
        self.G_SO2eR3 = SO2eR3().to(self.device)  # ρ_θ(g) for 3D rotations 
        self.G_trivialR1 = Trivial(1).to(self.device)  # ρ_e(g) for scalar values

        # Input representation: ρ_θ(g)ex, ρ_θ(g)eIx, ρ_θ(g)ev, ρ_θ(g)b3, ρ_θ(g)eω12
        self.rep_in = Vector(self.G_SO2eR3)*5
        # Output representation: ρ_e(g)V(s)
        self.rep_out = Scalar(self.G_trivialR1) 
        
        # Define the hidden layers for the critic network
        middle_layers = hidden_num*[uniform_rep(args.critic_hidden_dim, self.G_SO2eR3)] 
        reps = [self.rep_in]+middle_layers

        # Build the network as a sequence of EMLP blocks
        self.network = torch.nn.Sequential(
            *[EMLPBlock(rin,rout) for rin,rout in zip(reps,reps[1:])],
            Linear(reps[-1],self.rep_out)
        ).to(self.device)

    def forward(self, state):
        """
        Forward pass through the critic network to estimate the state value.
        
        Args:
            state: Input state tensor.

        Returns:
            Estimated value function V(s).
        """
        return self.network(state)
    
    def spectral_norm_regularization(self):
        """
        Apply spectral normalization regularization to the network weights.
        """
        return spectral_norm(self.network, self.device)


class EMLP_MODUL2_DTDE_Critic_PPO(nn.Module):
    def __init__(self, args, agent_id, hidden_num=2):
        """
        Equivariant MLP-based 2nd module's decentralized critic network for PPO.

        Args:
            args: Namespace containing hyperparameters and configurations.
            agent_id: Unique identifier for the agent.
            hidden_num : Number of hidden layers.

        Group representation:
            Input: ρ_r(g)eb1, ρ_r(g)eIb1, ρ_r(g)eΩ3
            Output: ρ_e(g)V
        """
        super().__init__()
        self.device = args.device

        # Define groups
        self.G_trivialR1 = Trivial(1).to(self.device)  # ρ_e(g) for scalar values
        self.G_reflection = Mirror(1).to(self.device)  # ρ_r(g), Mirror symmetry group 

        # Input representation: ρ_r(g)eb1, ρ_r(g)eIb1, ρ_r(g)eΩ3
        self.rep_in = Vector(self.G_reflection)*3
        # Output representation: ρ_e(g)V(s)
        self.rep_out = Scalar(self.G_trivialR1) 
        
        # Define the hidden layers for the critic network
        middle_layers = hidden_num*[uniform_rep(args.critic_hidden_dim, self.G_reflection)]
        reps = [self.rep_in]+middle_layers

        # Build the network as a sequence of EMLP blocks
        self.network = torch.nn.Sequential(
            *[EMLPBlock(rin,rout) for rin,rout in zip(reps,reps[1:])],
            Linear(reps[-1],self.rep_out)
        ).to(self.device)

    def forward(self, state):
        """
        Forward pass through the critic network to estimate the state value.
        
        Args:
            state: Input state tensor.

        Returns:
            Estimated value function V(s).
        """        
        return self.network(state)
    
    def spectral_norm_regularization(self):
        """
        Apply spectral normalization regularization to the network weights.
        """
        return spectral_norm(self.network, self.device)