import torch
import torch.nn as nn
import torch.nn.functional as F
from algos.emlp_torch.groups import *
from algos.emlp_torch.reps import *
from algos.emlp_torch.nn import EMLPBlock, Linear, uniform_rep
from algos.emlp_torch.groups import SO2eR3, Trivial
from algos.emlp_torch.reps import Vector, Scalar
from algos.spectral_norm_regularization import spectral_norm

###################################################################################
############## Monolithic Architecture ############################################
###################################################################################
class EMLP_MONO_Actor_TD3(nn.Module):
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

    def forward(self, state):
        """
        Forward pass through the network.

        Args:
            state: Input observation (ex, eIx, ev, R, eb1, eIb1, eΩ).

        Returns:
            action: The action (f, M) in body-fixed frame.
        """
        return torch.tanh(self.network(state))
    
    def spectral_norm_regularization(self):
        """
        Apply spectral normalization regularization to the network weights.
        """
        return spectral_norm(self.network, self.device)
    

class EMLP_MONO_Critic(nn.Module):
    def __init__(self, args, agent_id, hidden_num=2):
        """
        Equivariant MLP-based monolithic critic network for TD3.

        Args:
            args: Namespace containing hyperparameters and configurations.
            agent_id: Unique identifier for the agent.
            hidden_num : Number of hidden layers.

        Group representation:
            Input: ρ_θ(g)ex, ρ_θ(g)eIx, ρ_θ(g)ev, ρ_θ(g)b1, ρ_θ(g)b2, ρ_θ(g)b3, ρ_e(g)eb1, ρ_e(g)eIb1, ρ_e(g)eΩ, ρ_e(g)f, ρ_e(g)M
                                                  <--------- ρ_θ(g)R ---------> 
            Output: ρ_e(g)Q1, ρ_e(g)Q2
        """
        super().__init__()
        self.device = args.device

        # Define groups
        self.G_SO2eR3 = SO2eR3().to(self.device)  # ρ_θ(g) for 3D rotations
        self.G_trivialR1 = Trivial(1).to(self.device)  # ρ_e(g) for scalar values
        self.G_trivialR3 = Trivial(3).to(self.device)  # ρ_e(g) for 3D vector values

        # Input representation: ρ_θ(g)ex, ρ_θ(g)eIx, ρ_θ(g)ev, ρ_θ(g)b1, ρ_θ(g)b2, ρ_θ(g)b3, ρ_e(g)eb1, ρ_e(g)eIb1, ρ_e(g)eΩ, ρ_e(g)f, ρ_e(g)M
        self.rep_in  = Vector(self.G_SO2eR3)*6 + Scalar(self.G_trivialR1)*2 + Vector(self.G_trivialR3) \
            + Scalar(self.G_trivialR1) + Vector(self.G_trivialR3)
        # Output representation: ρ_e(g)Q(s,a)
        self.rep_out = Scalar(self.G_trivialR1) 
        
        # Define the hidden layers for the critic network
        middle_layers = hidden_num*[uniform_rep(args.critic_hidden_dim, self.G_SO2eR3)]
        reps = [self.rep_in]+middle_layers

        # Build two separate critic networks for TD3 (double Q-learning)
        self.network1 = torch.nn.Sequential(
            *[EMLPBlock(rin,rout) for rin,rout in zip(reps,reps[1:])],
            Linear(reps[-1],self.rep_out)
        ).to(self.device)
        self.network2 = torch.nn.Sequential(
            *[EMLPBlock(rin,rout) for rin,rout in zip(reps,reps[1:])],
            Linear(reps[-1],self.rep_out)
        ).to(self.device)

    def forward(self, state, action):
        """
        Forward pass through both Q-networks.
        Takes the state and action as input and returns two Q-values (q1, q2).

        Args:
            state: Input state tensor.
            action: Input action tensor.

        Returns:
            Q-value estimates (q1, q2) from both networks.
        """
        input = torch.cat([state, action], 1)  # Concatenate state and action as input to the critic networks
        return self.network1(input), self.network2(input)
    
    def spectral_norm_regularization(self):
        """
        Apply spectral normalization regularization to the network weights.
        """
        return spectral_norm(self.network1, self.device) + spectral_norm(self.network2, self.device)
    

###################################################################################
############## Modular Architecture ###############################################
###################################################################################
class EMLP_MODUL1_Actor_TD3(nn.Module):
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

        # Define groups
        self.G_SO2eR3 = SO2eR3().to(self.device)  # ρ_θ(g) for 3D rotations
        self.G_trivialR1 = Trivial(1).to(self.device)  # ρ_e(g) for scalar values

        # Input representation: ρ_θ(g)ex, ρ_θ(g)eIx, ρ_θ(g)ev, ρ_θ(g)b3, ρ_θ(g)eω12
        self.rep_in  = Vector(self.G_SO2eR3)*5
        # Output representation: ρ_e(g)f, ρ_θ(g)τ
        self.rep_out = Scalar(self.G_trivialR1) + Vector(self.G_SO2eR3)
        
        # Define the hidden layers based on the number of hidden layers and size
        middle_layers = hidden_num*[uniform_rep(args.actor_hidden_dim[agent_id], self.G_SO2eR3)]
        reps = [self.rep_in]+middle_layers
        
        # Build the network as a sequence of EMLP blocks
        self.network = torch.nn.Sequential(
            *[EMLPBlock(rin,rout) for rin,rout in zip(reps,reps[1:])],
            Linear(reps[-1],self.rep_out)
        ).to(self.device)

    def forward(self, state):
        """
        Forward pass through the network.

        Args:
            state: Input observation (ex, eIx, ev, b3, eω12).

        Returns:
            action: The action (f, τ).
        """
        return torch.tanh(self.network(state))
    
    def spectral_norm_regularization(self):
        """
        Apply spectral normalization regularization to the network weights.
        """
        return spectral_norm(self.network, self.device)


class EMLP_MODUL2_Actor_TD3(nn.Module):
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

        # Define groups
        self.G_reflection = Mirror(1).to(self.device)  # ρ_r(g), Mirror symmetry group 

        # Input representation: ρ_r(g)eb1, ρ_r(g)eIb1, ρ_r(g)eΩ3
        self.rep_in  = Vector(self.G_reflection)*3
        # Output representation: ρ_r(g)M3
        self.rep_out = Vector(self.G_reflection)
        
        # Define the hidden layers based on the number of hidden layers and size
        middle_layers = hidden_num*[uniform_rep(args.actor_hidden_dim[agent_id], self.G_reflection)]
        reps = [self.rep_in]+middle_layers

        # Build the network as a sequence of EMLP blocks
        self.network = torch.nn.Sequential(
            *[EMLPBlock(rin,rout) for rin,rout in zip(reps,reps[1:])],
            Linear(reps[-1],self.rep_out)
        ).to(self.device)

    def forward(self, state):
        """
        Forward pass through the network.

        Args:
            state: Input observation (eb1, eIb1, eΩ3).

        Returns:
            action: The action (M3).
        """
        return torch.tanh(self.network(state))
    
    def spectral_norm_regularization(self):
        """
        Apply spectral normalization regularization to the network weights.
        """
        return spectral_norm(self.network, self.device)


class EMLP_MODUL1_CTDE_Critic(nn.Module):
    def __init__(self, args, agent_id, hidden_num=2):
        """
        Equivariant MLP-based 1st module's centralized critic network for TD3.

        Args:
            args: Namespace containing hyperparameters and configurations.
            agent_id: Unique identifier for the agent.
            hidden_num : Number of hidden layers.

        Group representation:
            Input: ρ_θ(g)ex, ρ_θ(g)eIx, ρ_θ(g)ev, ρ_θ(g)b3, ρ_θ(g)eω12, ρ_r(g)eb1, ρ_r(g)eIb1, ρ_r(g)eΩ3, ρ_e(g)f, ρ_θ(g)τ, ρ_r(g)M3
            Output: ρ_e(g)Q1, ρ_e(g)Q2
        """
        super().__init__()
        self.device = args.device

        # Define groups
        self.G_SO2eR3 = SO2eR3().to(self.device)  # ρ_θ(g) for 3D rotations
        self.G_trivialR1 = Trivial(1).to(self.device)  # ρ_e(g) for scalar values
        self.G_reflection = Mirror(1).to(self.device)  # ρ_r(g), Mirror symmetry group

        # Input representation: ρ_θ(g)ex, ρ_θ(g)eIx, ρ_θ(g)ev, ρ_θ(g)b3, ρ_θ(g)eω12, ρ_r(g)eb1, ρ_r(g)eIb1, ρ_r(g)eΩ3, \
        #                       ρ_e(g)f, ρ_θ(g)τ, ρ_r(g)M3
        self.rep_in = Vector(self.G_SO2eR3)*5 + Vector(self.G_reflection)*3 + \
            Scalar(self.G_trivialR1) + Vector(self.G_SO2eR3) + Vector(self.G_reflection)
        # Output representation: ρ_e(g)Q(s,a)
        self.rep_out = Scalar(self.G_trivialR1) 
        
        # Define the hidden layers for the critic network
        middle_layers = hidden_num*[uniform_rep(args.critic_hidden_dim, self.G_SO2eR3)] 
        reps = [self.rep_in]+middle_layers

        # Build two separate critic networks for TD3 (double Q-learning)
        self.network1 = torch.nn.Sequential(
            *[EMLPBlock(rin,rout) for rin,rout in zip(reps,reps[1:])],
            Linear(reps[-1],self.rep_out)
        ).to(self.device)
        self.network2 = torch.nn.Sequential(
            *[EMLPBlock(rin,rout) for rin,rout in zip(reps,reps[1:])],
            Linear(reps[-1],self.rep_out)
        ).to(self.device)

    def forward(self, state, action):
        """
        Forward pass through both Q-networks.
        Takes the state and action as input and returns two Q-values (q1, q2).

        Args:
            state: Input state tensor.
            action: Input action tensor.

        Returns:
            Q-value estimates (q1, q2) from both networks.
        """
        state = torch.cat(state, dim=1)
        action = torch.cat(action, dim=1)

        input = torch.cat([state, action], 1)
        return self.network1(input), self.network2(input)
    
    def spectral_norm_regularization(self):
        """
        Apply spectral normalization regularization to the network weights.
        """
        return spectral_norm(self.network1, self.device) + spectral_norm(self.network2, self.device)


class EMLP_MODUL2_CTDE_Critic(nn.Module):
    def __init__(self, args, agent_id, hidden_num=2):
        """
        Equivariant MLP-based 2nd module's centralized critic network for TD3.

        Args:
            args: Namespace containing hyperparameters and configurations.
            agent_id: Unique identifier for the agent.
            hidden_num : Number of hidden layers.

        Group representation:
            Input: ρ_θ(g)ex, ρ_θ(g)eIx, ρ_θ(g)ev, ρ_θ(g)b3, ρ_θ(g)eω12, ρ_r(g)eb1, ρ_r(g)eIb1, ρ_r(g)eΩ3, ρ_e(g)f, ρ_θ(g)τ, ρ_r(g)M3
            Output: ρ_e(g)Q1, ρ_e(g)Q2
        """
        super().__init__()
        self.device = args.device

        # Define groups
        self.G_SO2eR3 = SO2eR3().to(self.device)  # ρ_θ(g) for 3D rotations 
        self.G_trivialR1 = Trivial(1).to(self.device)  # ρ_e(g) for scalar values
        self.G_reflection = Mirror(1).to(self.device)  # ρ_r(g), Mirror symmetry group 

        # Input representation: ρ_θ(g)ex, ρ_θ(g)eIx, ρ_θ(g)ev, ρ_θ(g)b3, ρ_θ(g)eω12, ρ_r(g)eb1, ρ_r(g)eIb1, ρ_r(g)eΩ3, \
        #                       ρ_e(g)f, ρ_θ(g)τ, ρ_r(g)M3
        self.rep_in = Vector(self.G_SO2eR3)*5 + Vector(self.G_reflection)*3 + \
            Scalar(self.G_trivialR1) + Vector(self.G_SO2eR3) + Vector(self.G_reflection)
        # Output representation: ρ_e(g)Q(s,a)
        self.rep_out = Scalar(self.G_trivialR1) 
        
        # Define the hidden layers for the critic network
        middle_layers = hidden_num*[uniform_rep(args.critic_hidden_dim, self.G_reflection)]
        reps = [self.rep_in]+middle_layers

        # Build two separate critic networks for TD3 (double Q-learning)
        self.network1 = torch.nn.Sequential(
            *[EMLPBlock(rin,rout) for rin,rout in zip(reps,reps[1:])],
            Linear(reps[-1],self.rep_out)
        ).to(self.device)
        self.network2 = torch.nn.Sequential(
            *[EMLPBlock(rin,rout) for rin,rout in zip(reps,reps[1:])],
            Linear(reps[-1],self.rep_out)
        ).to(self.device)

    def forward(self, state, action):
        """
        Forward pass through both Q-networks.
        Takes the state and action as input and returns two Q-values (q1, q2).

        Args:
            state: Input state tensor.
            action: Input action tensor.

        Returns:
            Q-value estimates (q1, q2) from both networks.
        """
        state = torch.cat(state, dim=1)
        action = torch.cat(action, dim=1)

        input = torch.cat([state, action], 1)
        return self.network1(input), self.network2(input)
    
    def spectral_norm_regularization(self):
        """
        Apply spectral normalization regularization to the network weights.
        """
        return spectral_norm(self.network1, self.device) + spectral_norm(self.network2, self.device)
    

class EMLP_MODUL1_DTDE_Critic(nn.Module): 
    def __init__(self, args, agent_id, hidden_num=2):
        """
        Equivariant MLP-based 1st module's decentralized critic network for TD3.

        Args:
            args: Namespace containing hyperparameters and configurations.
            agent_id: Unique identifier for the agent.
            hidden_num : Number of hidden layers.

        Group representation:
            Input: ρ_θ(g)ex, ρ_θ(g)eIx, ρ_θ(g)ev, ρ_θ(g)b3, ρ_θ(g)eω12, ρ_e(g)f, ρ_θ(g)τ
            Output: ρ_e(g)Q1, ρ_e(g)Q2
        """
        super().__init__()
        self.device = args.device

        # Define groups
        self.G_SO2eR3 = SO2eR3().to(self.device)  # ρ_θ(g) for 3D rotations 
        self.G_trivialR1 = Trivial(1).to(self.device)  # ρ_e(g) for scalar values

        # Input representation: ρ_θ(g)ex, ρ_θ(g)eIx, ρ_θ(g)ev, ρ_θ(g)b3, ρ_θ(g)eω12, ρ_e(g)f, ρ_θ(g)τ
        self.rep_in = Vector(self.G_SO2eR3)*5 + Scalar(self.G_trivialR1) + Vector(self.G_SO2eR3)
        # Output representation: ρ_e(g)Q(s,a)
        self.rep_out = Scalar(self.G_trivialR1) 
        
        # Define the hidden layers for the critic network
        middle_layers = hidden_num*[uniform_rep(args.critic_hidden_dim, self.G_SO2eR3)]
        reps = [self.rep_in]+middle_layers

        # Build two separate critic networks for TD3 (double Q-learning)
        self.network1 = torch.nn.Sequential(
            *[EMLPBlock(rin,rout) for rin,rout in zip(reps,reps[1:])],
            Linear(reps[-1],self.rep_out)
        ).to(self.device)
        self.network2 = torch.nn.Sequential(
            *[EMLPBlock(rin,rout) for rin,rout in zip(reps,reps[1:])],
            Linear(reps[-1],self.rep_out)
        ).to(self.device)

    def forward(self, state, action):
        """
        Forward pass through both Q-networks.
        Takes the state and action as input and returns two Q-values (q1, q2).

        Args:
            state: Input state tensor.
            action: Input action tensor.

        Returns:
            Q-value estimates (q1, q2) from both networks.
        """
        input = torch.cat([state, action], 1)
        return self.network1(input), self.network2(input)
    
    def spectral_norm_regularization(self):
        """
        Apply spectral normalization regularization to the network weights.
        """
        return spectral_norm(self.network1, self.device) + spectral_norm(self.network2, self.device)
    

class EMLP_MODUL2_DTDE_Critic(nn.Module): 
    def __init__(self, args, agent_id, hidden_num=2):
        """
        Equivariant MLP-based 2nd module's decentralized critic network for TD3.

        Args:
            args: Namespace containing hyperparameters and configurations.
            agent_id: Unique identifier for the agent.
            hidden_num : Number of hidden layers.

        Group representation:
            Input: ρ_r(g)eb1, ρ_r(g)eIb1, ρ_r(g)eΩ3, ρ_r(g)M3
            Output: ρ_e(g)Q1, ρ_e(g)Q2
        """
        super().__init__()
        self.device = args.device

        # Define groups
        self.G_trivialR1 = Trivial(1).to(self.device)  # ρ_e(g) for scalar values
        self.G_reflection = Mirror(1).to(self.device)  # ρ_r(g), Mirror symmetry group 

        # Input representation: ρ_r(g)eb1, ρ_r(g)eIb1, ρ_r(g)eΩ3, ρ_r(g)M3
        self.rep_in  = Vector(self.G_reflection)*4
        # Output representation: ρ_e(g)Q(s,a)
        self.rep_out = Scalar(self.G_trivialR1) 
        
        # Define the hidden layers for the critic network
        middle_layers = hidden_num*[uniform_rep(args.critic_hidden_dim, self.G_reflection)]
        reps = [self.rep_in]+middle_layers

        # Build two separate critic networks for TD3 (double Q-learning)
        self.network1 = torch.nn.Sequential(
            *[EMLPBlock(rin,rout) for rin,rout in zip(reps,reps[1:])],
            Linear(reps[-1],self.rep_out)
        ).to(self.device)
        self.network2 = torch.nn.Sequential(
            *[EMLPBlock(rin,rout) for rin,rout in zip(reps,reps[1:])],
            Linear(reps[-1],self.rep_out)
        ).to(self.device)

    def forward(self, state, action):
        """
        Forward pass through both Q-networks.
        Takes the state and action as input and returns two Q-values (q1, q2).

        Args:
            state: Input state tensor.
            action: Input action tensor.

        Returns:
            Q-value estimates (q1, q2) from both networks.
        """
        input = torch.cat([state, action], 1)
        return self.network1(input), self.network2(input)
    
    def spectral_norm_regularization(self):
        """
        Apply spectral normalization regularization to the network weights.
        """
        return spectral_norm(self.network1, self.device) + spectral_norm(self.network2, self.device)
    