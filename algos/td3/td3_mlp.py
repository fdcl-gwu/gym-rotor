import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP_Actor_TD3(nn.Module):
    def __init__(self, args, agent_id):
        """
        MLP-based actor network for TD3.
        Maps observations to actions using a feedforward neural network.

        Args:
            args: Namespace containing hyperparameters and configurations.
            agent_id: Unique identifier for the agent.
        """
        super(MLP_Actor_TD3, self).__init__()
        self.fc1 = nn.Linear(args.obs_dim_n[agent_id], args.actor_hidden_dim[agent_id])  # Input layer
        self.fc2 = nn.Linear(args.actor_hidden_dim[agent_id], args.actor_hidden_dim[agent_id])  # Hidden layer
        self.fc3 = nn.Linear(args.actor_hidden_dim[agent_id], args.action_dim_n[agent_id])  # Output layer

    def forward(self, x):
        """
        Forward pass through the actor network.
        Uses ReLU activations for hidden layers and Tanh for action output.

        Args:
            x: Input observation.

        Returns:
            Action output bounded between [-1, 1] using tanh activation.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # Tanh ensures action values are in the range [-1, 1]


class MLP_Critic(nn.Module):
    def __init__(self, args, agent_id):
        """
        MLP-based critic network for TD3.
        Implements `Clipped Double-Q Learning` with two separate Q-value estimations.

        Args:
            args: Namespace containing hyperparameters and configurations.
            agent_id: Unique identifier for the agent.
        """
        super(MLP_Critic, self).__init__()

        # Q1 Network
        self.fc1 = nn.Linear(args.obs_dim_n[agent_id] + args.action_dim_n[agent_id], args.critic_hidden_dim)  # Input layer
        self.fc2 = nn.Linear(args.critic_hidden_dim, args.critic_hidden_dim)  # Hidden layer
        self.fc3 = nn.Linear(args.critic_hidden_dim, 1)  # Output layer (Q-value)

        # Q2 Network (Independent second critic for Double-Q Learning)
        self.fc4 = nn.Linear(args.obs_dim_n[agent_id] + args.action_dim_n[agent_id], args.critic_hidden_dim)
        self.fc5 = nn.Linear(args.critic_hidden_dim, args.critic_hidden_dim)
        self.fc6 = nn.Linear(args.critic_hidden_dim, 1)

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
        sa = torch.cat([state, action], dim=1)

        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        q2 = F.relu(self.fc4(sa))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)

        return q1, q2  # Return both Q-values for Clipped Double-Q Learning

    def Q1(self, state, action):
        """
        Compute only Q1 value, used for policy updates.

        Args:
            state: Input state tensor.
            action: Input action tensor.

        Returns:
            q1: Q-value estimate from the first Q-network.
        """
        sa = torch.cat([state, action], dim=1)

        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        return q1


class MLP_Critic_CTDE(nn.Module):
    def __init__(self, args):
        """
        MLP-based centralized critic network for TD3.
        Uses global state and action information to compute Q-values.

        Args:
            args: Namespace containing hyperparameters and configurations.
        """
        super(MLP_Critic_CTDE, self).__init__()

        # Q1 Network
        self.fc1 = nn.Linear(sum(args.obs_dim_n) + sum(args.action_dim_n), args.critic_hidden_dim)  # input layer
        self.fc2 = nn.Linear(args.critic_hidden_dim, args.critic_hidden_dim)  # Hidden layer
        self.fc3 = nn.Linear(args.critic_hidden_dim, 1)  # Output layer

        # Q2 Network (Independent second critic for Double-Q Learning)
        self.fc4 = nn.Linear(sum(args.obs_dim_n) + sum(args.action_dim_n), args.critic_hidden_dim)
        self.fc5 = nn.Linear(args.critic_hidden_dim, args.critic_hidden_dim)
        self.fc6 = nn.Linear(args.critic_hidden_dim, 1)

    def forward(self, state, action):
        """
        Forward pass through both Q-networks in a centralized manner.
        Takes the global state and actions as input and returns two Q-values.

        Args:
            state: List of individual agent states, concatenated.
            action: List of individual agent actions, concatenated.

        Returns:
            Q-value estimates (q1, q2) from both networks.
        """
        state = torch.cat(state, dim=1)  # Concatenate state tensors from all agents
        action = torch.cat(action, dim=1)  # Concatenate action tensors from all agents
        sa = torch.cat([state, action], dim=1)

        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        q2 = F.relu(self.fc4(sa))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)

        return q1, q2

    def Q1(self, state, action):
        """
        Compute only Q1 value, used for policy updates in a centralized setting.

        Args:
            state: List of individual agent states, concatenated.
            action: List of individual agent actions, concatenated.

        Returns:
            q1: Q-value estimate from the first Q-network.
        """
        state = torch.cat(state, dim=1)  # Concatenate state tensors from all agents
        action = torch.cat(action, dim=1)  # Concatenate action tensors from all agents
        sa = torch.cat([state, action], dim=1)

        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        return q1
