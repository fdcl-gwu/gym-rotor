import torch
import torch.nn as nn
import torch.nn.functional as F

# Different agents have different observation dimensions and action dimensions, so we need to use 'agent_id' to distinguish them
class Actor(nn.Module):
    def __init__(self, args, agent_id):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(args.obs_dim_n[agent_id], args.actor_hidden_dim[agent_id])
        self.fc2 = nn.Linear(args.actor_hidden_dim[agent_id], args.actor_hidden_dim[agent_id])
        self.fc3 = nn.Linear(args.actor_hidden_dim[agent_id], args.action_dim_n[agent_id])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic_MADDPG(nn.Module):
    def __init__(self, args):
        super(Critic_MADDPG, self).__init__()
        self.fc1 = nn.Linear(sum(args.obs_dim_n) + sum(args.action_dim_n), args.critic_hidden_dim)
        self.fc2 = nn.Linear(args.critic_hidden_dim, args.critic_hidden_dim)
        self.fc3 = nn.Linear(args.critic_hidden_dim, 1)

    def forward(self, state, action):
        state = torch.cat(state, dim=1)
        action = torch.cat(action, dim=1)

        q = F.relu(self.fc1(torch.cat([state, action], dim=1)))
        q = F.relu(self.fc2(q))
        q = self.fc3(q)
        return q


class Critic_MATD3(nn.Module):
    def __init__(self, args):
        super(Critic_MATD3, self).__init__()
        """
        Clipped Double-Q Learning:
        """
        # Q1 architecture:
        self.fc1 = nn.Linear(sum(args.obs_dim_n) + sum(args.action_dim_n), args.critic_hidden_dim)
        self.fc2 = nn.Linear(args.critic_hidden_dim, args.critic_hidden_dim)
        self.fc3 = nn.Linear(args.critic_hidden_dim, 1)

        # Q2 architecture:
        self.fc4 = nn.Linear(sum(args.obs_dim_n) + sum(args.action_dim_n), args.critic_hidden_dim)
        self.fc5 = nn.Linear(args.critic_hidden_dim, args.critic_hidden_dim)
        self.fc6 = nn.Linear(args.critic_hidden_dim, 1)

    def forward(self, state, action):
        state = torch.cat(state, dim=1)
        action = torch.cat(action, dim=1)

        q1 = F.relu(self.fc1(torch.cat([state, action], dim=1)))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        q2 = F.relu(self.fc4(torch.cat([state, action], dim=1)))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        return q1, q2

    def Q1(self, state, action):
        state = torch.cat(state, dim=1)
        action = torch.cat(action, dim=1)
        
        q1 = F.relu(self.fc1(torch.cat([state, action], dim=1)))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        return q1


class Critic_TD3(nn.Module):
    def __init__(self, args, agent_id):
        super(Critic_TD3, self).__init__()
        """
        Clipped Double-Q Learning:
        """
        # Q1 architecture:
        self.fc1 = nn.Linear(args.obs_dim_n[agent_id] + args.action_dim_n[agent_id], args.critic_hidden_dim)
        self.fc2 = nn.Linear(args.critic_hidden_dim, args.critic_hidden_dim)
        self.fc3 = nn.Linear(args.critic_hidden_dim, 1)

        # Q2 architecture:
        self.fc4 = nn.Linear(args.obs_dim_n[agent_id] + args.action_dim_n[agent_id], args.critic_hidden_dim)
        self.fc5 = nn.Linear(args.critic_hidden_dim, args.critic_hidden_dim)
        self.fc6 = nn.Linear(args.critic_hidden_dim, 1)

    def forward(self, state, action):
        q1 = F.relu(self.fc1(torch.cat([state, action], dim=1)))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        q2 = F.relu(self.fc4(torch.cat([state, action], dim=1)))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        return q1, q2

    def Q1(self, state, action):
        q1 = F.relu(self.fc1(torch.cat([state, action], dim=1)))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        return q1
