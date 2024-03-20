import torch
import torch.nn as nn
import torch.nn.functional as F
from algos.networks.emlp_utils import get_equiv_state, get_equiv_state_batch

class Equiv_Actor_SARL(nn.Module):
    def __init__(self, args, agent_id):
        super(Equiv_Actor_SARL, self).__init__()
        self.fc1 = nn.Linear(args.obs_dim_n[agent_id]-1, args.actor_hidden_dim[agent_id])
        self.fc2 = nn.Linear(args.actor_hidden_dim[agent_id], args.actor_hidden_dim[agent_id])
        self.fc3 = nn.Linear(args.actor_hidden_dim[agent_id], args.action_dim_n[agent_id])

    def forward(self, state):
        #--- Get (de-rotated) equivalent states:
        if state.shape[0] == 1:
            state_equiv = get_equiv_state(state) 
        else:
            state_equiv = get_equiv_state_batch(state)
        action_equiv = F.relu(self.fc1(state_equiv))
        action_equiv = F.relu(self.fc2(action_equiv))
        action_equiv = torch.tanh(self.fc3(action_equiv)) # [-1, 1]

        return action_equiv


class Equiv_Critic_SARL(nn.Module):
    def __init__(self, args, agent_id):
        super(Equiv_Critic_SARL, self).__init__()
        """
        Clipped Double-Q Learning:
        """
        # Q1 architecture:
        self.fc1 = nn.Linear(args.obs_dim_n[agent_id]-1+ args.action_dim_n[agent_id], args.critic_hidden_dim)
        self.fc2 = nn.Linear(args.critic_hidden_dim, args.critic_hidden_dim)
        self.fc3 = nn.Linear(args.critic_hidden_dim, 1)

        # Q2 architecture:
        self.fc4 = nn.Linear(args.obs_dim_n[agent_id]-1 + args.action_dim_n[agent_id], args.critic_hidden_dim)
        self.fc5 = nn.Linear(args.critic_hidden_dim, args.critic_hidden_dim)
        self.fc6 = nn.Linear(args.critic_hidden_dim, 1)

    def forward(self, state, action_equiv):
        #--- Get (de-rotated) equivalent states:
        state_equiv = get_equiv_state_batch(state)

        q1 = F.relu(self.fc1(torch.cat([state_equiv, action_equiv], dim=1)))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        q2 = F.relu(self.fc4(torch.cat([state_equiv, action_equiv], dim=1)))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        return q1, q2

    def Q1(self, state, action_equiv):
        #--- Get (de-rotated) equivalent states:
        state_equiv = get_equiv_state_batch(state)

        q1 = F.relu(self.fc1(torch.cat([state_equiv, action_equiv], dim=1)))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        return q1