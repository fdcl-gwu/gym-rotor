"""
Re-tuned version of Deep Deterministic Policy Gradients (DDPG)
Paper: https://arxiv.org/abs/1509.02971
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, actor_hidden_dim, \
                       min_act, max_act, avrg_act, scale_act):
        super(Actor, self).__init__()

        # Fully-Connected (FC) layers
        self.fc1 = nn.Linear(state_dim,  actor_hidden_dim)
        self.fc2 = nn.Linear(actor_hidden_dim, actor_hidden_dim)
        self.fc3 = nn.Linear(actor_hidden_dim, action_dim)
        
        self.min_act = min_act
        self.max_act = max_act
        self.avrg_act = avrg_act
        self.scale_act = scale_act


    def forward(self, state):
        action = F.relu(self.fc1(state))
        action = F.relu(self.fc2(action))
        '''
        # Linear scale, [-1, 1] -> [min_act, max_act] 
        return self.scale_act * torch.tanh(self.fc3(action)) + self.avrg_act
        '''
        return torch.tanh(self.fc3(action))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, critic_hidden_dim):
		super(Critic, self).__init__()

		# FC layers
		self.fc1 = nn.Linear(state_dim + action_dim, critic_hidden_dim)
		self.fc2 = nn.Linear(critic_hidden_dim, critic_hidden_dim)
		self.fc3 = nn.Linear(critic_hidden_dim, 1)


	def forward(self, state, action):
		Q = F.relu(self.fc1(torch.cat([state, action], 1)))
		Q = F.relu(self.fc2(Q))
		return self.fc3(Q)


class DDPG(object):
    def __init__(self, state_dim, action_dim, actor_hidden_dim, critic_hidden_dim, \
                       min_act, max_act, avrg_act, scale_act, \
                       discount, lr, tau):
        
        self.actor = Actor(state_dim, action_dim, actor_hidden_dim, min_act, max_act, avrg_act, scale_act).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr)

        self.critic = Critic(state_dim, action_dim, critic_hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr)

        self.min_act = min_act
        self.max_act = max_act
        self.avrg_act = avrg_act
        self.scale_act = scale_act
        self.discount = discount
        self.tau = tau


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state)
        return action.cpu().data.numpy().flatten()


    def train(self, replay_buffer, batch_size):
        # Randomly sample a batch of transitions from an experience replay buffer:
        state, action, next_state, reward, done = replay_buffer.sample(batch_size)

        """
        Q-Learning side of DDPG with critic networks:
        """
        # Get current Q-values, Q(s, a):
        current_Q = self.critic(state, action)	
        
        # Compute target actions from a target policy network:
        next_action = (
            self.actor_target(next_state)
            ).clamp(self.min_act, self.max_act)

        # Get target Q-values, Q_targ(s', a'): 
        target_Q = self.critic_target(next_state, next_action)

        # Compute targets, y(r, s', d):
        target_Q = reward + (self.discount * (1 - done) * target_Q).detach()

        # Set a mean-squared Bellman error (MSBE) loss function:
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Update Q-functions by gradient descent:
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


        """
        Policy learning side of DDPG with actor networks:
        """
        # Set actor loss s.t. Q(s,\mu(s)) approximates \max_a Q(s,a):
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        # Update policy by gradient ascent:
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target targets:
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
		