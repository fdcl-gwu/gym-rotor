import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CyclicLR, OneCycleLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, actor_hidden_dim):
        super(Actor, self).__init__()

        # Fully-Connected (FC) layers
        self.fc1 = nn.Linear(state_dim,  actor_hidden_dim)
        self.fc2 = nn.Linear(actor_hidden_dim, actor_hidden_dim)
        self.fc3 = nn.Linear(actor_hidden_dim, action_dim)        


    def forward(self, state):
        action = F.relu(self.fc1(state))
        action = F.relu(self.fc2(action))
        return torch.tanh(self.fc3(action)) # [-1, 1]


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, critic_hidden_dim):
        super(Critic, self).__init__()
        """
        Clipped Double-Q Learning:
        """
        # Q1 architecture
        self.fc1 = nn.Linear(state_dim + action_dim, critic_hidden_dim)
        self.fc2 = nn.Linear(critic_hidden_dim, critic_hidden_dim)
        self.fc3 = nn.Linear(critic_hidden_dim, 1)

        # Q2 architecture
        self.fc4 = nn.Linear(state_dim + action_dim, critic_hidden_dim)
        self.fc5 = nn.Linear(critic_hidden_dim, critic_hidden_dim)
        self.fc6 = nn.Linear(critic_hidden_dim, 1)

    def forward(self, state, action):
        q1 = F.relu(self.fc1(torch.cat([state, action], 1)))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        q2 = F.relu(self.fc4(torch.cat([state, action], 1)))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)

        return q1, q2


    def Q1(self, state, action):
        q1 = F.relu(self.fc1(torch.cat([state, action], 1)))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)        

        return q1


class TD3_CAPS(object):
    def __init__(self, state_dim, action_dim, actor_hidden_dim, critic_hidden_dim, \
                       max_act, min_act, discount, lr, tau, env, 
                       target_noise, noise_clip, policy_update_freq):

        self.actor = Actor(state_dim, action_dim, actor_hidden_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr)
        self.actor_scheduler = CyclicLR(self.actor_optimizer, base_lr = 1e-7,  max_lr = 7e-4, \
            cycle_momentum = False, mode='triangular') # {triangular, triangular2, exp_range}
        '''
        self.actor_scheduler = OneCycleLR(self.actor_optimizer, max_lr = 3e-4, total_steps = int(3e6), \
            anneal_strategy='cos') # {'cos', 'linear'}
        '''

        self.critic = Critic(state_dim, action_dim, critic_hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr)
        self.critic_scheduler = CyclicLR(self.critic_optimizer, base_lr = 1e-7,  max_lr = 7e-4, \
            cycle_momentum = False, mode='triangular') # {triangular, triangular2, exp_range}
        '''
        self.critic_scheduler = OneCycleLR(self.critic_optimizer, max_lr = 3e-4, total_steps = int(3e6), \
            anneal_strategy='cos') # {'cos', 'linear'}
        '''

        self.min_act = min_act
        self.max_act = max_act
        self.discount = discount
        self.tau = tau
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.policy_update_freq = policy_update_freq

        self.hover_force = env.hover_force # [N]
        self.min_force = env.min_force # [N]
        self.max_force = env.max_force # [N]
        self.action_dim = action_dim 

        self.total_it = 0


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state)
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size):
        self.total_it += 1

        # Randomly sample a batch of transitions from an experience replay buffer:
        state, action, next_state, reward, done = replay_buffer.sample(batch_size)

        """
        Q-Learning side of TD3 with critic networks:
        """
        # Get current Q-values, Q1(s, a) and Q2(s, a):
        current_Q1, current_Q2 = self.critic(state, action)

        with torch.no_grad():
            # Add clipped noise to target actions for 'target policy smoothing':
            noise = (
                torch.randn_like(action) * self.target_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            
            # Compute target actions from a target policy network:
            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(self.min_act, self.max_act) 

            # Get target Q-values, Q_targ(s', a'): 
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)

            # Use a smaller target Q-value:
            target_Q = torch.min(target_Q1, target_Q2)

            # Compute targets, y(r, s', d):
            target_Q = reward + self.discount * (1 - done) * target_Q

        # Set a mean-squared Bellman error (MSBE) loss function:
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Update Q-functions by gradient descent:
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic_scheduler.step()

        """
        Policy learning side of TD3 with actor networks:
        """
        # Update policy less frequently than Q-function for 'delayed policy updates':
        if self.total_it % self.policy_update_freq == 0:

            # Set actor loss s.t. Q(s,\mu(s)) approximates \max_a Q(s,a):
            clipped_action = (self.actor(state)).clamp(self.min_act, self.max_act) 
            actor_loss = -self.critic.Q1(state, clipped_action).mean()
            
            # Regularizing action policies for smooth control
            lam_T = 0.3 # 0.5 - 0.8
            if lam_T > 0: # Temporal Smoothness
                next_action_T = (self.actor(next_state)).clamp(self.min_act, self.max_act) 
                Loss_T = F.mse_loss(clipped_action, next_action_T)
                actor_loss += lam_T * Loss_T

            lam_S = 0.3 # 0.3 - 0.5
            if lam_S > 0: # Spatial Smoothness
                noise_S = (
                    torch.normal(mean=0., std=0.05, size=(1, self.action_dim))
                    ).clamp(-self.noise_clip, self.noise_clip).to(device) # mean and standard deviation
                action_bar = (clipped_action + noise_S).clamp(self.min_act, self.max_act)
                Loss_S = F.mse_loss(clipped_action, action_bar)
                actor_loss += lam_S * Loss_S

            lam_M = 0.4 # 0.2 - 0.5
            if lam_M > 0: # Magnitude Smoothness
                hover_action = np.interp(
                    self.hover_force, [self.min_force, self.max_force], [self.min_act, self.max_act]
                    ) * torch.ones(batch_size, self.action_dim).to(device) # normalized into [-1, 1]
                Loss_M = F.mse_loss(clipped_action, hover_action)
                actor_loss += lam_M * Loss_M

            # Update policy by gradient ascent:
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.actor_scheduler.step()

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
		