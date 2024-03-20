import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from algos.networks.mlp import Actor, Critic_TD3
from algos.networks.emlp import Equiv_Actor_SARL, Equiv_Critic_SARL

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA found.")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS found.")
else:
    device = torch.device("cpu")


class TD3(object):
    def __init__(self, args, agent_id):
        self.framework = args.framework
        self.N = args.N
        self.agent_id = agent_id
        self.max_action = args.max_action
        self.action_dim = args.action_dim_n[agent_id]
        self.lr_a = args.lr_a[agent_id]
        self.lr_c = args.lr_c[agent_id]
        self.discount = args.discount
        self.tau = args.tau
        self.use_clip_grad_norm = args.use_clip_grad_norm
        self.grad_max_norm = args.grad_max_norm
        self.target_noise = args.target_noise
        self.noise_clip = args.noise_clip
        self.policy_update_freq = args.policy_update_freq
        self.lam_T, self.lam_S, self.lam_M = args.lam_T, args.lam_S, args.lam_M
        self.total_it = 0

        # Train models with equivariant reinforcement learning:
        if args.use_equiv:
            self.actor = Equiv_Actor_SARL(args, agent_id).to(device)
            self.actor_target = copy.deepcopy(self.actor)
            self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=self.lr_a)
            self.actor_scheduler = CosineAnnealingWarmRestarts(self.actor_optimizer, T_0=3500000, eta_min=1e-6)

            self.critic = Equiv_Critic_SARL(args, agent_id).to(device)
            self.critic_target = copy.deepcopy(self.critic)
            self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=self.lr_c)
            self.critic_scheduler = CosineAnnealingWarmRestarts(self.critic_optimizer, T_0=3500000, eta_min=1e-6)
        else:
            self.actor = Actor(args, agent_id).to(device)
            self.actor_target = copy.deepcopy(self.actor)
            self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=self.lr_a)
            self.actor_scheduler = CosineAnnealingWarmRestarts(self.actor_optimizer, T_0=3500000, eta_min=1e-6)

            self.critic = Critic_TD3(args, agent_id).to(device)
            self.critic_target = copy.deepcopy(self.critic)
            self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=self.lr_c)
            self.critic_scheduler = CosineAnnealingWarmRestarts(self.critic_optimizer, T_0=3500000, eta_min=1e-6)

    # Each agent selects actions based on its own local observations(add noise for exploration)
    def choose_action(self, obs, explor_noise_std):
        obs = torch.unsqueeze(torch.tensor(obs, dtype=torch.float), 0).to(device)
        act = self.actor(obs).cpu().data.numpy().flatten()
        return (act + np.random.normal(0, explor_noise_std, size=self.action_dim)).clip(-self.max_action, self.max_action)


    def train(self, replay_buffer, agent_n, env):
        self.total_it += 1

        # Randomly sample a batch of transitions from an experience replay buffer:
        batch_obs_n, batch_act_n, batch_rwd_n, batch_obs_next_n, batch_done_n = replay_buffer.sample()

        batch_obs = batch_obs_n[self.agent_id]
        batch_act = batch_act_n[self.agent_id]
        batch_rwd = batch_rwd_n[self.agent_id]
        batch_obs_next = batch_obs_next_n[self.agent_id]
        batch_done = batch_done_n[self.agent_id]

        """
        Q-Learning side of TD3 with critic networks:
        """
        with torch.no_grad():  # target_Q has no gradient
            batch_act_next = agent_n[self.agent_id].actor_target(batch_obs_next)
            # Add clipped noise to target actions for 'target policy smoothing':
            noise = (torch.randn_like(batch_act_next) * self.target_noise).clamp(-self.noise_clip, self.noise_clip)
            # Compute target actions from a target policy network:
            batch_act_next = (batch_act_next + noise).clamp(-self.max_action, self.max_action)

            # Get target Q-values, Q_targ(s', a'): 
            target_Q1, target_Q2 = self.critic_target(batch_obs_next, batch_act_next)

            # Use a smaller target Q-value:
            target_Q = torch.min(target_Q1, target_Q2)

            # Compute targets, y(r, s', d):
            target_Q = batch_rwd + self.discount * (1 - batch_done) * target_Q  # shape:(batch_size,1)

        # Get current Q-values, Q1(s, a) and Q2(s, a):
        current_Q1, current_Q2 = self.critic(batch_obs, batch_act)   # shape:(batch_size,1)

        # Set a mean-squared Bellman error (MSBE) loss function:
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Update Q-functions by gradient descent:
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.use_clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_max_norm)
        self.critic_optimizer.step()
        self.critic_scheduler.step()

        """
        Policy learning side of TD3 with actor networks:
        """
        # Update policy less frequently than Q-function for 'delayed policy updates':
        if self.total_it % self.policy_update_freq == 0:

            # Set actor loss s.t. Q(s,\mu(s)) approximates \max_a Q(s,a):
            batch_act = (self.actor(batch_obs)).clamp(-self.max_action, self.max_action)
            actor_loss = -self.critic.Q1(batch_obs, batch_act).mean()  # Only use Q1
            
            # Regularizing action policies for smooth control
            lam_T = self.lam_T # Temporal Smoothness
            batch_act_next = self.actor(batch_obs_next).clamp(-self.max_action, self.max_action)
            Loss_T = F.mse_loss(batch_act, batch_act_next)
            actor_loss += lam_T * Loss_T

            lam_S = self.lam_S # Spatial Smoothness
            noise_S = (
                torch.normal(mean=0., std=0.05, size=(1, self.action_dim))
                ).clamp(-self.noise_clip, self.noise_clip).to(device) # mean and standard deviation
            action_bar = (batch_act + noise_S).clamp(-self.max_action, self.max_action)
            Loss_S = F.mse_loss(batch_act, action_bar)
            actor_loss += lam_S * Loss_S

            lam_M = self.lam_M # Magnitude Smoothness
            batch_size = batch_act.shape[0]
            if self.framework == "SARL":
                f_total_hover = np.interp(4.*env.hover_force, 
                                        [4.*env.min_force, 4.*env.max_force], 
                                        [-self.max_action, self.max_action]
                                ) * torch.ones(batch_size, 1) # normalized into [-1, 1]
                M_hover = torch.zeros(batch_size, 3)
                nominal_action = torch.cat([f_total_hover, M_hover], 1).to(device)
            elif self.framework == "DTDE":
                if self.agent_id == 0:
                    f_total_hover = np.interp(4.*env.hover_force, 
                                            [4.*env.min_force, 4.*env.max_force], 
                                            [-self.max_action, self.max_action]
                                    ) * torch.ones(batch_size, 1) # normalized into [-1, 1]
                    tau_hover = torch.zeros(batch_size, 3)
                    nominal_action = torch.cat([f_total_hover, tau_hover], 1).to(device)
                elif self.agent_id == 1:
                    nominal_action = torch.zeros(batch_size, 1).to(device) # M3_hover
            Loss_M = F.mse_loss(batch_act, nominal_action)
            actor_loss += lam_M * Loss_M

            # Update policy by gradient ascent:
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.use_clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_max_norm)
            self.actor_optimizer.step()
            self.actor_scheduler.step()

            # Softly update the target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def save_model(self, framework, total_steps, agent_id, seed):
        torch.save(self.actor.state_dict(), "./models/{}_{}k_steps_agent_{}_{}.pth".format(framework, total_steps/1000, agent_id, seed))


    def save_solved_model(self, framework, total_steps, agent_id, seed):
        torch.save(self.actor.state_dict(), "./models/{}_{}k_steps_agent_{}_solved_{}.pth".format(framework, total_steps/1000, agent_id, seed))


    def load(self, framework, total_steps, agent_id, seed):
        if device == "gpu":
            self.actor.load_state_dict(torch.load("./models/{}_{}k_steps_agent_{}_{}.pth".format(framework, total_steps/1000, agent_id, seed)))
        else:
            self.actor.load_state_dict(torch.load("./models/{}_{}k_steps_agent_{}_{}.pth".format(framework, total_steps/1000, agent_id, seed), map_location=torch.device('cpu')))


    def load_solved_model(self, framework, total_steps, agent_id, seed):
        if device == "gpu":
            self.actor.load_state_dict(torch.load("./models/{}_{}k_steps_agent_{}_solved_{}.pth".format(framework, total_steps/1000, agent_id, seed)))
        else:
            self.actor.load_state_dict(torch.load("./models/{}_{}k_steps_agent_{}_solved_{}.pth".format(framework, total_steps/1000, agent_id, seed), map_location=torch.device('cpu')))