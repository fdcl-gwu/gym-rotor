import torch
import numpy as np

class ReplayBuffer(object):
    def __init__(self, args):
        self.rl_algo = args.rl_algo  # selected RL algorithm
        self.N = args.N  # the number of agents
        self.device = args.device
        self.count, self.current_size = 0, 0
        self.buffer_obs_n, self.buffer_act_n, self.buffer_rwd_n, self.buffer_obs_next_n, self.buffer_done_n = [], [], [], [], []
        if self.rl_algo in ["TD3", "SAC"]:
            self.replay_buffer_size = args.replay_buffer_size
            self.batch_size = args.batch_size
        elif self.rl_algo == "PPO":
            self.replay_buffer_size = args.T_horizon
            self.batch_size = args.T_horizon
            self.buffer_logprob_n = []

        # Initialize the buffer:
        for agent_id in range(self.N):
            self.buffer_obs_n.append(np.empty((self.replay_buffer_size, args.obs_dim_n[agent_id])))
            self.buffer_act_n.append(np.empty((self.replay_buffer_size, args.action_dim_n[agent_id])))
            self.buffer_rwd_n.append(np.empty((self.replay_buffer_size, 1)))
            self.buffer_obs_next_n.append(np.empty((self.replay_buffer_size, args.obs_dim_n[agent_id])))
            self.buffer_done_n.append(np.empty((self.replay_buffer_size, 1)))
            if self.rl_algo == "PPO":
                self.buffer_logprob_n.append(np.empty((self.replay_buffer_size, args.action_dim_n[agent_id])))            

    def store_transition(self, obs_n, act_n, rwd_n, obs_next_n, done_n, logprob_n=None):
        for agent_id in range(self.N):
            self.buffer_obs_n[agent_id][self.count] = obs_n[agent_id]
            self.buffer_act_n[agent_id][self.count] = act_n[agent_id]
            self.buffer_rwd_n[agent_id][self.count] = rwd_n[agent_id]
            self.buffer_obs_next_n[agent_id][self.count] = obs_next_n[agent_id]
            self.buffer_done_n[agent_id][self.count] = done_n[agent_id]
            if self.rl_algo == "PPO":
                self.buffer_logprob_n[agent_id][self.count] = logprob_n[agent_id]
        self.count = (self.count + 1) % self.replay_buffer_size  # When the 'count' reaches max_size, it will be reset to 0.
        self.current_size = min(self.current_size + 1, self.replay_buffer_size)

    def sample(self):
        if self.rl_algo in {"TD3", "SAC"}:
            index = np.random.choice(self.current_size, size=self.batch_size, replace=False)
        elif self.rl_algo == "PPO":
            index = np.arange(self.batch_size)
        
        batch_obs_n = [torch.tensor(self.buffer_obs_n[agent_id][index], dtype=torch.float, device=self.device) for agent_id in range(self.N)]
        batch_act_n = [torch.tensor(self.buffer_act_n[agent_id][index], dtype=torch.float, device=self.device) for agent_id in range(self.N)]
        batch_rwd_n = [torch.tensor(self.buffer_rwd_n[agent_id][index], dtype=torch.float, device=self.device) for agent_id in range(self.N)]
        batch_obs_next_n = [torch.tensor(self.buffer_obs_next_n[agent_id][index], dtype=torch.float, device=self.device) for agent_id in range(self.N)]
        batch_done_n = [torch.tensor(self.buffer_done_n[agent_id][index], dtype=torch.float, device=self.device) for agent_id in range(self.N)]
        if self.rl_algo == "PPO":
            batch_logprob_n = [torch.tensor(self.buffer_logprob_n[agent_id], dtype=torch.float, device=self.device) for agent_id in range(self.N)]
            return batch_obs_n, batch_act_n, batch_rwd_n, batch_obs_next_n, batch_done_n, batch_logprob_n

        return batch_obs_n, batch_act_n, batch_rwd_n, batch_obs_next_n, batch_done_n

