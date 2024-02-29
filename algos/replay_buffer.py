import torch
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA found.")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS found.")
else:
    device = torch.device("cpu")


class ReplayBuffer(object):
    def __init__(self, args):
        self.N = args.N  # The number of agents
        self.replay_buffer_size = args.replay_buffer_size
        self.batch_size = args.batch_size
        self.count = 0
        self.current_size = 0
        self.buffer_obs_n, self.buffer_a_n, self.buffer_r_n, self.buffer_s_next_n, self.buffer_done_n = [], [], [], [], []
        for agent_id in range(self.N):
            self.buffer_obs_n.append(np.empty((self.replay_buffer_size, args.obs_dim_n[agent_id])))
            self.buffer_a_n.append(np.empty((self.replay_buffer_size, args.action_dim_n[agent_id])))
            self.buffer_r_n.append(np.empty((self.replay_buffer_size, 1)))
            self.buffer_s_next_n.append(np.empty((self.replay_buffer_size, args.obs_dim_n[agent_id])))
            self.buffer_done_n.append(np.empty((self.replay_buffer_size, 1)))

    def store_transition(self, obs_n, act_n, r_n, obs_next_n, done_n):
        for agent_id in range(self.N):
            self.buffer_obs_n[agent_id][self.count] = obs_n[agent_id]
            self.buffer_a_n[agent_id][self.count] = act_n[agent_id]
            self.buffer_r_n[agent_id][self.count] = r_n[agent_id]
            self.buffer_s_next_n[agent_id][self.count] = obs_next_n[agent_id]
            self.buffer_done_n[agent_id][self.count] = done_n[agent_id]
        self.count = (self.count + 1) % self.replay_buffer_size  # When the 'count' reaches max_size, it will be reset to 0.
        self.current_size = min(self.current_size + 1, self.replay_buffer_size)

    def sample(self, ):
        index = np.random.choice(self.current_size, size=self.batch_size, replace=False)
        batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = [], [], [], [], []
        for agent_id in range(self.N):
            batch_obs_n.append(torch.tensor(self.buffer_obs_n[agent_id][index], dtype=torch.float, device=device))
            batch_a_n.append(torch.tensor(self.buffer_a_n[agent_id][index], dtype=torch.float, device=device))
            batch_r_n.append(torch.tensor(self.buffer_r_n[agent_id][index], dtype=torch.float, device=device))
            batch_obs_next_n.append(torch.tensor(self.buffer_s_next_n[agent_id][index], dtype=torch.float, device=device))
            batch_done_n.append(torch.tensor(self.buffer_done_n[agent_id][index], dtype=torch.float, device=device))

        return batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n
