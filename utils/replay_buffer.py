import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, replay_buffer_size):
		self.state = np.zeros((replay_buffer_size, state_dim))
		self.action = np.zeros((replay_buffer_size, action_dim))
		self.next_state = np.zeros((replay_buffer_size, state_dim))
		self.reward = np.zeros((replay_buffer_size, 1))
		self.done = np.zeros((replay_buffer_size, 1))

		self.replay_buffer_size = replay_buffer_size
		self.num  = 0
		self.size = 0


	# 
	def add(self, state, action, next_state, reward, done):
		self.state[self.num] = state
		self.action[self.num] = action
		self.next_state[self.num] = next_state
		self.reward[self.num] = reward
		self.done[self.num] = done

		self.num  = (self.num + 1) % self.replay_buffer_size
		self.size = min(self.size + 1, self.replay_buffer_size)


	# 
	def sample(self, batch_size):
		index = np.random.randint(0, self.size, size=batch_size)

		return(torch.FloatTensor(self.state[index]).to(device),
			   torch.FloatTensor(self.action[index]).to(device),
			   torch.FloatTensor(self.next_state[index]).to(device),
			   torch.FloatTensor(self.reward[index]).to(device),
			   torch.FloatTensor(self.done[index]).to(device))