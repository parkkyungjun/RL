# dqn_utils.py
import torch
import torch.nn as nn
from collections import namedtuple, deque
import random
import torch.nn.functional as F

# Define a named tuple for storing experiences
Transition = namedtuple('Transition', ['state', 'action', 'next_state', 'reward'])

# Replay Memory Class to store and sample experiences
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Randomly samples batch_size transitions from memory."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Returns the current size of internal memory."""
        return len(self.memory)

# DQN Model
# class DQN(nn.Module):
#     def __init__(self, n_observations, n_actions):
#         super(DQN, self).__init__()
#         self.layer1 = nn.Linear(n_observations, 256)
#         self.bn1 = nn.BatchNorm1d(256)
#         self.layer2 = nn.Linear(256, 512)
#         self.bn2 = nn.BatchNorm1d(512)
#         self.layer3 = nn.Linear(512, 1024)
#         self.bn3 = nn.BatchNorm1d(1024)
#         self.layer4 = nn.Linear(1024, 512)
#         self.bn4 = nn.BatchNorm1d(512)
#         self.layer5 = nn.Linear(512, n_actions)

#     def forward(self, x):
#         """Forward pass through the network with batch normalization."""
#         print(x.shape)
#         x = F.leaky_relu(self.bn1(self.layer1(x)))
#         x = F.leaky_relu(self.bn2(self.layer2(x)))
#         x = F.leaky_relu(self.bn3(self.layer3(x)))
#         x = F.leaky_relu(self.bn4(self.layer4(x)))
#         return self.layer5(x)
    # def __init__(self, n_observations, n_actions):
    #     super(DQN, self).__init__()
    #     self.layer1 = nn.Linear(n_observations, 256)
    #     self.layer2 = nn.Linear(256, 512)
    #     self.layer3 = nn.Linear(512, 1024)
    #     self.layer4 = nn.Linear(1024, 512)
    #     self.layer5 = nn.Linear(512, n_actions)
    #     # self.dropout = nn.Dropout(p=0.3)

    # def forward(self, x):
    #     """Forward pass through the network."""
    #     x = F.leaky_relu(self.layer1(x))
    #     x = F.leaky_relu(self.layer2(x))
    #     x = F.leaky_relu(self.layer3(x))
    #     x = F.leaky_relu(self.layer4(x))
    #     return self.layer5(x)

    
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, n_actions)
        
    def forward(self, x):
        x = F.leaky_relu(self.layer1(x))
        x = F.leaky_relu(self.layer2(x))
        x = self.layer3(x)
        return x