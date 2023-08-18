import collections
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

buffer_limit  = 50000

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
        
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        s_array = np.array(s_lst, dtype=np.float32)
        a_array = np.array(a_lst, dtype=np.int64)
        r_array = np.array(r_lst, dtype=np.float32)
        s_prime_array = np.array(s_prime_lst, dtype=np.float32)
        done_mask_array = np.array(done_mask_lst, dtype=np.float32)

        return (
            torch.tensor(s_array),
            torch.tensor(a_array),
            torch.tensor(r_array),
            torch.tensor(s_prime_array),
            torch.tensor(done_mask_array)
        )

    
    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2)
        self.fc1 = nn.Linear(64,4)

    def forward(self, x, mode='train'):
        if mode == 'sample':
            x = x.unsqueeze(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        return x
      
    def sample_action(self, obs, epsilon):
        out = self.forward(obs, mode='sample')
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,1)
        else : 
            return out.argmax().item()