import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

#Hyperparameters
learning_rate = 0.00002
gamma         = 0.98

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []
        
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 3)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x
      
    def put_data(self, item):
        self.data.append(item)
        
    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + gamma * R
            loss = -torch.log(prob) * R
            loss.backward()
        self.optimizer.step()
        self.data = []

def getE(x, v):
  h = np.sin(3 * x) *.45 + .55
  return (h*0.0025 + (v**2)/2 + v*0.0025*0.01)

def main():
    env = gym.make('MountainCar-v0')
    pi = Policy()
    score = 0.0
    print_interval = 20
    
    for n_epi in range(300):
        state = env.reset()
        done = False
        
        s = (state - env.observation_space.low)*np.array([10, 100])
        s = np.round(s, 0).astype(int)

        while not done: # MountainCar-v1 forced to terminates at 300 step.
            if n_epi >= (300 - 5):
                env.render()

            prob = pi(torch.from_numpy(s).float())
            m = Categorical(prob)
            a = m.sample()
            state2, r, done, truncated = env.step(a.item())
            
            # x1, v1 = state[0], state[1]
            x2, v2 = state2[0], state2[1]
            E1, E2 = getE(0.5, 0), getE(x2, v2)

            if x2 >= 0.5 :
              r = 1
            else:
              r = (E2 - E1)*1000
            # print(E1, E2, r)
            # -1.2 0.6 -0.07 0.07
            
            state = state2
            s_prime = (state2 - env.observation_space.low)*np.array([10, 100])
            s_prime = np.round(s_prime, 0).astype(int)

            pi.put_data((r,prob[a]))
            s = s_prime
            score += -1
            
        pi.train_net()
        
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {}".format(n_epi, score/print_interval))
            score = 0.0
    env.close()
    
if __name__ == '__main__':
    main()