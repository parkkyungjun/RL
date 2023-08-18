import gymnasium as gym  # 0.25.2
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

from utils import ReplayBuffer, Qnet

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

def train(q, q_target, memory, optimizer):
    for i in range(10):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1,a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
batch_size    = 32
episodes = 1

mapping = {
    b'F': 0,
    b'H': 1,
    b'S': 2,
    b'G': 3
}


def main():
    env = gym.make('FrozenLake-v1', desc=generate_random_map(size=4), render_mode="rgb_array", is_slippery=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q = Qnet().to(device)

    q_target = Qnet()

    memory = ReplayBuffer()

    print_interval = 1000
    score = 0.0  
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    
    checkpoint = torch.load('./checkpoint_50000.pth')
    q.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    q_target.load_state_dict(q.state_dict())
    q_target.to(device)
        
    # for n_epi in range(50000):
    #     epsilon = max(0.01, 0.1 - 0.01*(n_epi/5000)) #Linear annealing from 8% to 1%
    #     if n_epi > 50000:
    #         env = gym.make('FrozenLake-v1', desc=generate_random_map(size=4), render_mode="rgb_array", is_slippery=False)
    #     else:
    #         env = gym.make('FrozenLake-v1', desc=generate_random_map(size=4), render_mode="rgb_array")

    #     grid_map = np.vectorize(mapping.get)(np.array([env.unwrapped.desc]))
    #     grid_map[0][0][0] = 0
    #     s, _ = env.reset()
    #     done = False

    #     while not done:
    #         s_map = grid_map.copy()
    #         row, col = s // len(s_map[0][0]), s % len(s_map[0][0])
    #         s_map[0][row][col] = 2
    #         # print(s_map)
    #         a = q.sample_action(torch.from_numpy(s_map).float().to(device), epsilon)      
    #         s_prime, r, done, truncated, info = env.step(a)
            
            
    #         if done:
    #             if s_prime != 15:
    #                 r = 0
    #             else:
    #                 r = 1
                
    #         done_mask = 0.0 if done else 1.0
    #         s_prime_map = grid_map.copy()
    #         row, col = s_prime // len(s_map[0][0]), s_prime % len(s_map[0][0])
    #         if s_prime_map[0][row][col] != 1:
    #             s_prime_map[0][row][col] = 2
                
    #         memory.put((s_map,a,r,s_prime_map, done_mask))
    #         s = s_prime
            

    #         score += r
            
    #     if memory.size()>2000:
    #         train(q, q_target, memory, optimizer)

    #     if n_epi%print_interval==0 and n_epi!=0:
    #         q_target.load_state_dict(q.state_dict())
    #         print("n_episode :{}, score : {:.4f}, n_buffer : {}, eps : {:.1f}%".format(
    #                                                         n_epi, score/print_interval, memory.size(), epsilon*100))
    #         score = 0.0
    # env.close()
    # torch.save({
    #     'model_state_dict': q.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    # }, 'checkpoint_300000.pth')
    q.eval()
    for n_epi in range(5):
        env_test = gym.make('FrozenLake-v1', desc=generate_random_map(size=4), render_mode="human")
        s, _ = env_test.reset()

        grid_map = np.vectorize(mapping.get)(np.array([env.unwrapped.desc]))
        grid_map[0][0][0] = 0
        done = False
        while not done:
            s_map = grid_map.copy()
            row, col = s // len(s_map[0][0]), s % len(s_map[0][0])
            s_map[0][row][col] = 2
            action = q.sample_action(torch.from_numpy(s_map).float(), 0.01)
            s_prime, reward, done, _, _ = env_test.step(action)
            s = s_prime
            env_test.render()

    env_test.close()
    

if __name__ == '__main__':
    main()