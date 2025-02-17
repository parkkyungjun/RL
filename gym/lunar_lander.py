import os
import gym
import numpy as np
import random
import collections
import torch
import torch.nn as nn
import torch.optim as optim
import imageio

LEARNING_RATE = 1e-3
GAMMA = 0.99
BATCH_SIZE = 64
BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 10000
EPS_START = 1.0
EPS_END = 0.02
EPS_DECAY = 1e-5
TARGET_UPDATE_FREQ = 1000

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )

    def size(self):
        return len(self.buffer)

def main():
    device = torch.device("cuda")

    env_id = "LunarLander-v2"

    env = gym.make(env_id)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_network = QNetwork(state_dim, action_dim).to(device)
    target_q_network = QNetwork(state_dim, action_dim).to(device)

    target_q_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=LEARNING_RATE)

    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    eps = EPS_START
    total_steps = 0

    obs, info = env.reset()
    for _ in range(MIN_REPLAY_SIZE):
        action = env.action_space.sample()
        next_obs, reward, done, truncated, info = env.step(action)
        replay_buffer.add((obs, action, reward, next_obs, done))
        obs = next_obs

        if done or truncated:
            obs, info = env.reset()

    num_episodes = 1000
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0

        while not (done or truncated):
            total_steps += 1

            if random.random() < eps:
                action = env.action_space.sample()
            else:
                state_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                q_values = q_network(state_t).to('cpu')
                action = torch.argmax(q_values, dim=1).item()

            eps = max(EPS_END, eps - EPS_DECAY)
            next_obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            replay_buffer.add((obs, action, reward, next_obs, done))

            obs = next_obs
            states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
            states_t = torch.FloatTensor(states)
            actions_t = torch.LongTensor(actions)
            rewards_t = torch.FloatTensor(rewards)
            next_states_t = torch.FloatTensor(next_states)
            dones_t = torch.FloatTensor(dones)

            q_values = q_network(states_t.to(device)).to('cpu')
            q_a = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                max_q_next = target_q_network(next_states_t.to(device)).max(dim=1)[0].to('cpu')
                target = rewards_t + GAMMA * max_q_next * (1 - dones_t)

            loss = nn.MSELoss()(q_a, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if total_steps % TARGET_UPDATE_FREQ == 0:
                target_q_network.load_state_dict(q_network.state_dict())

        print(f"Episode {episode} | Reward: {episode_reward:.2f} | Eps: {eps:.3f}")

    env.close()

    test_env = gym.make(env_id, render_mode='rgb_array')
    discrete_actions = [0, 1, 2, 3]
    frames = []
    record_steps = 500

    obs, info = test_env.reset()
    obs = np.array(obs, dtype=np.float32)

    for _ in range(record_steps):
        state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

        with torch.no_grad():
            action_idx = q_network(state_tensor).argmax().item().to('cpu')

        action_value = discrete_actions[action_idx]
        next_obs, reward, done, truncated, info = test_env.step(action_value)
        next_obs = np.array(next_obs, dtype=np.float32)
        frame = test_env.render()
        frames.append(frame)
        obs = next_obs

        if done or truncated:
            obs, info = test_env.reset()
            obs = np.array(obs, dtype=np.float32)

        for _ in range(2 + 10 // 2):
            frames.append(frame)

    test_env.close()

    os.makedirs(env_id, exist_ok=True)

    imageio.mimsave(os.path.join(env_id, env_id + '.mp4'), frames, fps=300)
    imageio.mimsave(os.path.join(env_id, env_id + '.gif'), frames, fps=300, loop=0)

    print(f"[{env_id}] 학습 완료! 1 에피소드 시연 영상을 '{env_id + '.mp4'}'로 저장했습니다.")

if __name__ == "__main__":
    main()
