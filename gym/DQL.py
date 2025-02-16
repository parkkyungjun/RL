import os
import yaml
import gym
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import imageio

# -------------------------
# 1) Q-Network 정의
# -------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        # 여기서는 간단히 2개의 히든 레이어를 사용
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# -------------------------
# 2) 리플레이 버퍼 정의
# -------------------------
class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.max_size:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.max_size

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def main():
    # ---------------------------------------------------
    # 0) YAML 설정 불러오기
    # ---------------------------------------------------
    with open("DQL.yaml", "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)

    env_id = config["env_id"]
    discrete_actions = config["discrete_actions"] # np.linspace(-2, 2, num=20) # config["discrete_actions"]
    episodes = config["episodes"]
    max_steps_per_episode = config["max_steps_per_episode"]
    batch_size = config["batch_size"]
    gamma = config["gamma"]
    epsilon_start = config["epsilon_start"]
    epsilon_end = config["epsilon_end"]
    epsilon_decay = config["epsilon_decay"]
    learning_rate = config["learning_rate"]
    target_update_frequency = config["target_update_frequency"]
    replay_buffer_size = config["replay_buffer_size"]
    record_steps = config["record_steps"]

    if not os.path.exists(env_id):
        os.makedirs(env_id)

    env = gym.make(env_id)

    state_dim = env.observation_space.shape[0]
    action_dim = len(discrete_actions)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    q_network = QNetwork(state_dim, action_dim).to(device)
    target_network = QNetwork(state_dim, action_dim).to(device)
    target_network.load_state_dict(q_network.state_dict())  

    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

    replay_buffer = ReplayBuffer(replay_buffer_size)

    epsilon = epsilon_start

    def select_action(state):
        """Epsilon-greedy로 이산 행동 선택"""
        if np.random.rand() < epsilon:
            return np.random.randint(action_dim)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = q_network(state_tensor)
            return q_values.argmax(dim=1).item()

    global_step = 0
    for episode in range(episodes):
        state, info = env.reset()
        state = np.array(state, dtype=np.float32)

        episode_reward = 0
        for t in range(max_steps_per_episode):
            action_idx = select_action(state)
            action_value = discrete_actions[action_idx]  
            next_state, reward, done, truncated, info = env.step([action_value])  
            next_state = np.array(next_state, dtype=np.float32)

            replay_buffer.push(state, action_idx, reward, next_state, done)

            state = next_state
            episode_reward += reward
            global_step += 1

            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).to(device)

                q_values = q_network(states)             
                q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_q_values = target_network(next_states)
                    max_next_q_values = next_q_values.max(1)[0]
                    target = rewards + (1 - dones) * gamma * max_next_q_values
                # import code; code.interact(local=locals())

                loss = torch.nn.SmoothL1Loss()(q_values, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if global_step % target_update_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

            if done:
                break

        if epsilon > epsilon_end:
            epsilon *= epsilon_decay
        print(f"Episode {episode}/{episodes}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.3f}")
        if episode_reward > -5:
            break

    env.close()

    test_env = gym.make(env_id, render_mode='rgb_array')
    frames = []

    for demo_ep in range(1):
        obs, info = test_env.reset()
        obs = np.array(obs, dtype=np.float32)

        for step in range(record_steps):
            state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action_idx = q_network(state_tensor).argmax().item()
            action_value = discrete_actions[action_idx]

            next_obs, reward, done, truncated, info = test_env.step([action_value])
            next_obs = np.array(next_obs, dtype=np.float32)

            frame = test_env.render()  
            frames.append(frame)
            obs = next_obs
            if done:
                obs, _ = test_env.reset()
                obs = np.array(obs, dtype=np.float32)

            for _ in range(2 + 10 // 2):
                frames.append(frame)

    test_env.close()

    os.makedirs(env_id, exist_ok=True)
    imageio.mimsave(os.path.join(env_id, env_id + '.mp4'), frames, fps=300)

    imageio.mimsave(os.path.join(env_id, env_id + '.gif'), frames, fps=300, loop=0)
    print(f"[{env_id}] 학습 완료! {1} 에피소드 시연 영상을 '{env_id + 'mp4'}'로 저장했습니다.")



if __name__ == "__main__":
    main()
