import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import imageio

# ------------------------------------
# 1. 하이퍼파라미터 및 설정
# ------------------------------------
env_id = "CliffWalking-v0"
num_episodes = 500          # 학습 에피소드 수
max_steps_per_episode = 200 # 각 에피소드 최대 스텝
gamma = 0.99                # 감가율
epsilon_start = 1.0         # 초기 epsilon
epsilon_end = 0.01          # 최소 epsilon
epsilon_decay = 0.995       # 에피소드마다 epsilon 줄이는 비율
batch_size = 32
buffer_size = 10000
lr = 1e-3                   # 학습률
target_update_interval = 50 # 타겟 네트워크 동기화 주기

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

env = gym.make(env_id)

n_states = env.observation_space.n  # CliffWalking의 상태 개수 = 4*12 = 48
n_actions = env.action_space.n      # CliffWalking의 액션 개수 = 4


# ------------------------------------
# 2. Replay Buffer (경험 재플레이)
# ------------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return (np.array(states),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_states),
                np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


# ------------------------------------
# 3. 신경망 정의
# ------------------------------------
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        # 여기서는 상태(state)가 "정수 인덱스" 형태(0~47)이므로,
        # one-hot으로 변환해서 입력해 줄 수도 있습니다.
        # (혹은 Embedding을 사용할 수도 있음)
        # 간단하게 one-hot 변환 후 MLP로 진행해 보겠습니다.
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        # x: (batch, ) 형태의 state 인덱스가 들어왔다면 one-hot으로 변환
        #    (batch_size, n_states) 형태로 만들어야 함
        #    예를 들어 batch 개의 상태가 있다고 하면, 각 상태를 one-hot으로 만들기
        one_hot = torch.nn.functional.one_hot(x, num_classes=n_states).float()
        
        x = self.fc1(one_hot)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x


# ------------------------------------
# 4. 에이전트(DQN) 초기화
# ------------------------------------
policy_net = DQN(n_states, n_actions).to(device)
target_net = DQN(n_states, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=lr)
replay_buffer = ReplayBuffer(buffer_size)


def select_action(state, epsilon):
    """epsilon-greedy로 액션 선택"""
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        # Q값이 가장 큰 액션 선택
        state_t = torch.LongTensor([state]).to(device)
        with torch.no_grad():
            q_values = policy_net(state_t)
        return q_values.argmax().item()


def update_model():
    """Replay Buffer에서 배치를 뽑아 DQN 업데이트"""
    if len(replay_buffer) < batch_size:
        return

    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    states_t = torch.LongTensor(states).to(device)
    actions_t = torch.LongTensor(actions).to(device).unsqueeze(1)  # shape: (batch, 1)
    rewards_t = torch.FloatTensor(rewards).to(device).unsqueeze(1) # shape: (batch, 1)
    next_states_t = torch.LongTensor(next_states).to(device)
    dones_t = torch.FloatTensor(dones).to(device).unsqueeze(1)

    # Q(s,a) = policy_net(states)[range(batch), actions]
    q_values = policy_net(states_t).gather(1, actions_t)  # shape: (batch, 1)

    # Q'(s',a') = target_net(next_states).max(1)
    with torch.no_grad():
        max_next_q_values = target_net(next_states_t).max(1)[0].unsqueeze(1)  # shape: (batch, 1)
        target = rewards_t + gamma * max_next_q_values * (1 - dones_t)

    loss = nn.MSELoss()(q_values, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# ------------------------------------
# 5. 학습 루프
# ------------------------------------
epsilon = epsilon_start
episode_rewards = []

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    for step in range(max_steps_per_episode):
        action = select_action(state, epsilon)
        next_state, reward, done, truncated, info = env.step(action)

        # CliffWalking은 -1 보상과 낭떠러지(-100) 등이 존재
        # 보상이 크거나 작을 수 있으니 그대로 저장
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state

        total_reward += reward

        update_model()

        if done or truncated:
            break

    # epsilon 점차 감소
    epsilon = max(epsilon * epsilon_decay, epsilon_end)

    # 타겟 네트워크 업데이트
    if episode % target_update_interval == 0:
        target_net.load_state_dict(policy_net.state_dict())

    episode_rewards.append(total_reward)
    if (episode+1) % 50 == 0:
        avg_reward = np.mean(episode_rewards[-50:])
        print(f"Episode: {episode+1}, Avg.Reward(최근50): {avg_reward:.2f}, Epsilon: {epsilon:.3f}")


env.close()


# ------------------------------------
# 6. 학습된 정책으로 시연 & mp4 저장
# ------------------------------------
# 최종 정책으로 1 에피소드 실행하면서 매 스텝 이미지를 가져와 mp4로 만들기
test_env = gym.make(env_id, render_mode='rgb_array')

frames = []
state, _ = test_env.reset()
done = False

for t in range(max_steps_per_episode):
    state_t = torch.LongTensor([state]).to(device)
    with torch.no_grad():
        q_values = policy_net(state_t)
    action = q_values.argmax().item()

    next_state, reward, done, truncated, info = test_env.step(action)
    
    # 프레임 수집 (Gym 0.26+)
    frame = test_env.render()
    frames.append(frame)

    state = next_state
    if done or truncated:
        break

test_env.close()

output_filename = "cliffwalking_dqn_result.mp4"
imageio.mimsave(output_filename, frames, fps=30)
print(f"학습 결과 동영상을 '{output_filename}'로 저장했습니다.")
