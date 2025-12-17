import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class C51Network(nn.Module):
    def __init__(self, state_dim, action_dim, num_atoms=51, v_min=-10.0, v_max=10.0):
        super(C51Network, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        # 지지대(Support/Atom) 생성: v_min ~ v_max를 51개로 쪼갬
        # 예: [-10, -9.6, ..., 10]
        self.register_buffer(
            "support", 
            torch.linspace(v_min, v_max, num_atoms)
        )
        self.delta_z = (v_max - v_min) / (num_atoms - 1)

        # 신경망 구조
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        
        # [중요] 출력층: 행동 개수 * 51개
        self.fc_out = nn.Linear(128, action_dim * num_atoms)

    def forward(self, x):
        dist = F.relu(self.fc1(x))
        dist = F.relu(self.fc2(dist))
        dist = self.fc_out(dist)
        
        # (Batch, Action, Atoms) 형태로 변환
        dist = dist.view(-1, self.action_dim, self.num_atoms)
        
        # [중요] 마지막 차원(Atom)에 대해 Softmax -> 확률 분포로 만듦
        probs = F.softmax(dist, dim=2) 
        return probs

    def get_q_value(self, x):
        # Q값(기댓값) = sum(확률 * 지지대 값)
        # 물리학에서 기댓값 구하는 것과 동일: <E> = sum P(x) * x
        probs = self.forward(x)
        weights = probs * self.support  # 확률 * 값
        q_values = weights.sum(dim=2)   # 합산
        return q_values
    

def compute_c51_loss(model, target_model, batch, gamma=0.99):
    states, actions, rewards, next_states, dones = batch
    batch_size = states.size(0)
    num_atoms = model.num_atoms
    v_min, v_max = model.v_min, model.v_max
    delta_z = model.delta_z
    support = model.support # [-10, ... , 10]

    # 1. Next State의 분포 예측 (Target Network)
    with torch.no_grad():
        # 다음 상태에서의 분포 예측
        next_dist = target_model(next_states) # Shape: (Batch, Action, Atoms)
        
        # 다음 상태에서 가장 좋은 행동 선택 (Main Network가 선택 - DDQN 방식)
        # 기댓값(Q) 기준으로 argmax
        next_action_q = model.get_q_value(next_states)
        next_actions = next_action_q.argmax(1) 
        
        # 선택된 행동에 해당하는 분포만 가져오기
        next_dist = next_dist[range(batch_size), next_actions] # Shape: (Batch, Atoms)

        # ------------------------------------------------------
        # 2. 분포 투영 (Projection) - 핵심 로직
        # ------------------------------------------------------
        # Tz = R + gamma * z (분포 이동 및 축소)
        Tz = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * gamma * support.unsqueeze(0)
        
        # 값이 범위를 벗어나지 않게 자름 (Clip)
        Tz = Tz.clamp(min=v_min, max=v_max)
        
        # 투영할 인덱스 계산 (b = (값 - 최소값) / 간격)
        b = (Tz - v_min) / delta_z
        l = b.floor().long() # 내림 (왼쪽 인덱스)
        u = b.ceil().long()  # 올림 (오른쪽 인덱스)

        # m: 투영된 분포를 담을 빈 그릇
        m = torch.zeros(batch_size, num_atoms).to(states.device)
        
        # 확률 질량을 이웃한 두 인덱스(l, u)에 거리에 비례해서 분배
        # Linear Interpolation: 가까운 쪽에 더 많이 줌
        
        # Case 1: l(왼쪽)에 줄 양 = (오른쪽 인덱스 - 실제 위치) * 확률
        offset_l = (u.float() - b) * next_dist
        # Case 2: u(오른쪽)에 줄 양 = (실제 위치 - 왼쪽 인덱스) * 확률
        offset_u = (b - l.float()) * next_dist
        
        # scatter_add_로 해당 인덱스에 확률 누적
        m.scatter_add_(1, l, offset_l)
        m.scatter_add_(1, u, offset_u)
        
    # 3. 현재 State의 분포 예측 (Main Network)
    current_dist = model(states)
    # 실제 수행한 Action의 분포만 가져오기
    current_dist = current_dist[range(batch_size), actions.squeeze()]

    # 4. Cross Entropy Loss 계산
    # KL Divergence와 유사하지만, Target(m)이 고정값이므로 Cross Entropy만 최소화하면 됨
    # log(0) 방지를 위해 아주 작은 값(1e-6) 더함
    loss = - (m * torch.log(current_dist + 1e-6)).sum(dim=1).mean()
    
    return loss

import gymnasium as gym
import torch.optim as optim
import random
from collections import deque

# 하이퍼파라미터
v_min, v_max = 0.0, 200.0 # CartPole은 최대 500점이지만, Discount 고려해서 적당히 설정
gamma = 0.99
lr = 1e-3
batch_size = 64

# 환경 및 모델 설정
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

main_net = C51Network(state_dim, action_dim, v_min=v_min, v_max=v_max)
target_net = C51Network(state_dim, action_dim, v_min=v_min, v_max=v_max)
target_net.load_state_dict(main_net.state_dict())

optimizer = optim.Adam(main_net.parameters(), lr=lr)
replay_buffer = deque(maxlen=10000)

# 학습 루프
score = 0
state, _ = env.reset()

for step in range(10000):
    # 행동 선택 (Epsilon-greedy)
    # 분포 기반이므로, 기댓값을 계산해서 행동 선택
    if random.random() < 0.1: # 간단히 epsilon 0.1 고정
        action = env.action_space.sample()
    else:
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = main_net.get_q_value(state_t) # 기댓값 계산
            action = q_values.argmax().item()

    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    
    replay_buffer.append((state, action, reward, next_state, done))
    state = next_state
    score += reward

    if done:
        print(f"Step: {step}, Score: {score}")
        state, _ = env.reset()
        score = 0

    # 학습
    if len(replay_buffer) > batch_size:
        batch = random.sample(replay_buffer, batch_size)
        
        # 데이터 전처리
        states = torch.FloatTensor([x[0] for x in batch])
        actions = torch.LongTensor([x[1] for x in batch]).unsqueeze(1)
        rewards = torch.FloatTensor([x[2] for x in batch])
        next_states = torch.FloatTensor([x[3] for x in batch])
        dones = torch.FloatTensor([x[4] for x in batch])

        # Loss 계산 및 업데이트
        loss = compute_c51_loss(main_net, target_net, 
                                (states, actions, rewards, next_states, dones), 
                                gamma)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 타겟 네트워크 업데이트 (Soft or Hard)
    if step % 100 == 0:
        target_net.load_state_dict(main_net.state_dict())