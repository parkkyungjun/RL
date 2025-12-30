import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gymnasium.wrappers import RecordVideo
import gymnasium as gym
import torch.optim as optim
import random
from collections import deque

# [추가] GPU 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class C51Network(nn.Module):
    def __init__(self, state_dim, action_dim, num_atoms=51, v_min=-10.0, v_max=10.0):
        super(C51Network, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        # 지지대(Support/Atom) 생성
        # register_buffer로 등록하면 model.to(device) 호출 시 자동으로 함께 이동합니다.
        self.register_buffer(
            "support", 
            torch.linspace(v_min, v_max, num_atoms)
        )
        self.delta_z = (v_max - v_min) / (num_atoms - 1)

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_out = nn.Linear(128, action_dim * num_atoms)

    def forward(self, x):
        dist = F.relu(self.fc1(x))
        dist = F.relu(self.fc2(dist))
        dist = self.fc_out(dist)
        dist = dist.view(-1, self.action_dim, self.num_atoms)
        probs = F.softmax(dist, dim=2) 
        return probs

    def get_q_value(self, x):
        probs = self.forward(x)
        weights = probs * self.support 
        q_values = weights.sum(dim=2)
        return q_values
    

def compute_c51_loss(model, target_model, batch, gamma=0.99):
    # batch 안의 텐서들은 이미 device에 올라가 있다고 가정합니다.
    states, actions, rewards, next_states, dones = batch
    batch_size = states.size(0)
    
    # model.num_atoms 등은 스칼라 값이므로 device 상관 없음
    num_atoms = model.num_atoms
    v_min, v_max = model.v_min, model.v_max
    delta_z = model.delta_z
    support = model.support # model이 GPU에 있으면 얘도 GPU에 있음

    # 1. Next State의 분포 예측 (Target Network)
    with torch.no_grad():
        next_dist = target_model(next_states) 
        next_action_q = model.get_q_value(next_states)
        next_actions = next_action_q.argmax(1) 
        next_dist = next_dist[range(batch_size), next_actions] 

        # 2. 분포 투영 (Projection)
        # rewards, dones, support 모두 같은 device에 있어야 함
        Tz = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * gamma * support.unsqueeze(0)
        Tz = Tz.clamp(min=v_min, max=v_max)
        b = (Tz - v_min) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        # m 생성 시 device 지정 (states.device 이용)
        m = torch.zeros(batch_size, num_atoms).to(states.device)
        
        offset_l = (u.float() - b) * next_dist
        offset_u = (b - l.float()) * next_dist
        
        m.scatter_add_(1, l, offset_l)
        m.scatter_add_(1, u, offset_u)
        
    # 3. 현재 State의 분포 예측
    current_dist = model(states)
    current_dist = current_dist[range(batch_size), actions.squeeze()]

    # 4. Loss 계산
    loss = - (m * torch.log(current_dist + 1e-6)).sum(dim=1).mean()
    
    return loss

# 하이퍼파라미터
v_min, v_max = 0.0, 150.0 
gamma = 0.99
lr = 1e-3
batch_size = 64
max_episode_steps = 1000
# 환경 및 모델 설정
env = gym.make("CartPole-v1", max_episode_steps=max_episode_steps)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# [수정] 모델을 device로 이동
main_net = C51Network(state_dim, action_dim, v_min=v_min, v_max=v_max).to(device)
target_net = C51Network(state_dim, action_dim, v_min=v_min, v_max=v_max).to(device)
target_net.load_state_dict(main_net.state_dict())

optimizer = optim.AdamW(main_net.parameters(), lr=lr)
replay_buffer = deque(maxlen=100000)

# 학습 루프
score = 0
state, _ = env.reset()

for step in range(20000):
    if random.random() < 0.1 and step < 10000: 
        action = env.action_space.sample()
    else:
        # [수정] 입력 텐서를 device로 이동
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = main_net.get_q_value(state_t)
            action = q_values.argmax().item() # .item()은 CPU 스칼라로 반환

    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    
    replay_buffer.append((state, action, reward, next_state, done))
    state = next_state
    score += reward

    if done:
        print(f"Step: {step}, Score: {score}")
        state, _ = env.reset()
        if score == max_episode_steps:
            break
        score = 0
        

    if len(replay_buffer) > batch_size:
        batch = random.sample(replay_buffer, batch_size)
        
        # [수정] 배치 데이터를 device로 이동
        states = torch.FloatTensor(np.array([x[0] for x in batch])).to(device)
        actions = torch.LongTensor(np.array([x[1] for x in batch])).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(np.array([x[2] for x in batch])).to(device)
        next_states = torch.FloatTensor(np.array([x[3] for x in batch])).to(device)
        dones = torch.FloatTensor(np.array([x[4] for x in batch])).to(device)

        loss = compute_c51_loss(main_net, target_net, 
                                (states, actions, rewards, next_states, dones), 
                                gamma)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if step % 100 == 0:
        target_net.load_state_dict(main_net.state_dict())
        

# 테스트 및 비디오 저장
test_env = gym.make('CartPole-v1', render_mode='rgb_array', max_episode_steps=500)
test_env = RecordVideo(test_env, video_folder='./c51_cartpole_video', episode_trigger=lambda x: True)

s, _ = test_env.reset()
done = False

with torch.no_grad():
    while not done:
        # [수정] 테스트 시에도 입력 텐서를 device로 이동
        s_tensor = torch.Tensor(s).unsqueeze(0).to(device)
        q_values = main_net.get_q_value(s_tensor)
        action = q_values.argmax().item()
        s, r, terminated, truncated, _ = test_env.step(action)
        done = terminated or truncated

test_env.close()
print("Video recording completed.")