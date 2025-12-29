import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from gymnasium.wrappers import RecordVideo

# --- 하이퍼파라미터 ---
LR_ACTOR = 0.0001      # 로봇 제어는 학습률을 조금 낮추는 게 안정적일 때가 많습니다.
LR_CRITIC = 0.001
GAMMA = 0.99
TAU = 0.005
MEMORY_SIZE = 100000   # 버퍼 사이즈를 좀 늘렸습니다.
BATCH_SIZE = 64
OU_NOISE_THETA = 0.15
OU_NOISE_SIGMA = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. 유틸리티 클래스 (버퍼, 노이즈) ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (torch.FloatTensor(np.array(state)).to(device),
                torch.FloatTensor(np.array(action)).to(device),
                torch.FloatTensor(np.array(reward)).unsqueeze(1).to(device),
                torch.FloatTensor(np.array(next_state)).to(device),
                torch.FloatTensor(np.array(done)).unsqueeze(1).to(device))

    def __len__(self):
        return len(self.buffer)

class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=OU_NOISE_THETA, sigma=OU_NOISE_SIGMA):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

# --- 2. 네트워크 (BipedalWalker용 확장 버전) ---
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        # BipedalWalker는 상태가 24개라 뇌가 좀 커야 함 (400 -> 300)
        self.net = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, state):
        return self.net(state) * self.max_action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)

# --- 3. DDPG 에이전트 클래스 (이게 빠져 있었음!) ---
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=LR_ACTOR)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=LR_CRITIC)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.max_action = max_action
        self.replay_buffer = ReplayBuffer(MEMORY_SIZE)
        self.noise = OUNoise(action_dim)

    def get_action(self, state, noise=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()[0]
        self.actor.train()

        if noise:
            action += self.noise.sample()
        
        return np.clip(action, -self.max_action, self.max_action)

    def update(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(BATCH_SIZE)

        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q = self.critic_target(next_state, next_action)
            target_q = reward + (1 - done) * GAMMA * target_q

        current_q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

# --- 4. 메인 학습 루프 ---
if __name__ == "__main__":
    # 학습 때는 화면 렌더링을 끄는 게 속도에 좋습니다 (render_mode=None)
    env = gym.make('BipedalWalker-v3', render_mode=None)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = DDPGAgent(state_dim, action_dim, max_action)
    
    EPISODES = 1000  # BipedalWalker는 1000번은 해야 좀 걷습니다.
    print(f"Start Training BipedalWalker-v3 on {device}...")

    for episode in range(EPISODES):
        state, _ = env.reset()
        agent.noise.reset()
        episode_reward = 0
        
        # BipedalWalker는 최대 스텝이 1600으로 깁니다.
        for step in range(1600):
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # BipedalWalker는 넘어지면 -100점을 받고 끝납니다.
            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.update()
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}")
                
    print("Training Finished! Starting Test & Recording...")

    # 1. 기존 환경 닫기
    env.close()

    # 2. 테스트용 환경 설정 (중요: render_mode='rgb_array')
    test_env = gym.make('BipedalWalker-v3', render_mode='rgb_array')

    # 3. 비디오 녹화 래퍼(Wrapper) 씌우기
    # - video_folder: 영상 저장될 폴더명
    # - episode_trigger: 어떤 에피소드를 녹화할지 결정 (여기선 모든 에피소드 녹화)
    test_env = RecordVideo(
        test_env, 
        video_folder='./videos', 
        name_prefix='bipedal_walker_result',
        episode_trigger=lambda x: True 
    )

    # 4. 테스트 루프 (무한 루프가 아니라 횟수를 정해야 파일이 저장됩니다)
    TEST_EPISODES = 3  # 3번 정도 걷는 것을 녹화

    for ep in range(TEST_EPISODES):
        state, _ = test_env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # 노이즈 없이 순수 실력으로 걷기
            action = agent.get_action(state, noise=False)
            state, reward, terminated, truncated, _ = test_env.step(action)
            done = terminated or truncated
            total_reward += reward

        print(f"Test Episode {ep+1} Result: {total_reward:.2f}")

    # 5. 환경을 닫아야 비디오 파일이 최종 저장됩니다.
    test_env.close()
    print("Video saved in './videos' folder!")