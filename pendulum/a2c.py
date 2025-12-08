import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from gymnasium.wrappers import RecordVideo

# -----------------------------------------
# 1. 하이퍼파라미터
# -----------------------------------------
learning_rate = 2e-3
gamma = 0.98
num_envs = 8        # 8개의 환경을 동시에 실행 (병렬 처리)
n_steps = 10         # N-step Return (5스텝마다 업데이트)
max_updates = 3000  # 총 업데이트 횟수

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")

# -----------------------------------------
# 2. 벡터화된 환경 생성 (Parallel Environments)
# -----------------------------------------
def make_env():
    # 람다 함수 내부에서 gym.make를 호출해야 각 프로세스에서 별도로 생성됨
    def _thunk():
        env = gym.make('CartPole-v1', max_episode_steps=1000)
        return env
    return _thunk

# SyncVectorEnv: 여러 환경을 순차적으로 실행하지만 입출력은 배치로 묶어줌
# (CPU 부하가 적은 CartPole은 Sync가 Async보다 빠를 수 있음)
envs = gym.vector.SyncVectorEnv([make_env() for _ in range(num_envs)])

# -----------------------------------------
# 3. Actor-Critic 네트워크 (배치 처리 가능)
# -----------------------------------------
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        feat_dim = 256
        self.fc1 = nn.Linear(4, feat_dim)
        self.actor = nn.Linear(feat_dim, 2)
        self.critic = nn.Linear(feat_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # dim=1 주의: 입력이 (batch_size, 4)이므로 두 번째 차원에서 Softmax
        prob = F.softmax(self.actor(x), dim=1)
        value = self.critic(x)
        return prob, value

# -----------------------------------------
# 4. 학습 루프
# -----------------------------------------
def train():
    model = ActorCritic().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # 초기 상태 (8개의 환경에 대한 상태가 한 번에 나옴)
    # s shape: (8, 4)
    s, _ = envs.reset()
    s = torch.from_numpy(s).float().to(device)

    for update in range(max_updates):
        
        # N-step 동안의 데이터를 모을 버퍼
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropies = []

        # --- N-step 진행 (Rollout) ---
        for _ in range(n_steps):
            # 1. 행동 결정
            prob, value = model(s)
            dist = torch.distributions.Categorical(prob)
            action = dist.sample()

            # 2. 환경 진행 (8개 환경 동시 진행)
            # next_s shape: (8, 4), reward shape: (8,)
            next_s, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            
            # x_pos = next_s[:, 0]
            # theta_dot = next_s[:, 3]
            # dynamic_bonus = np.abs(theta_dot)
            
            # dist_penalty = np.abs(x_pos) / 2.4
            reward = reward# - dist_penalty + dynamic_bonus
            # 3. 데이터 저장
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            
            log_probs.append(log_prob)
            values.append(value.squeeze(1)) # (8, 1) -> (8,)
            rewards.append(torch.from_numpy(reward).float().to(device))
            entropies.append(entropy)
            
            # 종료 여부 마스크 (끝났으면 0, 안 끝났으면 1)
            # terminated나 truncated가 True면 done=True
            done = np.logical_or(terminated, truncated)
            mask = 1 - terminated
            masks.append(torch.from_numpy(mask).float().to(device))
            
            s = torch.from_numpy(next_s).float().to(device)

        # --- N-step Return 계산 (Bootstrapping) ---
        # 마지막 상태의 가치 예측 (V(s_{t+N}))
        _, next_value = model(s)
        next_value = next_value.squeeze(1)
        
        returns = []
        R = next_value
        
        # 뒤에서부터 계산 (Backwards)
        for step in reversed(range(n_steps)):
            # R = r + gamma * R * mask
            # mask가 0이면(게임 끝), 미래 가치 R을 0으로 만들어버림 (Reset 효과)
            R = rewards[step] + gamma * R * masks[step]
            returns.insert(0, R)
            
        returns = torch.stack(returns)      # (5, 8)
        values = torch.stack(values)        # (5, 8)
        log_probs = torch.stack(log_probs)  # (5, 8)
        entropies = torch.stack(entropies)  # (5, 8)

        # --- Loss 계산 ---
        # 1. Advantage = Return - Value (이미 N-step Return이 계산됨)
        advantage = returns - values
        
        # 2. Actor Loss
        actor_loss = -(log_probs * advantage.detach()).mean()
        
        # 3. Critic Loss (MSE)
        critic_loss = F.mse_loss(values, returns)
        
        # 4. Entropy Loss (탐험 유도, 선택사항이지만 A2C에서 중요)
        entropy_loss = -entropies.mean()

        total_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss

        # --- 업데이트 ---
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient Clipping (학습 안정화 필수 테크닉)
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        
        optimizer.step()

        if update % 100 == 0:
            # 첫 번째 환경의 보상만 대략적으로 확인
            print(f"Update: {update}, Loss: {total_loss.item():.4f}, Value Mean: {values.mean().item():.2f}")

    print("Training Finished.")
    envs.close()
    
    # RecordVideo: 0번째 에피소드를 녹화
    env = gym.make('CartPole-v1', render_mode='rgb_array', max_episode_steps=2000)
    env = RecordVideo(env, video_folder='./cartpole_video', episode_trigger=lambda x: x == 0)

    s, _ = env.reset()
    done = False
    
    while not done:
        # [수정됨] 테스트 때도 입력을 GPU로 보내야 함
        s_tensor = torch.from_numpy(s).float().to(device)
        
        prob, _ = model(s_tensor.unsqueeze(0))
        a = torch.argmax(prob).item()
        
        s, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
    
    env.close()
    print("Video saved in './cartpole_video' folder.")

if __name__ == "__main__":
    train()