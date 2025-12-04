import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

# -----------------------------------------
# 1. 하이퍼파라미터
# -----------------------------------------
learning_rate = 1e-3
gamma = 0.98
n_steps = 10  # [핵심] 몇 스텝마다 업데이트할 것인가?

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")

# -----------------------------------------
# 2. 네트워크 (Forward dim 수정)
# -----------------------------------------
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(4, 256)
        self.fc_pi = nn.Linear(256, 2)
        self.fc_v = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # [수정] 배치가 들어오므로 dim=1이어야 함 (N, 2)
        # 기존 dim=0은 (2) 벡터일 때 쓰는 것
        prob = F.softmax(self.fc_pi(x), dim=-1) 
        value = self.fc_v(x)
        return prob, value

# -----------------------------------------
# 3. 학습 루프
# -----------------------------------------
def train():
    env = gym.make('CartPole-v1')
    model = ActorCritic().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    print("Start Training CartPole (N-step)...")

    for n_epi in range(2000): # 에피소드 수
        s, _ = env.reset()
        done = False
        score = 0.0
        
        # N-step 데이터를 모을 임시 버퍼
        s_lst, a_lst, r_lst = [], [], []

        while not done:
            # 1. 행동 결정
            s_tensor = torch.from_numpy(s).float().unsqueeze(0).to(device) # (1, 4)
            prob, _ = model(s_tensor)
            m = Categorical(prob)
            a = m.sample()

            s_prime, r, terminated, truncated, _ = env.step(a.item())
            done = terminated or truncated
            
            x_pos = s_prime[0]
            theta_dot = s_prime[3]
            dynamic_bonus = np.abs(theta_dot)
            
            dist_penalty = np.abs(x_pos) / 2.4
            r = r# - dist_penalty# + dynamic_bonus
            
            # 버퍼에 저장
            s_lst.append(s)
            a_lst.append(a.item())
            r_lst.append(r/100.0) # 리워드 스케일링

            s = s_prime
            score += r

            # -----------------------------------------------------------
            # [핵심 변경] N-step이 찼거나, 게임이 끝났으면 업데이트!
            # -----------------------------------------------------------
            if len(s_lst) == n_steps or done:
                
                # 1. 마지막 상태의 가치(Next Value) 계산 - 부트스트래핑
                if done:
                    v_final = 1 if truncated else 0 # 죽었으면 미래 가치 0
                else:
                    # 안 죽었으면 Critic이 예측한 미래 가치 사용
                    s_prime_tensor = torch.from_numpy(s_prime).float().unsqueeze(0).to(device)
                    _, v_next = model(s_prime_tensor)
                    v_final = v_next.item()

                # 2. 타겟(Target) 계산 (역순으로)
                # G_t = r_t + gamma * G_{t+1}
                # 시작 G는 v_final (Critic의 예측값)
                R = v_final
                returns = []
                for reward in r_lst[::-1]:
                    R = reward + gamma * R
                    returns.insert(0, R)
                
                # 3. 텐서 변환 (Batch 처리)
                s_batch = torch.tensor(np.array(s_lst), dtype=torch.float).to(device)
                a_batch = torch.tensor(a_lst, dtype=torch.int64).unsqueeze(1).to(device)
                returns = torch.tensor(returns, dtype=torch.float).unsqueeze(1).to(device)

                # 4. 모델 예측 및 Loss 계산
                probs, values = model(s_batch)
                
                advantage = returns - values
                
                pi = probs.gather(1, a_batch)
                
                # Actor Loss + Entropy Bonus (탐험 유도, N-step에서 중요)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1, keepdim=True)
                actor_loss = -(torch.log(pi) * advantage.detach()).mean() - 0.01 * entropy.mean()
                
                critic_loss = F.mse_loss(values, returns)
                
                loss = actor_loss + critic_loss

                # 5. 업데이트
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # [중요] 버퍼 비우기 (다음 5스텝을 위해)
                s_lst, a_lst, r_lst = [], [], []

        if n_epi % 50 == 0:
            print(f"Episode: {n_epi}, Score: {score:.0f}")
            if score > 400:
                print("Solved!")
                break

    env.close()
    
    # -----------------------------------------
    # 4. 테스트 및 영상 저장 (서버용)
    # -----------------------------------------
    print("-" * 30)
    print("Recording Video...")
    
    # RecordVideo: 0번째 에피소드를 녹화
    env = gym.make('CartPole-v1', render_mode='rgb_array', max_episode_steps=2000)
    env = RecordVideo(env, video_folder='./cartpole_video', episode_trigger=lambda x: x == 0)

    s, _ = env.reset()
    done = False
    
    while not done:
        # [수정됨] 테스트 때도 입력을 GPU로 보내야 함
        s_tensor = torch.from_numpy(s).float().to(device)
        
        prob, _ = model(s_tensor)
        a = torch.argmax(prob).item()
        
        s, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
    
    env.close()
    print("Video saved in './cartpole_video' folder.")

if __name__ == '__main__':
    train()