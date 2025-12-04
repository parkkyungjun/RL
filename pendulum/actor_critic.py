import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

# -----------------------------------------
# 1. 하이퍼파라미터 및 디바이스 설정
# -----------------------------------------
learning_rate = 1e-3
gamma = 0.99

# GPU가 있으면 쓰고, 없으면 CPU를 씁니다.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")

# -----------------------------------------
# 2. Actor-Critic 네트워크
# -----------------------------------------
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(4, 256)
        self.fc_pi = nn.Linear(256, 2)
        self.fc_v = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        prob = F.softmax(self.fc_pi(x), dim=0)
        value = self.fc_v(x)
        return prob, value

# -----------------------------------------
# 3. 학습 루프
# -----------------------------------------
def train():
    env = gym.make('CartPole-v1')
    
    # 모델을 생성하고 GPU로 이동시킵니다 (.to(device))
    model = ActorCritic().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    print("Start Training CartPole...")

    for n_epi in range(1000):
        s, _ = env.reset()
        done = False
        
        s_lst, a_lst, r_lst = [], [], []

        while not done:
            # [수정됨] 입력을 GPU로 이동
            s_tensor = torch.from_numpy(s).float().to(device)
            
            prob, _ = model(s_tensor)
            m = Categorical(prob)
            a = m.sample()

            s_prime, r, terminated, truncated, _ = env.step(a.item())
            done = terminated or truncated
            
            s_lst.append(s)
            a_lst.append(a.item())
            r_lst.append(r/100.0)

            s = s_prime

        # 에피소드 종료 후 배치 학습
        # [수정됨] 모든 텐서를 GPU로 이동
        s_final = torch.tensor(np.array(s_lst), dtype=torch.float).to(device)
        a_final = torch.tensor(a_lst, dtype=torch.int64).unsqueeze(1).to(device)
        # r_final은 학습에 직접 안쓰고 returns 계산용이라 GPU에 안보내도 되지만 일관성을 위해 둠
        
        # 예측값 계산 (GPU에서 연산됨)
        probs, values = model(s_final)
        
        returns = []
        G = 0
        for reward in r_lst[::-1]:
            G = reward + gamma * G
            returns.insert(0, G)
        
        # [수정됨] 타겟값도 GPU로 이동해야 연산 가능
        returns = torch.tensor(returns, dtype=torch.float).unsqueeze(1).to(device)

        advantage = returns - values 

        pi = probs.gather(1, a_final)
        actor_loss = -(torch.log(pi) * advantage.detach()).mean()
        critic_loss = F.mse_loss(values, returns)

        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if n_epi % 50 == 0:
            print(f"Episode: {n_epi}, Score: {sum(r_lst)*100:.0f}")
            if sum(r_lst)*100 > 400:
                print("Solved!")
                break

    env.close()
    
    # -----------------------------------------
    # 4. 테스트 및 영상 저장 (서버용)
    # -----------------------------------------
    print("-" * 30)
    print("Recording Video...")
    
    # RecordVideo: 0번째 에피소드를 녹화
    env = gym.make('CartPole-v1', render_mode='rgb_array')
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