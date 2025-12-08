import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from gymnasium.wrappers import RecordVideo

# -----------------------------------------
# 1. 하이퍼파라미터 (PPO 전용 추가)
# -----------------------------------------
learning_rate = 2.5e-4  # PPO는 보통 조금 더 낮은 LR 사용
gamma = 0.99
gae_lambda = 0.95       # GAE 파라미터
clip_coef = 0.2         # PPO Clip 범위 (epsilon)
ent_coef = 0.01         # Entropy 계수
vf_coef = 0.5           # Value loss 계수
max_grad_norm = 0.5

num_envs = 8
n_steps = 128           # PPO는 더 긴 호라이즌을 한 번에 모음 (배치 크기 키우기)
batch_size = int(num_envs * n_steps)
minibatch_size = 256    # 업데이트 시 쪼개서 학습
update_epochs = 4       # 모은 데이터로 몇 번 반복 학습할지
max_updates = 500       # (batch_size * max_updates) steps

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")

# -----------------------------------------
# 2. 환경 생성
# -----------------------------------------
def make_env():
    def _thunk():
        env = gym.make('CartPole-v1', max_episode_steps=500)
        return env
    return _thunk

envs = gym.vector.SyncVectorEnv([make_env() for _ in range(num_envs)])

# -----------------------------------------
# 3. 네트워크 (Actor-Critic)
# -----------------------------------------
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        # PPO는 보통 초기화를 Orthogonal로 하면 학습이 더 안정적입니다.
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.actor = nn.Linear(64, 2)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return x

    def get_action_and_value(self, x, action=None):
        hidden = self.forward(x)
        logits = self.actor(hidden)
        probs = torch.distributions.Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

# -----------------------------------------
# 4. 학습 루프
# -----------------------------------------
def train():
    model = ActorCritic().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, eps=1e-5)

    # 데이터 저장용 텐서 미리 할당 (메모리 효율 및 속도 향상)
    obs = torch.zeros((n_steps, num_envs, 4)).to(device)
    actions = torch.zeros((n_steps, num_envs)).to(device)
    logprobs = torch.zeros((n_steps, num_envs)).to(device)
    rewards = torch.zeros((n_steps, num_envs)).to(device)
    dones = torch.zeros((n_steps, num_envs)).to(device)
    values = torch.zeros((n_steps, num_envs)).to(device)

    # 초기 상태
    global_step = 0
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(num_envs).to(device)

    for update in range(1, max_updates + 1):
        
        # --- A. Rollout: 데이터 수집 ---
        for step in range(n_steps):
            global_step += 1 * num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = model.get_action_and_value(next_obs)
                values[step] = value.flatten()
            
            actions[step] = action
            logprobs[step] = logprob

            # 환경 진행
            real_next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            # x_pos = real_next_obs[:, 0]
            # theta_dot = real_next_obs[:, 3]
            # dynamic_bonus = np.abs(theta_dot)
            
            # dist_penalty = np.abs(x_pos) / 2.4
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            
            # 다음 상태 준비
            next_obs = torch.Tensor(real_next_obs).to(device)
            # 종료 여부 (PPO GAE 계산을 위해 필요)
            next_done = np.logical_or(terminated, truncated)
            next_done = torch.Tensor(next_done).to(device)

        # --- B. GAE (Generalized Advantage Estimation) 계산 ---
        with torch.no_grad():
            next_value = model.get_action_and_value(next_obs)[3].reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            
            # 역순으로 계산 (Backwards)
            for t in reversed(range(n_steps)):
                if t == n_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                
                # Delta = r + gamma * V(s') * mask - V(s)
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                
                # Adv = Delta + (gamma * lambda * mask) * Adv_next
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            
            returns = advantages + values

        # --- C. 데이터 Flatten (Batch로 만들기) ---
        b_obs = obs.reshape((-1, 4))                     # (n_steps * num_envs, 4)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # -----------------------------------------
        # D. PPO Update (Epochs & Minibatch)
        # -----------------------------------------
        b_inds = np.arange(batch_size)
        clipfracs = []
        
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds) # 인덱스 섞기
            
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                # 현재 모델로 다시 평가 (Graph 생성)
                # 여기서 actions인자를 넣어주면 해당 action에 대한 새로운 log_prob를 구해줌
                _, newlogprob, entropy, newvalue = model.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp() # pi_new / pi_old

                # Debug: 클리핑이 얼마나 일어나는지 확인용
                with torch.no_grad():
                    clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                # Normalizing Advantage (학습 안정화에 매우 중요)
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # 1. Policy Loss (Clipped Surrogate)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # 2. Value Loss (MSE)
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # 3. Entropy Loss (탐험 유도)
                entropy_loss = entropy.mean()
                
                # 최종 Loss
                loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        if update % 20 == 0:
            print(f"Update: {update}, Loss: {loss.item():.4f}, "
                  f"Value: {b_values.mean().item():.2f}, "
                  f"Reward Mean: {rewards.sum().item() / num_envs:.2f}")

    print("Training Finished.")
    envs.close()
    
    # -----------------------------------------
    # 5. 테스트 및 비디오 저장
    # -----------------------------------------
    test_env = gym.make('CartPole-v1', render_mode='rgb_array')
    test_env = RecordVideo(test_env, video_folder='./ppo_cartpole_video', episode_trigger=lambda x: True)

    s, _ = test_env.reset()
    done = False
    
    with torch.no_grad():
        while not done:
            s_tensor = torch.Tensor(s).to(device).unsqueeze(0)
            action, _, _, _ = model.get_action_and_value(s_tensor)
            s, r, terminated, truncated, _ = test_env.step(action.item())
            done = terminated or truncated
    
    test_env.close()
    print("Video saved.")

if __name__ == "__main__":
    train()