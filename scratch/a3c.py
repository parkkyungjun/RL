import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.optim as optim
from torch.distributions import Normal

# -------------------------------
# 1. Hyperparameters
# -------------------------------
ENV_NAME = "Ant-v4"
NUM_PROCESSES = 8       # CPU 코어 수에 맞춰 조절
LR = 1e-4
GAMMA = 0.99
G_AE_LAMBDA = 0.95      # Generalized Advantage Estimation (Optional, 여기선 n-step 사용)
ENTROPY_BETA = 0.01     # 탐험을 위한 엔트로피 가중치
MAX_EPISODES = 3000     # 전체 학습 에피소드 수
T_MAX = 5               # n-step bootstrapping

# -------------------------------
# 2. Shared Optimizer (Adam)
# -------------------------------
class SharedAdam(optim.Adam):
    """멀티프로세싱을 위해 모멘텀 데이터를 공유 메모리에 저장하는 Optimizer"""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

# -------------------------------
# 3. Actor-Critic Network
# -------------------------------
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Shared Feature Extractor
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)

        # Actor Head (Continuous Action: Mean & Std)
        self.mu = nn.Linear(128, action_dim)
        self.sigma = nn.Linear(128, action_dim)

        # Critic Head (Value)
        self.v = nn.Linear(128, 1)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.constant_(module.bias, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mu = torch.tanh(self.mu(x))  # Ant의 action 범위는 보통 [-1, 1]
        sigma = F.softplus(self.sigma(x)) + 1e-5  # 표준편차는 항상 양수

        v = self.v(x)
        return mu, sigma, v

# -------------------------------
# 4. Worker Process
# -------------------------------
def worker(rank, global_model, optimizer, global_ep, global_ep_r, res_queue):
    local_env = gym.make(ENV_NAME)
    input_dim = local_env.observation_space.shape[0]
    action_dim = local_env.action_space.shape[0]

    local_model = ActorCritic(input_dim, action_dim)
    
    # 로컬 모델을 글로벌 모델과 동기화
    local_model.load_state_dict(global_model.state_dict())

    state, _ = local_env.reset()
    done = True
    
    while global_ep.value < MAX_EPISODES:
        s_lst, a_lst, r_lst = [], [], []
        
        # Local Model을 Global Model의 weight로 동기화
        local_model.load_state_dict(global_model.state_dict())

        # t_max 만큼 진행 (n-step rollout)
        for t in range(T_MAX):
            state_tensor = torch.from_numpy(state).float()
            mu, sigma, v = local_model(state_tensor)
            
            # Action Sampling (Reparameterization trick은 A3C에서 필수 아님)
            dist = Normal(mu, sigma)
            action = dist.sample()
            
            # Action Clipping (환경 허용 범위 내로)
            action_np = action.clamp(-1.0, 1.0).numpy()
            
            next_state, reward, terminated, truncated, _ = local_env.step(action_np)
            done = terminated or truncated

            # MuJoCo Ant는 넘어지면 종료되므로, reward scaling이 중요할 수 있음
            # reward = reward / 10.0  # (Optional)

            s_lst.append(state_tensor)
            a_lst.append(action)
            r_lst.append(reward)

            state = next_state
            if done:
                state, _ = local_env.reset()
                break

        # R (n-step return target) 계산
        R = 0.0
        if not done:
            next_state_tensor = torch.from_numpy(state).float()
            _, _, v_next = local_model(next_state_tensor)
            R = v_next.item()

        # Loss 계산을 위한 Tensor 변환
        s_batch = torch.stack(s_lst)
        a_batch = torch.stack(a_lst)
        r_batch = []
        
        # Reverse Order로 Return 계산
        for r in r_lst[::-1]:
            R = r + GAMMA * R
            r_batch.append(R)
        r_batch.reverse()
        r_batch = torch.tensor(r_batch, dtype=torch.float).unsqueeze(1)

        # Forward Pass
        mu, sigma, values = local_model(s_batch)
        td_target = r_batch
        advantage = td_target - values

        dist = Normal(mu, sigma)
        log_probs = dist.log_prob(a_batch).sum(dim=1, keepdim=True) # 다차원 action 합산
        entropy = dist.entropy().sum(dim=1, keepdim=True).mean()

        # Loss Function
        critic_loss = F.mse_loss(values, td_target)
        actor_loss = -(log_probs * advantage.detach()).mean() # Advantage는 Critic이 계산하므로 detach
        
        total_loss = actor_loss + 0.5 * critic_loss - ENTROPY_BETA * entropy

        # Update Global Model
        optimizer.zero_grad()
        total_loss.backward()
        
        # Local Gradients -> Global Gradients
        for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
            if local_param.grad is not None:
                global_param._grad = local_param.grad
        
        optimizer.step()

        # Logging
        if done:
            with global_ep.get_lock():
                global_ep.value += 1
            with global_ep_r.get_lock():
                if global_ep_r.value == 0.:
                    global_ep_r.value = sum(r_lst)
                else:
                    global_ep_r.value = global_ep_r.value * 0.99 + sum(r_lst) * 0.01
            
            if global_ep.value % 10 == 0:
                print(f"Episode: {global_ep.value} | Moving Avg Reward: {global_ep_r.value:.2f}")
            
            res_queue.put(global_ep_r.value)

# -------------------------------
# 5. Main Entry Point
# -------------------------------
if __name__ == '__main__':
    mp.set_start_method('spawn') # Mac/Linux 호환성 및 MuJoCo 충돌 방지

    # 환경 정보 가져오기용 임시 Env
    temp_env = gym.make(ENV_NAME)
    input_dim = temp_env.observation_space.shape[0]
    action_dim = temp_env.action_space.shape[0]
    
    # Global Model & Optimizer
    global_model = ActorCritic(input_dim, action_dim)
    global_model.share_memory() # 프로세스 간 메모리 공유 필수
    
    optimizer = SharedAdam(global_model.parameters(), lr=LR)

    # 공유 변수
    global_ep = mp.Value('i', 0)
    global_ep_r = mp.Value('d', 0.)
    res_queue = mp.Queue()

    # 프로세스 생성
    workers = []
    for rank in range(NUM_PROCESSES):
        p = mp.Process(target=worker, args=(rank, global_model, optimizer, global_ep, global_ep_r, res_queue))
        p.start()
        workers.append(p)

    # 모든 프로세스 종료 대기
    [w.join() for w in workers]