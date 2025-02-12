# main.py
import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
current_dir = os.path.dirname(os.path.abspath(__file__))


from gyms.cartpole_pendulum import PendulumGym, CartPolePendulumEnv
from dqn_utils import DQN, ReplayMemory, Transition
import torch
import torch.optim as optim
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from itertools import count

BATCH_SIZE = 64
GAMMA = 0.97
EPS_START = 1
EPS_END = 0.05
EPS_DECAY = 5000
TAU = 0.005
LR = 1e-3
REPLAY_MEMORY_SIZE = 20000
print(GAMMA)
# LR            = 0.01
# GAMMA         = 0.98
# tau           = 0.01
# EPS_START       = 1.0
# EPS_END   = 0.001
# REPLAY_MEMORY_SIZE   = 100000
# BATCH_SIZE    = 200

# ======================================
INTERVAL_RESULT_PRINT = 20
INTERVAL_MODEL_SAVE = 100
NUM_EPISODES = 1000
# 
steps_done = 0

# Load an existing model
model_file_path = 'policy_net.pth'  # 이어서 학습할 모델의 경로 지정
load_existing_model = False
if os.path.exists(f"{current_dir}/{model_file_path}"):
    load_existing_model = os.path.isfile(f"{current_dir}/{model_file_path}")

# Result 
result_folder_path = 'results'
os.makedirs(f"{current_dir}/{result_folder_path}", exist_ok=True)  

def calculate_score(next_state):
    theta = next_state[2]  # 각도
    theta_dot = next_state[3]  # 각속도
    pos = next_state[0]  # 위치
    score = 0
    
    if -0.2 <= theta <= 0.2 and -1.0 <= theta_dot <= 1.0 and -0.1 <= pos <= 0.1:
        score += 2
    elif -0.2 <= theta <= 0.2 and -1.0 <= theta_dot <= 1.0:
        score += 1
    elif -0.4 <= theta <= 0.4 and -1.0 <= theta_dot <= 1.0:
        score += 0.5
    elif -0.2 <= theta <= 0.2 and -2.0 <= theta_dot <= 2.0:
        score += 0.5
    elif -0.4 <= theta <= 0.4 and -2.0 <= theta_dot <= 2.0:
        score += 0.1

    return score

def select_action(state, policy_net, device, EPS_START, EPS_END, EPS_DECAY):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1), eps_threshold
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long), eps_threshold
    
def optimize_model(memory, policy_net, target_net, optimizer, device, BATCH_SIZE = 128, GAMMA = 0.99):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)    
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    if non_final_next_states.size(0) > 0:
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = torch.nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)    
    optimizer.step()

def save_model(model, episode):
    torch.save(model.state_dict(), f'{current_dir}/{result_folder_path}/policy_net_{episode}.pth')
    
def plot_durations(episode_durations, save_plot = False):
    plt.figure(1)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Episode Duration')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.pause(0.001)
    if save_plot:
        plt.savefig(f"{current_dir}/{result_folder_path}/training_plot_{len(episode_durations)}.png")    
    clear_output(wait=True)

def plot_scores(episode_scores, save_plot=False):
    plt.figure(2)
    plt.clf()
    scores_t = torch.tensor(episode_scores, dtype=torch.float)
    plt.title('Episode Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.plot(scores_t.numpy())
    if len(scores_t) >= 100:
        means = scores_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.pause(0.001)
    if save_plot:
        plt.savefig(f"{current_dir}/{result_folder_path}/score_plot_{len(episode_scores)}.png")
    clear_output(wait=True)        

class RewardWrapper():
    def __init__(self, env):
        self.env = env
        self.prev_shaping = None  #이전 shaping 저장 
    
    def step(self, action):        
        next_state, reward, terminated, truncated, _  = self.env.step(action)
  
        pos = next_state[0] #카트 위치  
        vel = next_state[1] #카트 속도  
        theta = next_state[2] #막대 각도 
        theta_dot = next_state[3] #막대 각속도
        
        theta_th = 12 * 2 * math.pi / 360
        
        reward = 1. - (abs(theta) / math.pi) - 0.5 * (abs(pos) / 2.4) - 0.01 * abs(theta_dot)
        
        # if abs(theta) < (math.pi / 6):
        #     reward += 0.51
        #     reward -= 0.01 * abs(theta_dot)
        #     reward -= 0.5 * (abs(pos) / 2.4)
        
        return next_state, reward, terminated, truncated, _
                
    
    
if __name__ == "__main__":
    env = PendulumGym()
    wrapped_env = RewardWrapper(env)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    n_actions = env.get_action_space().n
    state, info = env.reset()
    n_observations = len(state)
    
    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    
    if load_existing_model:
        print(f"Loading existing model from {model_file_path}")
        policy_net.load_state_dict(torch.load(f"{current_dir}/{model_file_path}"))
        print(f"Model loaded successfully from {model_file_path}")
            
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval() 

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    # optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayMemory(REPLAY_MEMORY_SIZE)
    
    episode_durations = []
    episode_scores = []  
    
    best_score = 100
    for i_episode in range(NUM_EPISODES):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        total_rewards = 0
        
        for t in count():
            action, epsilon = select_action(state, policy_net, device, EPS_START, EPS_END, EPS_DECAY)
            next_state, reward, terminated, truncated, _  = wrapped_env.step(action.item())
            
            done = terminated or truncated
            reward = torch.tensor([reward], device=device, dtype=torch.float32)

            if done:
                next_state = None
            else:
                next_state = torch.tensor(next_state, device=device, dtype=torch.float32).unsqueeze(0)

            memory.push(state, action, next_state, reward)
            
            state = next_state

            optimize_model(memory, policy_net, target_net, optimizer, device, BATCH_SIZE, GAMMA)
            
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)            
            
            total_rewards += reward
                        
            if done:
                episode_durations.append(t + 1)
                episode_scores.append(total_rewards.item())  
                plot_durations(episode_durations)
                plot_scores(episode_scores)
                break
            if torch.mean(total_rewards) > best_score:
                best_score = torch.mean(total_rewards)
                save_model(policy_net, 9999)
                print(best_score)
            
        if i_episode % INTERVAL_RESULT_PRINT==0:                
            print(f"n_episode :{i_episode}, score : {np.mean(episode_scores[-INTERVAL_RESULT_PRINT:]):.2f}, n_buffer : {memory.__len__()}, eps : {epsilon:.3f}")    
                            
        if i_episode % INTERVAL_MODEL_SAVE == 0:
            plot_durations(episode_durations, save_plot=True) 
            plot_scores(episode_scores, save_plot=True)
            save_model(policy_net, i_episode)
    
    env.close()
    plot_durations(episode_durations, save_plot=True) 
    print("Training complete.")
