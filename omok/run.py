import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ray
import time
import os
from collections import deque
import random

# =============================================================================
# [1] 하이퍼파라미터
# =============================================================================
BOARD_SIZE = 15
NUM_RES_BLOCKS = 10     
NUM_CHANNELS = 128      
NUM_MCTS_SIMS = 50     
BATCH_SIZE = 1024       
LR = 0.002
BUFFER_SIZE = 500000
NUM_ACTORS = 10
SAVE_INTERVAL = 200     

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# [2] 게임 로직 (Gomoku)
# =============================================================================
class Gomoku:
    def __init__(self):
        self.size = BOARD_SIZE
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.current_player = 1 
        self.last_move = None
        self.move_count = 0

    def reset(self):
        self.board.fill(0)
        self.current_player = 1
        self.last_move = None
        self.move_count = 0
        return self.get_state()

    def step(self, action):
        x, y = action // self.size, action % self.size
        if self.board[x, y] != 0:
            return self.get_state(), -10, True 
        
        self.board[x, y] = self.current_player
        self.last_move = (x, y)
        self.move_count += 1
        
        done, winner = self.check_win(x, y)
        reward = 0
        if done:
            if winner == self.current_player: reward = 1
            elif winner == 0: reward = 0 
        
        self.current_player *= -1
        return self.get_state(), reward, done

    def get_state(self):
        state = np.zeros((3, self.size, self.size), dtype=np.float32)
        if self.current_player == 1:
            state[0] = (self.board == 1)
            state[1] = (self.board == -1)
        else:
            state[0] = (self.board == -1)
            state[1] = (self.board == 1)
        state[2].fill(1.0) 
        return state

    def get_legal_actions(self):
        return np.where(self.board.flatten() == 0)[0]

    def check_win(self, x, y):
        player = self.board[x, y]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 1
            for d in [1, -1]:
                nx, ny = x, y
                while True:
                    nx, ny = nx + dx*d, ny + dy*d
                    if 0 <= nx < self.size and 0 <= ny < self.size and self.board[nx, ny] == player:
                        count += 1
                    else: break
            if count >= 5: return True, player
        
        if self.move_count == self.size * self.size: return True, 0 
        return False, 0

# =============================================================================
# [3] 신경망 (ResNet)
# =============================================================================
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        res = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + res)

class AlphaZeroNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.start_conv = nn.Conv2d(3, NUM_CHANNELS, 3, padding=1)
        self.bn_start = nn.BatchNorm2d(NUM_CHANNELS)
        self.backbone = nn.Sequential(*[ResBlock(NUM_CHANNELS) for _ in range(NUM_RES_BLOCKS)])
        
        self.policy_head = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, 2, 1), nn.BatchNorm2d(2), nn.ReLU(),
            nn.Flatten(), nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, 1, 1), nn.BatchNorm2d(1), nn.ReLU(),
            nn.Flatten(), nn.Linear(BOARD_SIZE * BOARD_SIZE, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Tanh()
        )

    def forward(self, x):
        x = F.relu(self.bn_start(self.start_conv(x)))
        x = self.backbone(x)
        policy = F.log_softmax(self.policy_head(x), dim=1)
        value = self.value_head(x)
        return policy, value

# =============================================================================
# [4] MCTS 
# =============================================================================
class Node:
    def __init__(self, prior):
        self.prior = prior
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0

    def value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0

    def select_child(self):
        best_score = -float('inf')
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            ucb = child.value() + 1.0 * child.prior * np.sqrt(self.visit_count) / (1 + child.visit_count)
            if ucb > best_score:
                best_score = ucb
                best_action = action
                best_child = child
        return best_action, best_child

class MCTS:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def search(self, game, root):
        for _ in range(NUM_MCTS_SIMS):
            node = root
            scratch_game = self.clone_game(game) 
            search_path = [node]
            
            while node.children:
                action, node = node.select_child()
                scratch_game.step(action)
                search_path.append(node)

            state = torch.tensor(scratch_game.get_state(), dtype=torch.float32).unsqueeze(0)
            
            # [수정] Worker는 CPU만 씁니다 (DEVICE가 아닌 'cpu' 명시)
            policy_logits, value = self.model(state)
            
            policy = torch.exp(policy_logits).numpy()[0]
            value = value.item()

            legal_moves = scratch_game.get_legal_actions()
            policy_mask = np.zeros(BOARD_SIZE*BOARD_SIZE)
            policy_mask[legal_moves] = 1
            policy = policy * policy_mask
            policy /= np.sum(policy) + 1e-8

            for action in legal_moves:
                if action not in node.children:
                    node.children[action] = Node(policy[action])

            for node in reversed(search_path):
                node.value_sum += value
                node.visit_count += 1
                value = -value 

    def clone_game(self, game):
        new_game = Gomoku()
        new_game.board = game.board.copy()
        new_game.current_player = game.current_player
        return new_game

# =============================================================================
# [5] Ray Actors
# =============================================================================
@ray.remote
class DataWorker:
    def __init__(self, buffer_ref, shared_weights_ref):
        self.buffer_ref = buffer_ref
        self.shared_weights_ref = shared_weights_ref
        self.game = Gomoku()
        self.model = AlphaZeroNet() # Worker는 기본적으로 CPU 모델 생성
        self.mcts = MCTS(self.model)
        self.update_weights() 

    def update_weights(self):
        # [수정] Ray에서 받은 가중치는 이미 CPU에 있으므로 바로 로드 가능
        weights = ray.get(self.shared_weights_ref[0])
        self.model.load_state_dict(weights)

    def run(self):
        while True:
            if random.random() < 0.1: self.update_weights()
            
            state = self.game.reset()
            root = Node(0)
            history = []
            
            while True:
                self.mcts.search(self.game, root)
                
                visits = np.array([child.visit_count for child in root.children.values()])
                actions = list(root.children.keys())
                
                if len(history) < 10: temp = 1.0
                else: temp = 0.1
                
                probs = visits ** (1/temp)
                probs = probs / np.sum(probs)
                
                action = np.random.choice(actions, p=probs)
                
                full_pi = np.zeros(BOARD_SIZE*BOARD_SIZE)
                full_pi[actions] = visits / np.sum(visits)
                
                history.append([state, full_pi, 0]) 
                
                state, _, done = self.game.step(action)
                root = root.children[action] 
                
                if done:
                    winner_reward = 1 
                    for i in reversed(range(len(history))):
                        history[i][2] = winner_reward
                        winner_reward = -winner_reward
                    
                    ray.get(self.buffer_ref.add.remote(history))
                    break

@ray.remote
class ReplayBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=BUFFER_SIZE)
    
    def add(self, history):
        self.buffer.extend(history)
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, pi, z = zip(*batch)
        return np.array(s), np.array(pi), np.array(z)
    
    def size(self):
        return len(self.buffer)

# =============================================================================
# [6] 메인 학습 루프 (Trainer)
# =============================================================================
if __name__ == "__main__":
    if ray.is_initialized(): ray.shutdown()
    ray.init()
    
    print(f"🚀 AlphaZero Scratch 시작! (A6000 + 5950X)")
    
    # 1. 모델 초기화 (GPU)
    model = AlphaZeroNet().to(DEVICE)
    
    # [수정 중요] 공유할 때는 무조건 CPU로 내린 가중치 딕셔너리만 보냄
    cpu_weights = {k: v.cpu() for k, v in model.state_dict().items()}
    weights_ref = [ray.put(cpu_weights)]
    
    # 2. 리플레이 버퍼
    buffer = ReplayBuffer.remote()
    
    # 3. 일꾼 생성
    workers = [DataWorker.remote(buffer, weights_ref) for _ in range(NUM_ACTORS)]
    for w in workers: w.run.remote()
    
    # 4. 학습
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    step = 0
    
    print("Waiting for data generation...")
    
    while True:
        current_size = ray.get(buffer.size.remote())
        if current_size < BATCH_SIZE * 2:
            print(f"\rCollecting data... ({current_size}/{BATCH_SIZE*2})", end="")
            time.sleep(1)
            continue
            
        s_batch, pi_batch, z_batch = ray.get(buffer.sample.remote(BATCH_SIZE))
        
        s_tensor = torch.tensor(s_batch, dtype=torch.float32).to(DEVICE)
        pi_tensor = torch.tensor(pi_batch, dtype=torch.float32).to(DEVICE)
        z_tensor = torch.tensor(z_batch, dtype=torch.float32).to(DEVICE).unsqueeze(1)
        
        pred_pi, pred_v = model(s_tensor)
        
        loss_pi = -torch.mean(torch.sum(pi_tensor * pred_pi, dim=1))
        loss_v = F.mse_loss(pred_v, z_tensor)
        total_loss = loss_pi + loss_v
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        step += 1
        if step % 10 == 0:
            print(f"[Step {step}] Loss: {total_loss.item():.4f} (Pi: {loss_pi.item():.4f}, V: {loss_v.item():.4f}) Buffer: {current_size}")
            
        # [수정 중요] 업데이트할 때도 CPU로 내려서 공유
        if step % 50 == 0:
            cpu_weights = {k: v.cpu() for k, v in model.state_dict().items()}
            weights_ref[0] = ray.put(cpu_weights)
            
        if step % SAVE_INTERVAL == 0:
            if not os.path.exists("models"): os.makedirs("models")
            torch.save(model.state_dict(), f"models/model_{step}.pth")
            print(f"💾 Model Saved: models/model_{step}.pth")