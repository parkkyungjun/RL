import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ray
import time
import os
import asyncio
from collections import deque
import random
import matplotlib.pyplot as plt

# =============================================================================
# [1] 설정 (A6000 + 5950X 풀파워)
# =============================================================================
BOARD_SIZE = 15
NUM_RES_BLOCKS = 5      # 5블록 (오목 최적화)
NUM_CHANNELS = 64       # 64채널 (속도/성능 밸런스)
NUM_MCTS_SIMS = 800     # 생각 깊이 (국룰)
BATCH_SIZE = 512        # 학습 배치
INFERENCE_BATCH_SIZE = 64 
LR = 0.001
BUFFER_SIZE = 200000    # 버퍼 크기 좀 늘림
NUM_ACTORS = 30         # CPU 코어 수
SAVE_INTERVAL = 1000

TRAINER_DEVICE = torch.device("cuda:0") 
INFERENCE_DEVICE = torch.device("cuda:0") 

# =============================================================================
# [2] 게임 및 신경망
# =============================================================================
class Gomoku:
    def __init__(self):
        self.size = BOARD_SIZE
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.current_player = 1
        self.last_move = None
        self.move_count = 0
    def reset(self):
        self.board.fill(0); self.current_player = 1; self.last_move = None; self.move_count = 0
        return self.get_state()
    def step(self, action):
        x, y = action // self.size, action % self.size
        if self.board[x, y] != 0: return self.get_state(), -10, True
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
            state[0] = (self.board == 1); state[1] = (self.board == -1)
        else:
            state[0] = (self.board == -1); state[1] = (self.board == 1)
        state[2].fill(1.0)
        return state
    def get_legal_actions(self): return np.where(self.board.flatten() == 0)[0]
    def check_win(self, x, y):
        player = self.board[x, y]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 1
            for d in [1, -1]:
                nx, ny = x, y
                while True:
                    nx, ny = nx + dx*d, ny + dy*d
                    if 0 <= nx < self.size and 0 <= ny < self.size and self.board[nx, ny] == player: count += 1
                    else: break
            if count >= 5: return True, player
        if self.move_count == self.size * self.size: return True, 0
        return False, 0

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
# [3] Inference Server (GPU Batching)
# =============================================================================
@ray.remote(num_gpus=0.5)
class InferenceServer:
    def __init__(self):
        self.model = AlphaZeroNet().to(INFERENCE_DEVICE)
        self.model.eval()
        self.queue = asyncio.Queue()
        self.loop = asyncio.get_event_loop()
        self.loop.create_task(self.run_batch_inference())

    def update_weights(self, weights):
        self.model.load_state_dict(weights)

    async def predict(self, state_numpy):
        future = self.loop.create_future()
        await self.queue.put((state_numpy, future))
        return await future

    async def run_batch_inference(self):
        while True:
            batch_inputs = []
            futures = []
            item = await self.queue.get()
            batch_inputs.append(item[0])
            futures.append(item[1])
            
            while len(batch_inputs) < INFERENCE_BATCH_SIZE:
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout=0.0001)
                    batch_inputs.append(item[0])
                    futures.append(item[1])
                except asyncio.TimeoutError:
                    break
            
            if batch_inputs:
                with torch.no_grad():
                    states = torch.tensor(np.array(batch_inputs), dtype=torch.float32).to(INFERENCE_DEVICE)
                    pi_logits, values = self.model(states)
                    probs = torch.exp(pi_logits).cpu().numpy()
                    vals = values.cpu().numpy()
                
                for i, future in enumerate(futures):
                    if not future.done():
                        future.set_result((probs[i], vals[i]))

# =============================================================================
# [4] MCTS (With Dirichlet Noise)
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
        best_score = -float('inf'); best_action = -1; best_child = None
        for action, child in self.children.items():
            ucb = child.value() + 1.0 * child.prior * np.sqrt(self.visit_count) / (1 + child.visit_count)
            if ucb > best_score: best_score = ucb; best_action = action; best_child = child
        return best_action, best_child

class MCTS:
    def __init__(self, inference_server):
        self.server = inference_server

    def search(self, game, root):
        dirichlet_alpha = 0.3
        epsilon = 0.25
        
        # 첫 번째 확장을 위해 루트 칠드런 확인
        if not root.children:
             # 더미 호출로 초기화 (실제로는 아래 루프에서 처리됨)
             pass 

        for i in range(NUM_MCTS_SIMS):
            node = root
            scratch_game = game.__class__()
            scratch_game.board = game.board.copy()
            scratch_game.current_player = game.current_player
            search_path = [node]

            while node.children:
                action, node = node.select_child()
                scratch_game.step(action)
                search_path.append(node)

            state = scratch_game.get_state()
            policy, value = ray.get(self.server.predict.remote(state))
            value = value.item()

            legal_moves = scratch_game.get_legal_actions()
            
            # Root Node Noise
            if node == root:
                noise = np.random.dirichlet([dirichlet_alpha] * len(legal_moves))
                policy_mask = np.zeros(BOARD_SIZE*BOARD_SIZE)
                legal_policy = policy[legal_moves]
                mixed_policy = (1 - epsilon) * legal_policy + epsilon * noise
                policy_mask[legal_moves] = mixed_policy
                policy = policy_mask
            else:
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

# =============================================================================
# [5] Worker (Data Augmentation 추가됨!)
# =============================================================================
@ray.remote(num_cpus=1)
class DataWorker:
    def __init__(self, buffer_ref, inference_server):
        self.buffer_ref = buffer_ref
        self.game = Gomoku()
        self.mcts = MCTS(inference_server)

    # [핵심] 데이터 증강 함수 (Junxiao Song 코드 이식)
    def get_equi_data(self, history):
        extend_data = []
        # history: [(state, pi, z), ...]
        for state, pi, z in history:
            pi_board = pi.reshape(BOARD_SIZE, BOARD_SIZE)
            for i in [1, 2, 3, 4]:
                # 회전 (0, 90, 180, 270)
                equi_state = np.array([np.rot90(s, k=i) for s in state])
                equi_pi = np.rot90(pi_board, k=i)
                extend_data.append([equi_state, equi_pi.flatten(), z])
                
                # 대칭 (좌우 반전)
                equi_state_flip = np.array([np.fliplr(s) for s in equi_state])
                equi_pi_flip = np.fliplr(equi_pi)
                extend_data.append([equi_state_flip, equi_pi_flip.flatten(), z])
        return extend_data

    def run(self):
        while True:
            state = self.game.reset()
            root = Node(0)
            history = [] # (state, pi, z) 저장용
            
            while True:
                self.mcts.search(self.game, root)
                
                visits = np.array([child.visit_count for child in root.children.values()])
                actions = list(root.children.keys())
                
                if len(history) < 30: temp = 1.0
                else: temp = 0.1
                
                probs = visits ** (1/temp)
                probs = probs / np.sum(probs)
                action = np.random.choice(actions, p=probs)
                
                full_pi = np.zeros(BOARD_SIZE*BOARD_SIZE)
                full_pi[actions] = visits / np.sum(visits)
                
                # 승패(z)는 아직 모르니 0으로 저장
                history.append([state, full_pi, 0])
                
                state, _, done = self.game.step(action)
                root = root.children[action]
                
                if done:
                    winner_reward = 1
                    # 1. 보상 백프로파게이션
                    for i in reversed(range(len(history))):
                        history[i][2] = winner_reward
                        winner_reward = -winner_reward
                    
                    # 2. [여기!] 데이터 증강 (1판 -> 8판)
                    augmented_data = self.get_equi_data(history)
                    
                    # 3. 버퍼 전송
                    ray.get(self.buffer_ref.add.remote(augmented_data))
                    break

@ray.remote
class ReplayBuffer:
    def __init__(self): self.buffer = deque(maxlen=BUFFER_SIZE)
    def add(self, history): self.buffer.extend(history)
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, pi, z = zip(*batch)
        return np.array(s), np.array(pi), np.array(z)
    def size(self): return len(self.buffer)

# =============================================================================
# [6] 메인 실행 (그래프 저장 포함)
# =============================================================================
if __name__ == "__main__":
    if ray.is_initialized(): ray.shutdown()
    ray.init()
    
    print(f"🚀 AlphaZero FINAL (Augmentation Enabled) Started!")
    
    inference_server = InferenceServer.remote()
    buffer = ReplayBuffer.remote()
    workers = [DataWorker.remote(buffer, inference_server) for _ in range(NUM_ACTORS)]
    for w in workers: w.run.remote()
    
    model = AlphaZeroNet().to(TRAINER_DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    step = 0
    loss_history = {'step': [], 'total': [], 'pi': [], 'v': []}
    
    print("Waiting for data generation...")
    last_train_size = 0
    
    while True:
        current_size = ray.get(buffer.size.remote())
        
        if current_size < BATCH_SIZE:
            print(f"\rWarm-up... ({current_size}/{BATCH_SIZE})", end="")
            time.sleep(2)
            continue
            
        # 데이터가 500개 이상 쌓여야 학습 (증강 덕분에 500개 금방 참)
        if current_size - last_train_size < 500:
            print(f"\rWaiting for fresh data... ({current_size})", end="")
            time.sleep(1)
            continue
            
        s_batch, pi_batch, z_batch = ray.get(buffer.sample.remote(BATCH_SIZE))
        
        s_tensor = torch.tensor(s_batch, dtype=torch.float32).to(TRAINER_DEVICE)
        pi_tensor = torch.tensor(pi_batch, dtype=torch.float32).to(TRAINER_DEVICE)
        z_tensor = torch.tensor(z_batch, dtype=torch.float32).to(TRAINER_DEVICE).unsqueeze(1)
        
        pred_pi, pred_v = model(s_tensor)
        loss_pi = -torch.mean(torch.sum(pi_tensor * pred_pi, dim=1))
        loss_v = F.mse_loss(pred_v, z_tensor)
        total_loss = loss_pi + loss_v
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        last_train_size = current_size
        step += 1
        
        if step % 10 == 0:
            print(f"[Step {step}] Loss: {total_loss.item():.4f} (Pi: {loss_pi.item():.4f}, V: {loss_v.item():.4f}) Buffer: {current_size}")
            loss_history['step'].append(step)
            loss_history['total'].append(total_loss.item())
            loss_history['pi'].append(loss_pi.item())
            loss_history['v'].append(loss_v.item())

        if step % 50 == 0:
            cpu_weights = {k: v.cpu() for k, v in model.state_dict().items()}
            inference_server.update_weights.remote(cpu_weights)
            
        if step % 100 == 0:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(loss_history['step'], loss_history['total'], label='Total')
            plt.plot(loss_history['step'], loss_history['pi'], label='Policy')
            plt.legend(); plt.grid(True)
            plt.subplot(1, 2, 2)
            plt.plot(loss_history['step'], loss_history['v'], label='Value', color='orange')
            plt.legend(); plt.grid(True)
            plt.savefig('training_loss.png')
            plt.close()

        if step % SAVE_INTERVAL == 0:
            if not os.path.exists("models"): os.makedirs("models")
            torch.save(model.state_dict(), f"models/model_{step}.pth")
            print(f"💾 Model Saved: models/model_{step}.pth")