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

# C++로 빌드한 모듈 임포트
import mcts_core 

# =============================================================================
# [1] 설정
# =============================================================================
BOARD_SIZE = 15
NUM_RES_BLOCKS = 5      
NUM_CHANNELS = 64       
BATCH_SIZE = 512        
INFERENCE_BATCH_SIZE = 256  # GPU 활용을 위해 배치 키움
LR = 0.001
BUFFER_SIZE = 200000    
NUM_ACTORS = 20         # 5950X 코어 수 고려 (컨텍스트 스위칭 방지)
SAVE_INTERVAL = 20000

TRAINER_DEVICE = torch.device("cuda:0") 
INFERENCE_DEVICE = torch.device("cuda:0") 

# =============================================================================
# [2] 신경망 (변경 없음)
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
# [3] Inference Server
# =============================================================================
@ray.remote(num_gpus=0.4) # GPU 메모리 점유율 조정
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
            
            # 첫 번째 요청 대기
            item = await self.queue.get()
            batch_inputs.append(item[0])
            futures.append(item[1])
            
            # 배치 모으기 (최대 3ms 대기)
            deadline = self.loop.time() + 0.003
            while len(batch_inputs) < INFERENCE_BATCH_SIZE:
                timeout = deadline - self.loop.time()
                if timeout <= 0: break
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                    batch_inputs.append(item[0])
                    futures.append(item[1])
                except asyncio.TimeoutError:
                    break
            
            if batch_inputs:
                with torch.no_grad():
                    # C++에서 넘어온 numpy array를 텐서로 변환
                    states = torch.tensor(np.array(batch_inputs), dtype=torch.float32).to(INFERENCE_DEVICE)
                    
                    # 추론
                    pi_logits, values = self.model(states)
                    
                    # Policy: LogSoftmax -> Exp(Prob) 변환해서 C++로 전달
                    probs = torch.exp(pi_logits).cpu().numpy()
                    vals = values.cpu().numpy()
                
                for i, future in enumerate(futures):
                    if not future.done():
                        future.set_result((probs[i], vals[i]))

# =============================================================================
# [4] Worker (C++ MCTS 탑재)
# =============================================================================
@ray.remote(num_cpus=1)
class DataWorker:
    def __init__(self, buffer_ref, inference_server):
        self.buffer_ref = buffer_ref
        self.inference_server = inference_server
        # 여기서 C++ 모듈의 MCTS 클래스 생성
        self.mcts = mcts_core.MCTS()

    def get_equi_data(self, history):
        extend_data = []
        for state, pi, z in history:
            pi_board = pi.reshape(BOARD_SIZE, BOARD_SIZE)
            for i in [1, 2, 3, 4]:
                equi_state = np.array([np.rot90(s, k=i) for s in state])
                equi_pi = np.rot90(pi_board, k=i)
                extend_data.append([equi_state, equi_pi.flatten(), z])
                equi_state_flip = np.array([np.fliplr(s) for s in equi_state])
                equi_pi_flip = np.fliplr(equi_pi)
                extend_data.append([equi_state_flip, equi_pi_flip.flatten(), z])
        return extend_data

    def run(self):
        while True:
            self.mcts.reset()
            history = [] # [(state, pi, player_turn), ...]
            step_count = 0
            
            while True:
                # 1. 시뮬레이션 (800회) - C++ 내부에서 트리 관리
                self.mcts.add_root_noise(0.3, 0.25)
                
                for _ in range(800): 
                    # A. C++이 Leaf 노드 선택
                    leaf_state = self.mcts.select_leaf()
                    
                    # None이면 해당 시뮬레이션 경로는 이미 게임이 끝난 노드
                    if leaf_state is None:
                        continue
                        
                    # B. Python이 GPU 추론
                    policy, value = ray.get(self.inference_server.predict.remote(leaf_state))
                    
                    # C. C++로 결과 주입
                    self.mcts.backpropagate(policy, value.item())
                
                # 2. 행동 선택
                temp = 1.0 if step_count < 30 else 0.1
                state, pi = self.mcts.get_action_probs(temp)
                
                # 현재 누구 턴이었는지 저장 (나중에 z값 계산용)
                current_player = self.mcts.get_current_player()
                history.append([state, pi, current_player])
                
                # 실제 수 두기 & 루트 이동
                action = np.random.choice(len(pi), p=pi)
                
                # update_root_game이 True를 반환하면 방금 둔 수로 게임이 끝났다는 뜻
                self.mcts.update_root_game(action)
                is_game_over, winner = self.mcts.check_game_status()
                step_count += 1
                
                if is_game_over or step_count >= BOARD_SIZE * BOARD_SIZE:
                    if step_count >= BOARD_SIZE * BOARD_SIZE:
                        winner = -1
                    
                    # z값(승패) 할당
                    # history에 저장된 current_player와 winner가 같으면 +1, 다르면 -1
                    processed_history = []
                    for h_state, h_pi, h_player in history:
                        if winner == 0:
                            z = 0
                        elif h_player == winner:
                            z = 1
                        else:
                            z = -1
                        processed_history.append([h_state, h_pi, z])
                    
                    # 데이터 증강 (8배)
                    augmented_data = self.get_equi_data(processed_history)
                    ray.get(self.buffer_ref.add.remote(augmented_data))
                    break

# =============================================================================
# [5] 학습 루프 (Main)
# =============================================================================
@ray.remote
class ReplayBuffer:
    def __init__(self): self.buffer = deque(maxlen=BUFFER_SIZE)
    def add(self, history): self.buffer.extend(history)
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, pi, z = zip(*batch)
        return np.array(s), np.array(pi), np.array(z)
    def size(self): return len(self.buffer)

if __name__ == "__main__":
    if ray.is_initialized(): ray.shutdown()
    ray.init()
    
    print(f"🚀 AlphaZero HYBRID (C++ Engine + Python GPU) Started!")
    print(f"run: watch -n 0.1 nvidia-smi (to check usage)")

    inference_server = InferenceServer.remote()
    buffer = ReplayBuffer.remote()
    
    # Worker 시작
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
        
        if current_size < BATCH_SIZE * 2:
            print(f"\rWarm-up... ({current_size}/{BATCH_SIZE*2})", end="")
            time.sleep(1)
            continue
            
        # 데이터 수집 속도가 빨라졌으므로 더 자주 학습
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
        
        step += 1
        
        if step % 10 == 0:
            print(f"[Step {step}] Loss: {total_loss.item():.4f} (Pi: {loss_pi.item():.4f}, V: {loss_v.item():.4f}) Buffer: {current_size}")
            loss_history['step'].append(step)
            loss_history['total'].append(total_loss.item())
            loss_history['pi'].append(loss_pi.item())
            loss_history['v'].append(loss_v.item())

        # 최신 가중치 전송 (50 스텝마다)
        if step % 50 == 0:
            cpu_weights = {k: v.cpu() for k, v in model.state_dict().items()}
            inference_server.update_weights.remote(cpu_weights)
            
        if step % SAVE_INTERVAL == 0:
            if not os.path.exists("models"): os.makedirs("models")
            torch.save(model.state_dict(), f"models/model_{step}.pth")
            # 그래프 저장
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
            print(f"💾 Model Saved & Graph Updated")