# cpp_run_hybrid.py
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
import sys
# C++ 모듈 (필요시 주석 처리하고 테스트 가능, 여기선 MCTS Worker에서 사용)
import mcts_core 
from logging_ import *

# =============================================================================
# [1] 설정
# =============================================================================
BOARD_SIZE = 8
NUM_RES_BLOCKS = 5
NUM_CHANNELS = 64
BATCH_SIZE = 1024
INFERENCE_BATCH_SIZE = 2048
LR = 0.002
BUFFER_SIZE = 20000
NUM_ACTORS = 8
SAVE_INTERVAL = 500
TARGET_SIMS = 800
RESUME_CHECKPOINT = None

TRAINER_DEVICE = torch.device("cuda:0") 
INFERENCE_DEVICE = torch.device("cuda:0") 

# =============================================================================
# [2] 신경망 (기존 동일)
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
# [3] Inference Server (기존 동일)
# =============================================================================
@ray.remote(num_gpus=0.4)
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
            
            deadline = self.loop.time() + 0.003
            current_batch_size = item[0].shape[0]
            
            while True:
                timeout = deadline - self.loop.time()
                if timeout <= 0 or current_batch_size >= INFERENCE_BATCH_SIZE: break
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                    batch_inputs.append(item[0])
                    futures.append(item[1])
                    current_batch_size += item[0].shape[0]
                except asyncio.TimeoutError:
                    break
            
            if batch_inputs:
                with torch.no_grad():
                    states_np = np.concatenate(batch_inputs, axis=0)
                    states = torch.tensor(states_np, dtype=torch.float32).to(INFERENCE_DEVICE)
                    pi_logits, values = self.model(states)
                    probs = torch.exp(pi_logits).cpu().numpy()
                    vals = values.cpu().numpy()
                
                cursor = 0
                for i, future in enumerate(futures):
                    if not future.done():
                        num_samples = batch_inputs[i].shape[0]
                        future.set_result((probs[cursor : cursor + num_samples], vals[cursor : cursor + num_samples]))
                        cursor += num_samples

# =============================================================================
# [4] Data Worker (MCTS - 기존과 동일)
# =============================================================================
@ray.remote(num_cpus=1)
class DataWorker:
    def __init__(self, buffer_ref, inference_server, worker_id):
        seed = int(time.time() * 1000000) % (2**32)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        
        self.worker_id = worker_id
        self.buffer_ref = buffer_ref
        self.inference_server = inference_server
        self.num_parallel_games = 2
        self.mcts_envs = [mcts_core.MCTS() for _ in range(self.num_parallel_games)]
        self.histories = [[] for _ in range(self.num_parallel_games)]
        self.sim_counts = [0] * self.num_parallel_games
        self.step_counts = [0] * self.num_parallel_games
        
        self.action_logs = [[] for _ in range(self.num_parallel_games)]
        self.game_counters = [0] * self.num_parallel_games
        
        # [변경] 이 게임에서 랜덤 수를 이미 뒀는지 체크하는 플래그
        self.has_played_random = [
            False if np.random.rand() < 0.5 else True 
            for _ in range(self.num_parallel_games)
        ]

        for mcts in self.mcts_envs:
            mcts.reset()
            mcts.add_root_noise(0.3, 0.25)

    def get_seed(self):
        return np.random.get_state()[1][0]

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
            # --- MCTS Simulation Phase (변경 없음) ---
            states_to_infer = []
            indices_to_infer = []
            for i in range(self.num_parallel_games):
                if self.sim_counts[i] < TARGET_SIMS:
                    leaf = self.mcts_envs[i].select_leaf()
                    if leaf is not None:
                        states_to_infer.append(leaf)
                        indices_to_infer.append(i)
                    else:
                        self.sim_counts[i] += 1
            
            if states_to_infer:
                states_np = np.stack(states_to_infer)
                policy_batch, value_batch = ray.get(self.inference_server.predict.remote(states_np))
                for idx, policy, value in zip(indices_to_infer, policy_batch, value_batch):
                    self.mcts_envs[idx].backpropagate(policy, value.item())
                    self.sim_counts[idx] += 1

            # --- Action Phase ---
            for i in range(self.num_parallel_games):
                if self.sim_counts[i] >= TARGET_SIMS:
                    mcts = self.mcts_envs[i]
                    temp = 1.0 if self.step_counts[i] < 30 else 0.1
                    state, pi = mcts.get_action_probs(temp)
                    current_player = mcts.get_current_player()
                    self.histories[i].append([state, pi, current_player])
                    
                    # [핵심 변경 로직]
                    # 1. 흑 차례인가? (보통 0부터 시작하므로 짝수 턴이 흑)
                    # 2. 이 게임에서 아직 랜덤 수를 안 뒀는가?
                    # 3. 10% 확률에 당첨되었는가?
                    is_black_turn = (self.step_counts[i] % 2 == 0)
                    triggered_random = False
                    
                    if is_black_turn and not self.has_played_random[i]:
                        if np.random.rand() < 0.1: # 10% 확률
                            triggered_random = True
                            self.has_played_random[i] = True # "사용함" 처리 (재사용 방지)
                    
                    if triggered_random:
                        # MCTS 정책(pi)에서 둘 수 있는 곳(0보다 큰 곳)만 필터링 -> 기존 돌 제외됨
                        valid_moves = np.where(pi > 0)[0]
                        if len(valid_moves) > 0:
                            action = np.random.choice(valid_moves)
                        else:
                            action = np.argmax(pi) # 예외 처리
                    else:
                        # 평소대로 확률 분포에 따라 선택
                        action = np.random.choice(len(pi), p=pi)
                    
                    self.action_logs[i].append(action)

                    mcts.update_root_game(action)
                    self.step_counts[i] += 1
                    self.sim_counts[i] = 0
                    mcts.add_root_noise(0.3, 0.25)
                    
                    is_game_over, winner = mcts.check_game_status()
                    
                    if is_game_over or self.step_counts[i] >= BOARD_SIZE * BOARD_SIZE:
                        save_game_log(self.worker_id, self.game_counters[i], self.action_logs[i], winner, BOARD_SIZE)
                        self.game_counters[i] += 1

                        processed_history = []
                        for h_state, h_pi, h_player in self.histories[i]:
                            if winner == 0: z = 0.0
                            elif h_player == winner: z = 1.0
                            else: z = -1.0
                            processed_history.append([h_state, h_pi, z])
                        
                        augmented = self.get_equi_data(processed_history)
                        self.buffer_ref.add.remote(augmented)
                        
                        mcts.reset()
                        mcts.add_root_noise(0.3, 0.25)
                        self.histories[i] = []
                        self.action_logs[i] = []
                        self.step_counts[i] = 0
                        self.sim_counts[i] = 0
                        
                        # [초기화] 새 게임이 시작되므로 플래그를 False로 리셋
                        self.has_played_random[i] = False
# =============================================================================
# [5] 학습 루프 (Main)
# =============================================================================
# [수정 1] ReplayBuffer에 카운터 기능 추가 (이 클래스로 교체하세요)
@ray.remote
class ReplayBuffer:
    def __init__(self): 
        self.buffer = deque(maxlen=BUFFER_SIZE)
        self.total_added = 0  # 누적 카운터

    def add(self, history): 
        self.buffer.extend(history)
        self.total_added += len(history) # 누적 증가``

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, pi, z = zip(*batch)
        return np.array(s), np.array(pi), np.array(z)

    def size(self): 
        return len(self.buffer)
    
    # [핵심] 누적 데이터 개수 반환
    def get_total_added(self):
        return self.total_added

if __name__ == "__main__":
    if not os.path.exists("logs"):
        os.makedirs("logs")
        
    sys.stdout = DualLogger("logs/training.log")

    if ray.is_initialized(): ray.shutdown()
    ray.init()

    print(f"🚀 AlphaZero HYBRID Started!")

    inference_server = InferenceServer.remote()
    buffer = ReplayBuffer.remote()
    
    model = AlphaZeroNet().to(TRAINER_DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    # [AMP 수정 1] GradScaler도 최신 문법으로 변경
    scaler = torch.amp.GradScaler('cuda')
    
    step = 0

    # =================================================================
    # [LOGIC] Resume vs Fresh Start
    # =================================================================
    if RESUME_CHECKPOINT and os.path.exists(RESUME_CHECKPOINT):
        print(f"🔄 Loading checkpoint from {RESUME_CHECKPOINT}...")
        checkpoint = torch.load(RESUME_CHECKPOINT, map_location=TRAINER_DEVICE)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        step = checkpoint['step']
        
        cpu_weights = {k: v.cpu() for k, v in model.state_dict().items()}
        ray.get(inference_server.update_weights.remote(cpu_weights))
        
        print(f"✅ Resumed successfully from Step {step}")
        print("⏩ Skipping Synthetic Data Generation (Already trained model).")

    # -------------------------------------------------------------
    # [STEP 2] Main Loop
    # -------------------------------------------------------------
    print("🚀 Starting MCTS Workers...")
    # workers = [DataWorker.remote(buffer, inference_server) for _ in range(NUM_ACTORS)]
    
    workers = [DataWorker.remote(buffer, inference_server, i) for i in range(NUM_ACTORS)]

    # [추가] 모든 워커가 초기화될 때까지 여기서 딱 대기함 (Block)
    print("⏳ Waiting for all workers to initialize...")
    
    # 워커들의 get_seed 함수를 호출해서 결과를 다 받을 때까지 메인 프로세스를 멈춤
    # 여기서 에러가 나거나 멈춰있으면 CPU 부족임
    seeds = ray.get([w.get_seed.remote() for w in workers])
    
    print(f"✅ All Workers Ready! Seeds: {seeds}")
    print(f"👉 Unique seeds count: {len(set(seeds))} / {NUM_ACTORS}") # 이게 10개여야 함
    
    for w in workers: w.run.remote()
    
    print("🚀 Starting Adaptive Main Training Loop...")
    
    last_total_added = ray.get(buffer.get_total_added.remote())
    last_log_step = step
    
    TARGET_REPLAY_RATIO = 8.0 
    MAX_STEPS_PER_CYCLE = 1000 

    loss_history = {'step': [], 'total': [], 'pi': [], 'v': []}

    while True:
        current_total_added = ray.get(buffer.get_total_added.remote())
        new_data_count = current_total_added - last_total_added

        if new_data_count < BATCH_SIZE:
            time.sleep(.1)
            continue
        
        needed_steps = int((new_data_count / BATCH_SIZE) * TARGET_REPLAY_RATIO)
        steps_to_run = max(1, min(needed_steps, MAX_STEPS_PER_CYCLE))

        # =================================================================
        # 학습 루프 (Training Loop)
        # =================================================================
        T = time.time()
        for _ in range(steps_to_run):
            s_batch, pi_batch, z_batch = ray.get(buffer.sample.remote(BATCH_SIZE))
            
            s_tensor = torch.tensor(s_batch, dtype=torch.float32).to(TRAINER_DEVICE)
            pi_tensor = torch.tensor(pi_batch, dtype=torch.float32).to(TRAINER_DEVICE)
            z_tensor = torch.tensor(z_batch, dtype=torch.float32).to(TRAINER_DEVICE).unsqueeze(1)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                pred_pi, pred_v = model(s_tensor)
                loss_pi = -torch.mean(torch.sum(pi_tensor * pred_pi, dim=1))
                loss_v = F.mse_loss(pred_v, z_tensor)
                total_loss = loss_pi + loss_v
            
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            step += 1

            if step == 1 or step % 1000 == 0:
                save_debug_files(step, s_tensor, pi_tensor, z_tensor)

            # [가중치 업데이트]
            if step % 50 == 0:
                cpu_weights = {k: v.cpu() for k, v in model.state_dict().items()}
                inference_server.update_weights.remote(cpu_weights)

            # [핵심 수정] 저장 체크를 for문 안에서 매 스텝마다 해야 함!
            if step % SAVE_INTERVAL == 0:
                if not os.path.exists("models"): os.makedirs("models")
                
                checkpoint = {
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict()
                }
                save_path = f"models/checkpoint_{step}.pth"
                torch.save(checkpoint, save_path)
                print(f"\n💾 Checkpoint Saved: {save_path}")
                
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
        # =================================================================
        # 루프 종료 후 동기화 및 로그 출력
        # =================================================================
        last_total_added = current_total_added # 포인터 최신화

        current_buffer_size = ray.get(buffer.size.remote())

        print(f"[Step {step}] Loss: {total_loss.item():.4f} | "
                f"New Data: +{new_data_count} / Trained: {steps_to_run} steps | "
                f"Buf: {current_buffer_size} / ⏱️ Training Cycle Completed in {time.time() - T:.2f}s / Total Adding {current_total_added}")
        
        last_log_step = step
        
        loss_history['step'].append(step)
        loss_history['total'].append(total_loss.item())
        loss_history['pi'].append(loss_pi.item())
        loss_history['v'].append(loss_v.item())