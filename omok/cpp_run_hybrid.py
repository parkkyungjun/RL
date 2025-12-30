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
BOARD_SIZE = 15
NUM_RES_BLOCKS = 8
NUM_CHANNELS = 128
BATCH_SIZE = 1024
INFERENCE_BATCH_SIZE = 512
LR = 0.001
BUFFER_SIZE = 150000
NUM_ACTORS = 8
SAVE_INTERVAL = 500
TARGET_SIMS = 1600
RESUME_CHECKPOINT = "models/checkpoint_20000.pth"  # None 이면 새로 시작
NUM_PARALLEL_GAMES = 16

TRAINER_DEVICE = torch.device("cuda:0") 
INFERENCE_DEVICE = torch.device("cuda:0") 

# =============================================================================
# [2] 신경망 (기존 동일)
# =============================================================================
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # [설정 추천] 64채널이면 groups=8 정도가 적당함 (그룹당 8채널)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.GroupNorm(num_groups=8, num_channels=channels)
        
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.GroupNorm(num_groups=8, num_channels=channels)

    def forward(self, x):
        res = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + res)
    
class AlphaZeroNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.start_conv = nn.Conv2d(3, NUM_CHANNELS, 3, padding=1)
        # [수정] bn_start도 GroupNorm으로 교체 (채널 64, 그룹 8)
        self.bn_start = nn.GroupNorm(num_groups=8, num_channels=NUM_CHANNELS)
        
        self.backbone = nn.Sequential(*[ResBlock(NUM_CHANNELS) for _ in range(NUM_RES_BLOCKS)])
        
        self.policy_head = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, 2, 1), 
            # 채널이 2개뿐이므로 그룹은 1개 또는 2개만 가능. 1개 추천(LayerNorm 효과)
            nn.GroupNorm(num_groups=1, num_channels=2), 
            nn.ReLU(),
            nn.Flatten(), 
            nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)
        )
        
        self.value_head = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, 1, 1), 
            # [버그 수정] 채널이 1개이므로 num_channels=1 이어야 함!
            nn.GroupNorm(num_groups=1, num_channels=1), 
            nn.ReLU(),
            nn.Flatten(), 
            nn.Linear(BOARD_SIZE * BOARD_SIZE, 64), 
            nn.ReLU(),
            nn.Linear(64, 1), 
            nn.Tanh()
        )
        
    def forward(self, x):
        x = F.relu(self.bn_start(self.start_conv(x)))
        x = self.backbone(x)
        policy = self.policy_head(x)
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
                    # probs = torch.exp(pi_logits).cpu().numpy()
                    probs = F.softmax(pi_logits, dim=1).cpu().numpy()
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
        self.num_parallel_games = NUM_PARALLEL_GAMES
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
    
    def save_crash_dump(self, game_idx, pi, error_msg):
        """에러 발생 시점의 데이터를 pickle 파일로 저장"""
        import pickle
        import datetime
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"crash_worker_{self.worker_id}_game_{game_idx}_{timestamp}.pkl"
        
        crash_data = {
            "worker_id": self.worker_id,
            "game_idx": game_idx,
            "error_message": str(error_msg),
            "pi": pi,  # 문제가 된 확률 벡터
            "history": self.histories[game_idx],  # 현재까지 둔 수순
            "step_count": self.step_counts[game_idx],
            "network_weights_sample": "Need to check trainer/inference server for weights"
        }
        
        with open(filename, "wb") as f:
            pickle.dump(crash_data, f)
        
        print(f"\n🔥 [CRASH SAVED] Worker {self.worker_id} saved crash dump to {filename}")
        
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
                    temp = 1.0 # if self.step_counts[i] < 30 else 0.1 # 0.1로 하면 10승이 되어서 발산 가능성이 있음
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
                            
                    # Action 결정
                    if triggered_random:
                        # --- 수정된 로직: Top-K Soft Sampling ---
                        # pi는 방문 횟수에 기반한 확률이므로, 이미 '수의 질'이 반영되어 있습니다.
                        # 0보다 큰 확률을 가진 인덱스 추출
                        valid_indices = np.where(pi > 0)[0]
                        
                        if len(valid_indices) > 1:
                            # 1. 확률 내림차순으로 정렬
                            sorted_indices = valid_indices[np.argsort(pi[valid_indices])[::-1]]
                            
                            # 2. 후보군 선정 (예: 1등은 제외하고, 2등~5등 사이에서 선택)
                            # 만약 유효한 수가 적다면 가능한 만큼만 슬라이싱
                            # top_k 범위는 조정 가능 (여기서는 상위 2~5번째 수)
                            candidates = sorted_indices[1:min(len(sorted_indices), 6)]
                            
                            # 3. 후보군 내에서 다시 확률 분포 계산 (Renormalize)
                            candidate_probs = pi[candidates]
                            candidate_probs_sum = candidate_probs.sum()
                            
                            if candidate_probs_sum > 0:
                                candidate_probs /= candidate_probs_sum
                                action = np.random.choice(candidates, p=candidate_probs)
                            else:
                                # 후보군 확률 합이 0이면(거의 희박하지만) 그냥 2등 수 선택
                                action = candidates[0]
                        else:
                            # 둘 수 있는 곳이 1곳 뿐이라면 선택의 여지가 없음
                            action = np.argmax(pi)
                            
                    elif self.step_counts[i] < 30:
                        # 30수 미만: 확률적 선택 (Temperature = 1.0 효과)
                        try:
                            action = np.random.choice(len(pi), p=pi)
                        except ValueError:
                            # 만약 여기서도 에러나면 안전하게 argmax
                            action = np.argmax(pi)
                    else:
                        # 30수 이상: 가장 많이 방문한 수 선택 (Temperature -> 0 효과)
                        # temp=0.1을 쓰는 대신 그냥 argmax를 쓰면 수학적으로 동일하고 안전함
                        action = np.argmax(pi)
                    
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
                        
                        if np.random.rand() < 0.5:
                            self.has_played_random[i] = False 
                        else:
                            self.has_played_random[i] = True

# =============================================================================
# [5] 학습 루프 (Main) - AMP 제거 버전
# =============================================================================
@ray.remote
class ReplayBuffer:
    def __init__(self): 
        self.buffer = deque(maxlen=BUFFER_SIZE)
        self.total_added = 0 

    def add(self, history): 
        self.buffer.extend(history)
        self.total_added += len(history)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, pi, z = zip(*batch)
        return np.array(s), np.array(pi), np.array(z)

    def size(self): 
        return len(self.buffer)
    
    def get_total_added(self):
        return self.total_added

if __name__ == "__main__":
    if not os.path.exists("logs"):
        os.makedirs("logs")

    sys.stdout = DualLogger("logs/training.log")

    if ray.is_initialized(): ray.shutdown()
    ray.init()

    print(f"🚀 AlphaZero HYBRID Started! (AMP Disabled - Float32 Mode)")

    inference_server = InferenceServer.remote()
    buffer = ReplayBuffer.remote()
    
    model = AlphaZeroNet().to(TRAINER_DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    # [제거됨] scaler = torch.amp.GradScaler('cuda') 
    
    step = 0

    # =================================================================
    # [LOGIC] Resume vs Fresh Start
    # =================================================================
    if RESUME_CHECKPOINT and os.path.exists(RESUME_CHECKPOINT):
        print(f"🔄 Loading checkpoint from {RESUME_CHECKPOINT}...")
        checkpoint = torch.load(RESUME_CHECKPOINT, map_location=TRAINER_DEVICE)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # [제거됨] scaler 로드 로직 삭제
            
        step = checkpoint['step']
        
        cpu_weights = {k: v.cpu() for k, v in model.state_dict().items()}
        ray.get(inference_server.update_weights.remote(cpu_weights))
        
        print(f"✅ Resumed successfully from Step {step}")

    # -------------------------------------------------------------
    # [STEP 2] Main Loop
    # -------------------------------------------------------------
    print("🚀 Starting MCTS Workers...")
    workers = [DataWorker.remote(buffer, inference_server, i) for i in range(NUM_ACTORS)]

    print("⏳ Waiting for all workers to initialize...")
    seeds = ray.get([w.get_seed.remote() for w in workers])
    print(f"✅ All Workers Ready! Seeds: {seeds}")
    
    for w in workers: w.run.remote()
    
    print("🚀 Starting Adaptive Main Training Loop...")
    
    last_total_added = ray.get(buffer.get_total_added.remote())
    
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
            
            # [수정] autocast 구문 제거 (Float32 연산)
            # with torch.amp.autocast('cuda'):  <-- 삭제
            pred_pi, pred_v = model(s_tensor)
            loss_pi = -torch.mean(torch.sum(pi_tensor * F.log_softmax(pred_pi, dim=1), dim=1))
            loss_v = F.mse_loss(pred_v, z_tensor)
            total_loss = loss_pi + loss_v
            
            # [수정] Scaler 없이 일반적인 역전파 수행
            # scaler.scale(total_loss).backward() <-- 삭제
            total_loss.backward()

            # [수정] Scaler 없이 일반적인 Optimizer Step
            # scaler.unscale_(optimizer) <-- 삭제
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # scaler.update() <-- 삭제
            
            step += 1

            if step == 1 or step % 1000 == 0:
                save_debug_files(step, s_tensor, pi_tensor, z_tensor)

            if step % 50 == 0:
                cpu_weights = {k: v.cpu() for k, v in model.state_dict().items()}
                inference_server.update_weights.remote(cpu_weights)
                
            if step % SAVE_INTERVAL == 0:
                if not os.path.exists("models"): os.makedirs("models")
                
                checkpoint = {
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # 'scaler_state_dict': scaler.state_dict() <-- 삭제
                }
                save_path = f"models/checkpoint_{step}.pth"
                torch.save(checkpoint, save_path)
                print(f"\n💾 Checkpoint Saved: {save_path}")
                
                # 그래프 그리기 코드는 동일...
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

        last_total_added = current_total_added 
        current_buffer_size = ray.get(buffer.size.remote())

        print(f"[Step {step}] Loss: {total_loss.item():.4f} | "
                f"New Data: +{new_data_count} / Trained: {steps_to_run} steps | "
                f"Buf: {current_buffer_size} / ⏱️ Training Cycle Completed in {time.time() - T:.2f}s")
        
        loss_history['step'].append(step)
        loss_history['total'].append(total_loss.item())
        loss_history['pi'].append(loss_pi.item())
        loss_history['v'].append(loss_v.item())