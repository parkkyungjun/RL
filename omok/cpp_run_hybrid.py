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

# =============================================================================
# [1] 설정
# =============================================================================
BOARD_SIZE = 15
NUM_RES_BLOCKS = 5
NUM_CHANNELS = 64
BATCH_SIZE = 1024
INFERENCE_BATCH_SIZE = 2048
LR = 0.002
BUFFER_SIZE = 200000
NUM_ACTORS = 8
SAVE_INTERVAL = 2000
TARGET_SIMS = 400
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
    def __init__(self, buffer_ref, inference_server):
        seed = int(time.time() * 1000000) % (2**32)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        
        self.buffer_ref = buffer_ref
        self.inference_server = inference_server
        self.num_parallel_games = 64
        self.mcts_envs = [mcts_core.MCTS() for _ in range(self.num_parallel_games)]
        self.histories = [[] for _ in range(self.num_parallel_games)]
        self.sim_counts = [0] * self.num_parallel_games
        self.step_counts = [0] * self.num_parallel_games
        for mcts in self.mcts_envs:
            mcts.reset()
            mcts.add_root_noise(0.3, 0.25)

    # DataWorker 클래스 내부
    def get_seed(self):
        return np.random.get_state()[1][0] # 혹은 저장해둔 self.seed 반환

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

            for i in range(self.num_parallel_games):
                if self.sim_counts[i] >= TARGET_SIMS:
                    mcts = self.mcts_envs[i]
                    temp = 1.0 if self.step_counts[i] < 5 else 0.1
                    state, pi = mcts.get_action_probs(temp)
                    current_player = mcts.get_current_player()
                    self.histories[i].append([state, pi, current_player])
                    action = np.random.choice(len(pi), p=pi)
                    mcts.update_root_game(action)
                    self.step_counts[i] += 1
                    self.sim_counts[i] = 0
                    mcts.add_root_noise(0.3, 0.25)
                    
                    is_game_over, winner = mcts.check_game_status()
                    if is_game_over or self.step_counts[i] >= BOARD_SIZE * BOARD_SIZE:
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
                        self.step_counts[i] = 0
                        self.sim_counts[i] = 0

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
        self.total_added += len(history) # 누적 증가

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, pi, z = zip(*batch)
        return np.array(s), np.array(pi), np.array(z)

    def size(self): 
        return len(self.buffer)
    
    # [핵심] 누적 데이터 개수 반환
    def get_total_added(self):
        return self.total_added

import matplotlib
matplotlib.use('Agg') # [중요] GUI 없는 리눅스에서 에러 방지
import matplotlib.pyplot as plt
import numpy as np

def save_debug_files(step, s_batch, pi_batch, z_batch):
    """
    배치 데이터 중 첫 번째 샘플을 텍스트와 이미지로 저장합니다.
    """
    # 텐서 -> 넘파이 변환 (필요시)
    if hasattr(s_batch, 'cpu'): s_batch = s_batch.cpu().numpy()
    if hasattr(pi_batch, 'cpu'): pi_batch = pi_batch.cpu().numpy()
    if hasattr(z_batch, 'cpu'): z_batch = z_batch.cpu().numpy()

    # 첫 번째 샘플만 추출
    state = s_batch[0]   # (3, 15, 15)
    pi = pi_batch[0]     # (225,)
    z = z_batch[0]       # Scalar

    # ---------------------------------------------------------
    # [1] 텍스트 파일 저장 (debug_logs 폴더)
    # ---------------------------------------------------------
    if not os.path.exists("debug_logs"): os.makedirs("debug_logs")
    
    txt_path = f"debug_logs/step_{step}.txt"
    with open(txt_path, "w") as f:
        f.write(f"Step: {step}\n")
        f.write(f"Target Value (z): {float(z):.4f}  (1=Win, -1=Loss, 0=Draw)\n")
        f.write("-" * 30 + "\n")
        
        # 바둑판 그리기 (ASCII)
        my_stones = state[0]
        opp_stones = state[1]
        
        f.write("   " + " ".join([f"{i%10}" for i in range(15)]) + "\n")
        for r in range(15):
            row_str = f"{r:2d} "
            for c in range(15):
                if my_stones[r, c] == 1:
                    row_str += "O " # 내 돌 (Channel 0)
                elif opp_stones[r, c] == 1:
                    row_str += "X " # 상대 돌 (Channel 1)
                else:
                    row_str += ". "
            f.write(row_str + "\n")
        
        f.write("-" * 30 + "\n")
        f.write("Top 5 Policy Probabilities:\n")
        top_indices = np.argsort(pi)[::-1][:5]
        for idx in top_indices:
            r, c = divmod(idx, 15)
            f.write(f"  Pos({r},{c}): {pi[idx]:.4f}\n")

    # ---------------------------------------------------------
    # [2] 이미지 파일 저장 (채널별 시각화)
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 내 돌 (Channel 0)
    axes[0].imshow(state[0], cmap='Blues', vmin=0, vmax=1)
    axes[0].set_title(f"My Stones (Ch0)\nTurn info: {state[2][0][0]}")
    
    # 상대 돌 (Channel 1)
    axes[1].imshow(state[1], cmap='Reds', vmin=0, vmax=1)
    axes[1].set_title("Opponent Stones (Ch1)")
    
    # Policy 분포 (Heatmap)
    pi_grid = pi.reshape(15, 15)
    im = axes[2].imshow(pi_grid, cmap='viridis')
    axes[2].set_title(f"Policy Heatmap\nTarget z={float(z):.2f}")
    plt.colorbar(im, ax=axes[2])
    
    plt.savefig(f"debug_logs/step_{step}.png")
    plt.close(fig)
    
    print(f"🐛 [DEBUG] Saved log and image to debug_logs/step_{step}.*")

class DualLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding='utf-8') # "a"는 append 모드

    def write(self, message):
        self.terminal.write(message) # 터미널에 출력
        self.log.write(message)      # 파일에 출력
        self.log.flush()             # 즉시 파일에 쓰기 (버퍼링 방지)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

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
    workers = [DataWorker.remote(buffer, inference_server) for _ in range(NUM_ACTORS)]
    
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
    last_log_total_added = last_total_added
    last_log_step = step
    
    TARGET_REPLAY_RATIO = 10.0 
    MAX_STEPS_PER_CYCLE = 1000 

    loss_history = {'step': [], 'total': [], 'pi': [], 'v': []}
    while True:
        current_total_added = ray.get(buffer.get_total_added.remote())
        new_data_count = current_total_added - last_total_added
        
        if new_data_count < BATCH_SIZE:
            time.sleep(0.1)
            continue

        needed_steps = int((new_data_count / BATCH_SIZE) * TARGET_REPLAY_RATIO)
        steps_to_run = max(1, min(needed_steps, MAX_STEPS_PER_CYCLE))

        # =================================================================
        # 학습 루프 (Training Loop)
        # =================================================================
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
        
        if step % 50 == 0 or steps_to_run > 100:
            trained_since_last_log = step - last_log_step
            current_buffer_size = ray.get(buffer.size.remote())
            
            print(f"[Step {step}] Loss: {total_loss.item():.4f} | "
                  f"New Data: +{new_data_count} / Trained: {trained_since_last_log} steps | "
                  f"Buf: {current_buffer_size}")
            
            last_log_step = step
            
            loss_history['step'].append(step)
            loss_history['total'].append(total_loss.item())
            loss_history['pi'].append(loss_pi.item())
            loss_history['v'].append(loss_v.item())