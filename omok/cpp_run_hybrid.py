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

# C++ 모듈 (필요시 주석 처리하고 테스트 가능, 여기선 MCTS Worker에서 사용)
import mcts_core 

# =============================================================================
# [1] 설정
# =============================================================================
BOARD_SIZE = 15
NUM_RES_BLOCKS = 5      
NUM_CHANNELS = 64       
BATCH_SIZE = 2048        
INFERENCE_BATCH_SIZE = 2048
LR = 0.001
BUFFER_SIZE = 1000000    
NUM_ACTORS = 10         
SAVE_INTERVAL = 20000
TARGET_SIMS = 200
RESUME_CHECKPOINT = 'models/checkpoint_20000.pth' 

# 합성 데이터 설정 (사용자 아이디어)
SYNTHETIC_GAMES = 2000  # 초기에 주입할 판 수 (흑승 1000 + 백승 1000)

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
# [NEW] Rule-Based Worker (합성 데이터 생성기)
# =============================================================================
@ray.remote(num_cpus=1)
class RuleBasedWorker:
    def __init__(self, buffer_ref):
        self.buffer_ref = buffer_ref
        self.board_size = BOARD_SIZE

    def check_win(self, board, player):
        # 간단한 파이썬 승리 판별 (가로, 세로, 대각선)
        # board: (15, 15), player: 1 or -1
        # (최적화보다는 가독성 위주 구현)
        for x in range(self.board_size):
            for y in range(self.board_size):
                if board[x, y] == player:
                    # 가로, 세로, 대각선 2방향
                    directions = [(0,1), (1,0), (1,1), (1,-1)]
                    for dx, dy in directions:
                        count = 1
                        for i in range(1, 5):
                            nx, ny = x + dx*i, y + dy*i
                            if 0 <= nx < self.board_size and 0 <= ny < self.board_size and board[nx, ny] == player:
                                count += 1
                            else:
                                break
                        if count == 5: return True
        return False

    def get_attacker_move(self, board, player):
        # 1. 5목이 되는 수가 있으면 무조건 둠 (승리)
        # 2. 4목이 되는 수가 있으면 둠
        # 3. 내 돌 주변(1칸 범위)에 둠
        # 4. 없으면 랜덤
        candidates = []
        my_stones = list(zip(*np.where(board == player)))
        empty_spots = list(zip(*np.where(board == 0)))
        
        if not empty_spots: return None

        # [Logic 1 & 2] 중요 위치 탐색 (단순화: 4->5, 3->4)
        # 파이썬으로 매번 전체 검사하면 느리므로, 내 돌 주변만 검사
        moves_score = {}
        
        potential_moves = set()
        for r, c in my_stones:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < self.board_size and 0 <= nc < self.board_size and board[nr, nc] == 0:
                        potential_moves.add((nr, nc))
        
        # 4목, 5목 체크 (시뮬레이션)
        best_move = None
        best_priority = -1 # 0: clustering, 1: make 4, 2: make 5
        
        if not potential_moves: # 첫 수 혹은 돌이 없을 때
             return random.choice(empty_spots)

        for r, c in potential_moves:
            # 가상 착수
            board[r, c] = player
            if self.check_win(board, player):
                board[r, c] = 0
                return (r, c) # 승리하는 수 발견
            
            # 4목 체크 (약식: 4개 연결되면 우선순위 높음)
            # 여기서는 복잡한 룰 대신 간단히 '연결성'만 봄
            board[r, c] = 0
        
        # [Logic 3] Clustering (그냥 랜덤하게 둠, 대신 내 돌 근처)
        return random.choice(list(potential_moves))

    def get_random_move(self, board):
        empty = np.where(board == 0)
        if len(empty[0]) == 0: return None
        idx = np.random.randint(len(empty[0]))
        return (empty[0][idx], empty[1][idx])

    def generate_game(self, attacker_color):
        # attacker_color: 1(Black) or -1(White)
        # Attacker는 규칙대로, Defender는 랜덤으로 둠
        board = np.zeros((self.board_size, self.board_size), dtype=int)
        history = [] # (state, pi, player)
        
        curr = 1 # Black starts
        
        for _ in range(self.board_size * self.board_size):
            # 행동 결정
            if curr == attacker_color:
                move = self.get_attacker_move(board, curr)
            else:
                move = self.get_random_move(board) # 수비수는 랜덤 (바보)
            
            if move is None: break # 보드 꽉 참
            
            # State 저장용
            state = np.zeros((3, self.board_size, self.board_size), dtype=np.float32)
            state[0] = (board == curr).astype(np.float32)
            state[1] = (board == -curr).astype(np.float32)
            state[2] = 1.0 
            
            # PI (One-hot)
            pi = np.zeros(self.board_size * self.board_size, dtype=np.float32)
            pi[move[0]*self.board_size + move[1]] = 1.0
            
            history.append([state, pi, curr])
            board[move] = curr
            
            # [수정됨] 누가 이겼든 상관없이 승패가 나면 무조건 저장!
            if self.check_win(board, curr):
                return history, curr # (Data, Winner) - 무조건 리턴
            
            curr = -curr
            
        return None, None # 무승부는 학습 가치가 낮으므로 버림 (혹은 취향껏)

    def get_equi_data(self, history, winner):
        extend_data = []
        # Winner 기준 Z값 설정
        for state, pi, player in history:
            z = 1.0 if player == winner else -1.0
            
            pi_board = pi.reshape(self.board_size, self.board_size)
            for i in [1, 2, 3, 4]:
                # Rotate
                equi_state = np.array([np.rot90(s, k=i) for s in state])
                equi_pi = np.rot90(pi_board, k=i)
                extend_data.append([equi_state, equi_pi.flatten(), z])
                
                # Flip
                equi_state_flip = np.array([np.fliplr(s) for s in equi_state])
                equi_pi_flip = np.fliplr(equi_pi)
                extend_data.append([equi_state_flip, equi_pi_flip.flatten(), z])
        return extend_data

    def run(self, num_games):
        generated = 0
        while generated < num_games:
            # 50:50 확률로 흑공격/백공격 생성
            attacker = 1 if random.random() < 0.5 else -1
            history, winner = self.generate_game(attacker)
            
            if history is not None:
                # 데이터 증강 후 버퍼 전송
                aug_data = self.get_equi_data(history, winner)
                self.buffer_ref.add.remote(aug_data)
                generated += 1
                if generated % 100 == 0:
                    print(f"⚡ Synthetic Data: {generated}/{num_games} generated")
        print("✅ Synthetic Data Generation Complete!")


# =============================================================================
# [4] Data Worker (MCTS - 기존과 동일)
# =============================================================================
@ray.remote(num_cpus=1)
class DataWorker:
    def __init__(self, buffer_ref, inference_server):
        self.buffer_ref = buffer_ref
        self.inference_server = inference_server
        self.num_parallel_games = 128
        self.mcts_envs = [mcts_core.MCTS() for _ in range(self.num_parallel_games)]
        self.histories = [[] for _ in range(self.num_parallel_games)]
        self.sim_counts = [0] * self.num_parallel_games
        self.step_counts = [0] * self.num_parallel_games
        for mcts in self.mcts_envs:
            mcts.reset()
            mcts.add_root_noise(0.3, 0.25)

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
                    temp = 1.0 if self.step_counts[i] < 30 else 0.1
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
    
if __name__ == "__main__":
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
        
    else:
        print("🆕 Starting training from scratch.")
        
        # [STEP 1] 합성 데이터 주입
        print(f"\n🧪 Generating Synthetic Data ({SYNTHETIC_GAMES} games)...")
        syn_workers = [RuleBasedWorker.remote(buffer) for _ in range(4)]
        ray.get([w.run.remote(SYNTHETIC_GAMES // 4) for w in syn_workers])
        del syn_workers
        
        initial_buffer_size = ray.get(buffer.size.remote())
        print(f"✅ Initial Buffer Filled: {initial_buffer_size} samples ready!\n")

        # [STEP 1.5] Pre-training
        print(f"🧠 Pre-training on Synthetic Data...")
        pretrain_steps = (initial_buffer_size // BATCH_SIZE)*10
        
        model.train()
        for i in range(pretrain_steps):
            s_batch, pi_batch, z_batch = ray.get(buffer.sample.remote(BATCH_SIZE))
            
            s_tensor = torch.tensor(s_batch, dtype=torch.float32).to(TRAINER_DEVICE)
            pi_tensor = torch.tensor(pi_batch, dtype=torch.float32).to(TRAINER_DEVICE)
            z_tensor = torch.tensor(z_batch, dtype=torch.float32).to(TRAINER_DEVICE).unsqueeze(1)
            
            optimizer.zero_grad()
            
            # [AMP 수정 2] 최신 문법 적용 (autocast('cuda'))
            with torch.amp.autocast('cuda'):
                pred_pi, pred_v = model(s_tensor)
                loss_pi = -torch.mean(torch.sum(pi_tensor * pred_pi, dim=1))
                loss_v = F.mse_loss(pred_v, z_tensor)
                total_loss = loss_pi + loss_v
            
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            step += 1
            print(f"\r[Pre-train] Step {i+1}/{pretrain_steps} | Loss: {total_loss.item():.4f}", end="")
        
        print(f"\n✅ Pre-training Complete!")
        cpu_weights = {k: v.cpu() for k, v in model.state_dict().items()}
        ray.get(inference_server.update_weights.remote(cpu_weights))


    # -------------------------------------------------------------
    # [STEP 2] Main Loop
    # -------------------------------------------------------------
    print("🚀 Starting MCTS Workers...")
    workers = [DataWorker.remote(buffer, inference_server) for _ in range(NUM_ACTORS)]
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