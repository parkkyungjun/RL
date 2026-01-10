import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
import time
from tqdm import tqdm

import mcts_core  # C++ 모듈

# =============================================================================
# [1] 설정
# =============================================================================
# 모델 경로 (수정 필요)
MODEL_A_PATH = "models/checkpoint_18000.pth"
MODEL_B_PATH = "models/checkpoint_80000.pth"

# 대결 설정
TOTAL_GAMES = 1000        # 총 대결 수
BATCH_SIZE = 1024           # 한 번에 동시에 돌릴 게임 수 (VRAM에 맞춰 조절, 32~128 추천)
NUM_MCTS_SIMS = 800       # 배치에서는 속도를 위해 시뮬레이션 수를 적절히 타협
BOARD_SIZE = 15
NUM_CHANNELS = 128
NUM_RES_BLOCKS = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# [2] 신경망 (기존과 동일)
# =============================================================================
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
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
        self.bn_start = nn.GroupNorm(num_groups=8, num_channels=NUM_CHANNELS)
        self.backbone = nn.Sequential(*[ResBlock(NUM_CHANNELS) for _ in range(NUM_RES_BLOCKS)])
        
        self.policy_head = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, 2, 1), 
            nn.GroupNorm(num_groups=1, num_channels=2), 
            nn.ReLU(),
            nn.Flatten(), 
            nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)
        )
        
        self.value_head = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, 1, 1), 
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

def load_model(path):
    model = AlphaZeroNet().to(DEVICE)
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=DEVICE)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        return model
    else:
        print(f"❌ 모델 없음: {path}")
        sys.exit()

# =============================================================================
# [3] 배치 추론 함수 (핵심)
# =============================================================================
def run_batch_mcts(active_games, active_indices, models, turns, sims):
    """
    여러 게임(active_games)의 MCTS 시뮬레이션을 한 번에 처리합니다.
    active_games: 현재 진행 중인 게임 리스트 (MCTS 객체들의 리스트)
    active_indices: 전체 배치 중 진행 중인 인덱스
    models: {1: 흑모델, -1: 백모델}
    turns: 각 게임의 현재 턴 [1, -1, 1, ...]
    """
    
    # MCTS 시뮬레이션 횟수만큼 반복
    for _ in range(sims):
        # 1. 모든 활성 게임에서 leaf node 수집
        leaves = []
        valid_indices = [] # 실제로 계산이 필요한 게임의 인덱스
        
        # 현재 턴인 모델끼리 묶어서 처리해야 함 (흑차례 게임들 / 백차례 게임들)
        # 하지만 간단하게 하기 위해 이번엔 그냥 순차적으로 수집 후 Batch 처리
        
        batch_states = []
        mapping = [] # (원래게임인덱스, 턴)
        
        for i, game_idx in enumerate(active_indices):
            mcts_black, mcts_white = active_games[i]
            turn = turns[i]
            
            # 현재 턴의 MCTS 선택
            current_mcts = mcts_black if turn == 1 else mcts_white
            
            leaf_state = current_mcts.select_leaf()
            
            # leaf_state가 None이면 이미 끝난 게임이거나 오류 (여기선 skip)
            if leaf_state is not None:
                batch_states.append(leaf_state)
                mapping.append((i, turn, current_mcts))
        
        if not batch_states:
            continue

        # 2. 텐서 변환 및 GPU 추론
        state_tensor = torch.tensor(np.array(batch_states), dtype=torch.float32).to(DEVICE)
        
        # 모델은 흑/백이 다를 수 있음. 마스크를 써서 따로 추론하거나, 
        # 그냥 단순히 나눠서 추론 후 합침. 여기선 나눠서 추론.
        
        results = [None] * len(batch_states)
        
        # (A) 흑 차례인 상태들 추론
        black_indices = [k for k, (g_idx, t, m) in enumerate(mapping) if t == 1]
        if black_indices:
            b_states = state_tensor[black_indices]
            with torch.no_grad():
                pi, v = models[1](b_states)
            pi = F.softmax(pi, dim=1).cpu().numpy()
            v = v.cpu().numpy().flatten()
            for k, idx in enumerate(black_indices):
                results[idx] = (pi[k], v[k])
                
        # (B) 백 차례인 상태들 추론
        white_indices = [k for k, (g_idx, t, m) in enumerate(mapping) if t == -1]
        if white_indices:
            w_states = state_tensor[white_indices]
            with torch.no_grad():
                pi, v = models[-1](w_states)
            pi = F.softmax(pi, dim=1).cpu().numpy()
            v = v.cpu().numpy().flatten()
            for k, idx in enumerate(white_indices):
                results[idx] = (pi[k], v[k])
        
        # 3. Backpropagation
        for k, (g_idx, t, mcts_obj) in enumerate(mapping):
            prob, val = results[k]
            mcts_obj.backpropagate(prob, val)

# =============================================================================
# [4] 배치 대결 실행기
# =============================================================================
def run_match_batch(model_b, model_w, num_games):
    """
    model_b: 흑돌 잡을 모델
    model_w: 백돌 잡을 모델
    num_games: 진행할 게임 수
    반환: {1: 흑승수, -1: 백승수, 0: 무승부}
    """
    
    results = {1: 0, -1: 0, 0: 0} # 1: 흑승, -1: 백승, 0: 무승부
    remaining_games = num_games
    
    # 진행바
    pbar = tqdm(total=num_games, desc="Running Batch")
    
    while remaining_games > 0:
        # 이번에 돌릴 배치 크기 결정
        current_batch_size = min(BATCH_SIZE, remaining_games)
        
        # 게임 초기화
        # games list: [(mcts_black, mcts_white), ...]
        games = []
        for _ in range(current_batch_size):
            mb = mcts_core.MCTS()
            mw = mcts_core.MCTS()
            mb.reset()
            mw.reset()
            games.append((mb, mw))
            
        game_turns = [1] * current_batch_size  # 모든 게임 흑부터 시작
        move_counts = [0] * current_batch_size
        game_active = [True] * current_batch_size
        active_count = current_batch_size
        
        # 배치 게임 루프
        while active_count > 0:
            # 1. 활성 게임 인덱스 추출
            active_indices = [i for i, active in enumerate(game_active) if active]
            active_game_objs = [games[i] for i in active_indices]
            active_turns = [game_turns[i] for i in active_indices]
            
            # 2. 배치 MCTS 수행 (생각하기)
            # 흑모델(model_b), 백모델(model_w) 전달
            run_batch_mcts(active_game_objs, active_indices, {1: model_b, -1: model_w}, active_turns, NUM_MCTS_SIMS)
            
            # 3. 착수 및 결과 확인
            for i in active_indices:
                mb, mw = games[i]
                turn = game_turns[i]
                current_mcts = mb if turn == 1 else mw
                
                # Temperature: 초반 6수까지 1.0, 이후 0.05
                temp = 1.0 if move_counts[i] < 6 else 0.05
                
                # Action 선택
                _, pi = current_mcts.get_action_probs(temp)
                action = np.random.choice(len(pi), p=pi)
                
                # 양쪽 MCTS 업데이트
                mb.update_root_game(action)
                mw.update_root_game(action)
                
                move_counts[i] += 1
                
                # 종료 체크
                is_over, winner = mb.check_game_status()
                
                if is_over:
                    results[winner] += 1
                    game_active[i] = False
                    active_count -= 1
                    pbar.update(1)
                elif move_counts[i] > 225: # 무승부 강제
                    results[0] += 1
                    game_active[i] = False
                    active_count -= 1
                    pbar.update(1)
                else:
                    game_turns[i] *= -1 # 턴 교체

        remaining_games -= current_batch_size
        
    pbar.close()
    return results

# =============================================================================
# [5] 메인 실행
# =============================================================================
if __name__ == "__main__":
    print(f"🔹 Device: {DEVICE}")
    print(f"🔹 Model A (Evaluator): {MODEL_A_PATH}")
    print(f"🔹 Model B (Target):    {MODEL_B_PATH}")
    print(f"🔹 Total Games: {TOTAL_GAMES} (Half & Half)")
    print(f"🔹 Batch Size:  {BATCH_SIZE}")
    print("-" * 50)

    model_a = load_model(MODEL_A_PATH)
    model_b = load_model(MODEL_B_PATH)

    # 1. 전반전: A가 흑, B가 백
    print("\n⚔️  [Round 1] Model A(흑) vs Model B(백) ...")
    half_games = TOTAL_GAMES // 2
    res1 = run_match_batch(model_a, model_b, half_games)
    
    # 2. 후반전: B가 흑, A가 백
    print("\n⚔️  [Round 2] Model B(흑) vs Model A(백) ...")
    res2 = run_match_batch(model_b, model_a, half_games)

    # 3. 결과 집계 및 출력
    # res1: {1: A승, -1: B승, 0: 무승부}
    # res2: {1: B승, -1: A승, 0: 무승부}
    
    a_wins_black = res1[1]
    a_wins_white = res2[-1]
    b_wins_black = res2[1]
    b_wins_white = res1[-1]
    draws = res1[0] + res2[0]
    
    total_a_wins = a_wins_black + a_wins_white
    total_b_wins = b_wins_black + b_wins_white
    
    print("\n" + "="*60)
    print(f"{'FINAL STATISTICS':^60}")
    print("="*60)
    
    print(f"📊 Model A ({MODEL_A_PATH})")
    print(f"   - 흑(Black) 승리: {a_wins_black} / {half_games}")
    print(f"   - 백(White) 승리: {a_wins_white} / {half_games}")
    print(f"   👉 Total Wins:   {total_a_wins} ({total_a_wins/TOTAL_GAMES*100:.1f}%)")
    print("-" * 60)
    
    print(f"📊 Model B ({MODEL_B_PATH})")
    print(f"   - 흑(Black) 승리: {b_wins_black} / {half_games}")
    print(f"   - 백(White) 승리: {b_wins_white} / {half_games}")
    print(f"   👉 Total Wins:   {total_b_wins} ({total_b_wins/TOTAL_GAMES*100:.1f}%)")
    print("-" * 60)
    
    print(f"🤝 무승부(Draws): {draws} ({draws/TOTAL_GAMES*100:.1f}%)")
    print("="*60)
    
    if total_a_wins > total_b_wins:
        print("🏆 Winner: Model A")
    elif total_b_wins > total_a_wins:
        print("🏆 Winner: Model B")
    else:
        print("🤝 Result: Tie")