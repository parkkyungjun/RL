import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
import time  # 딜레이를 위해 추가

# C++ 모듈
import mcts_core 

# =============================================================================
# [1] 설정
# =============================================================================
BOARD_SIZE = 15
NUM_RES_BLOCKS = 8      
NUM_CHANNELS = 128
MODEL_PATH = "models/checkpoint_20500.pth" # ✅ 관전하고 싶은 모델 경로
NUM_MCTS_SIMS = 1600     # 생각하는 횟수
WATCH_DELAY = 1.0       # 한 수 둘 때마다 1초씩 멈춤 (관전용)
TEMPERATURE = 0       # 0.0: 정수(Best)만 둠 / 1.0: 약간 다양하게 둠 (관전 꿀잼용)

# =============================================================================
# [2] 신경망 (동일)
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
# [3] 유틸리티
# =============================================================================
def print_board(board_state):
    """
    터미널에 바둑판을 예쁘게 출력합니다.
    board_state: 15x15 numpy array (1=흑, -1=백, 0=빈칸 가정)
    """
    print("\n   " + " ".join([f"{i:2}" for i in range(BOARD_SIZE)]))
    for r in range(BOARD_SIZE):
        line = f"{r:2} "
        for c in range(BOARD_SIZE):
            val = board_state[r][c]
            if val == 1:
                line += "⚫ " # 흑돌
            elif val == -1:
                line += "⚪ " # 백돌
            else:
                line += "➕ "
        print(line)
    print()

# =============================================================================
# [4] 관전 루프 (AI vs AI)
# =============================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE: {device}")

    # 모델 로드
    model = AlphaZeroNet().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device)['model_state_dict'])
        print(f"✅ 모델 로드 완료: {MODEL_PATH}")
    else:
        print(f"❌ 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        return
    model.eval()

    # MCTS 및 보드 초기화
    mcts = mcts_core.MCTS()
    mcts.reset()
    local_board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
    
    turn = 1 # 1: 흑, -1: 백
    move_count = 1
    
    print("="*40)
    print(f"      AI vs AI Self-Play Watch Mode      ")
    print(f"      Temperature: {TEMPERATURE} (0에 가까울수록 진지함)      ")
    print("="*40)
    
    print_board(local_board)
    time.sleep(1) # 시작 전 대기

    while True:
        player_name = "흑(Black)" if turn == 1 else "백(White)"
        print(f"[{move_count}수] {player_name} 생각 중...", end="", flush=True)

        # 1. AI 생각 (MCTS 시뮬레이션)
        # 관전용이므로 노이즈는 꺼도 되지만, 
        # 다양한 수를 보고 싶으면 add_root_noise(0.3, 0.25) 정도 줘도 됨
        mcts.add_root_noise(0.0, 0.0) 
        
        for i in range(NUM_MCTS_SIMS):
            leaf_state = mcts.select_leaf()
            if leaf_state is None: continue 
            
            state_tensor = torch.tensor(leaf_state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                pi_logits, value = model(state_tensor)
            
            probs = torch.exp(pi_logits).cpu().numpy().flatten()
            val = value.item()
            mcts.backpropagate(probs, val)

        print(" 결정!")

        # 2. 착수 선택
        # TEMPERATURE가 0이면 가장 승률 높은 수, 높으면 확률적으로 둠
        temp = 0
        _, pi = mcts.get_action_probs(TEMPERATURE) 
        
        if np.isnan(pi).any():
            print("⚠️ NaN detected in policy, falling back to argmax")
            action = np.argmax(pi) # NaN 무시하고 인덱스 반환 시도 (또는 랜덤)
            # 만약 argmax도 실패하면 그냥 랜덤
            if np.isnan(pi[action]): 
                action = np.random.choice(len(pi))
        else:
            # 확률 기반 선택
            action = np.random.choice(len(pi), p=pi)
            
        # 확률 기반 선택
        # action = np.random.choice(len(pi), p=pi)
        
        # 3. 보드 업데이트 및 출력
        r, c = action // BOARD_SIZE, action % BOARD_SIZE
        mcts.update_root_game(action)
        local_board[r][c] = turn
        
        print(f"👉 {player_name} 착수: ({r}, {c})")
        print_board(local_board)
        
        # 4. 종료 체크 (C++ 로직 사용)
        is_game_over, winner = mcts.check_game_status()
        if is_game_over:
            if winner == 0:
                print("🏁 무승부입니다! (Draw)")
            else:
                win_color = "흑(Black)" if winner == 1 else "백(White)"
                print(f"🎉 {win_color} 승리!")
            break
        
        # 5. 턴 넘김 및 딜레이
        turn *= -1
        move_count += 1
        
        # 사람이 볼 수 있게 멈춤
        # time.sleep(WATCH_DELAY)

if __name__ == "__main__":
    main()