import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

# C++ 모듈 (학습 코드와 동일하게 임포트)
import mcts_core 

# =============================================================================
# [1] 설정
# =============================================================================
BOARD_SIZE = 15
NUM_RES_BLOCKS = 8
NUM_CHANNELS = 128
MODEL_PATH = "models/checkpoint_20500.pth"  # ✅ 불러올 모델 경로 수정하세요
NUM_MCTS_SIMS = 800  # 생각하는 횟수 (높을수록 잘하지만 느려짐)

# =============================================================================
# [2] 신경망 클래스 (학습 코드와 동일해야 함)
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
# [3] 유틸리티 함수
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

def get_human_action():
    while True:
        try:
            inp = input("👉 당신의 차례입니다 (행,열 입력 예: 7,7): ")
            if ',' not in inp:
                print("형식이 잘못되었습니다. '행,열' 형태로 입력해주세요.")
                continue
            r, c = map(int, inp.split(','))
            
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                return r * BOARD_SIZE + c
            else:
                print(f"범위를 벗어났습니다. 0~{BOARD_SIZE-1} 사이로 입력하세요.")
        except ValueError:
            print("숫자를 입력해주세요.")

# =============================================================================
# [4] 게임 루프
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

    # MCTS 초기화
    mcts = mcts_core.MCTS()
    mcts.reset()

    # 시각화용 로컬 보드 (1: 흑(선공), -1: 백(후공))
    # 주의: 실제 게임 로직은 C++ mcts 내부에서 처리되지만, 
    # 화면 출력을 위해 파이썬 쪽에서도 보드 상태를 추적합니다.
    local_board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
    
    # 선공/후공 선택
    print("="*40)
    print("      OMOK AlphaZero Human vs AI      ")
    print("="*40)
    
    while True:
        choice = input("흑(선공)을 하시겠습니까? (y/n): ").lower()
        if choice in ['y', 'n']:
            human_color = 1 if choice == 'y' else -1
            break
            
    turn = 1 # 1=흑, -1=백
    game_over = False
    color = {-1: "백", 1: "흑"}
    print_board(local_board)

    while not game_over:
        # ---------------------------------------------------------
        # 1. Human Turn
        # ---------------------------------------------------------
        if turn == human_color:
            action = get_human_action()
            
            # 이미 둔 곳인지 체크 (로컬 보드 기준)
            r, c = action // BOARD_SIZE, action % BOARD_SIZE
            if local_board[r][c] != 0:
                print("⚠️ 이미 돌이 있는 자리입니다! 다시 두세요.")
                continue
                
            # C++ 엔진에 착수 업데이트
            # update_root_game은 해당 수가 승리수인지(게임종료) 반환한다고 가정
            mcts.update_root_game(action)
            is_game_over, winner = mcts.check_game_status()
            local_board[r][c] = turn
            print_board(local_board)
            
            if is_game_over:
                print(f"🎉 {color[winner]}이 이겼습니다! (믿기지 않네요)")
                break
            
        # ---------------------------------------------------------
        # 2. AI Turn
        # ---------------------------------------------------------
        else:
            print("🤖 AI가 생각 중입니다...", end="")
            
            # 시뮬레이션 수행
            # (학습 때 Worker 코드와 동일한 로직을 단일 스레드로 수행)
            mcts.add_root_noise(0.0, 0.0) # 실전에서는 노이즈 끔
            
            for i in range(NUM_MCTS_SIMS):
                leaf_state = mcts.select_leaf()
                if leaf_state is None: 
                    continue # 이미 종료된 노드
                
                # numpy -> tensor
                state_tensor = torch.tensor(leaf_state, dtype=torch.float32).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    pi_logits, value = model(state_tensor)
                
                # probs = torch.exp(pi_logits).cpu().numpy().flatten()
                probs = F.softmax(pi_logits, dim=1).cpu().numpy().flatten()
                val = value.item()
                
                mcts.backpropagate(probs, val)
                
                if i % 100 == 0: print(".", end="", flush=True)

            print(" 완료!")
            
            # 행동 선택 (실전이므로 탐험 없이 가장 많이 방문한 곳 선택 temp=0)
            # 학습 코드의 get_action_probs(temp) 함수 활용
            _, pi = mcts.get_action_probs(0.0) 
            ai_action = np.argmax(pi)
            
            r, c = ai_action // BOARD_SIZE, ai_action % BOARD_SIZE
            print(f"🤖 AI가 ({r}, {c})에 두었습니다.")
            
            mcts.update_root_game(ai_action)
            is_game_over, winner = mcts.check_game_status()
            local_board[r][c] = turn
            print_board(local_board)
            
            if is_game_over:
                print("💀 AI가 이겼습니다. 더 수련하고 오세요.")
                break

        # 턴 교체
        turn *= -1

if __name__ == "__main__":
    main()