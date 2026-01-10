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
MODEL_PATH = "models/checkpoint_100000.pth"
NUM_MCTS_SIMS = 1600

# =============================================================================
# [2] 신경망 클래스
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

# =============================================================================
# [3] 유틸리티 함수
# =============================================================================
def print_board(board_state):
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
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ 모델 로드 완료: {MODEL_PATH}")
        except Exception as e:
            print(f"❌ 모델 로드 중 에러 발생: {e}")
            return
    else:
        print(f"❌ 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        # 모델이 없으면 실행하지 않으려면 return, 그냥 초기화된 모델로 하려면 주석 처리
        # return 
    model.eval()

    # MCTS 초기화
    mcts = mcts_core.MCTS()
    mcts.reset()

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

    # [수정됨] 첫 수 카운팅을 위해 변수 추가 (또는 보드가 비었는지 확인해도 됨)
    move_count = 0 

    while not game_over:
        # =========================================================
        # [수정됨] 첫 수(Move 0) 강제 착수 로직
        # =========================================================
        if move_count == 0:
            print("✨ 공정한 시작을 위해 첫 수는 천원(7, 7)에 착수합니다.")
            
            # 강제로 (7, 7) 좌표 계산
            center_r, center_c = 7, 7
            center_action = center_r * BOARD_SIZE + center_c
            
            # 엔진 및 보드 업데이트
            mcts.update_root_game(center_action)
            is_game_over, winner = mcts.check_game_status() # 여기서 끝날 일은 없겠지만 형식상 호출
            local_board[center_r][center_c] = turn
            
            print_board(local_board)
            
            # 턴 넘기기
            turn *= -1
            move_count += 1
            continue  # 루프의 처음으로 돌아가 다음 턴 진행
        
        # ---------------------------------------------------------
        # 1. Human Turn
        # ---------------------------------------------------------
        if turn == human_color:
            action = get_human_action()
            
            r, c = action // BOARD_SIZE, action % BOARD_SIZE
            if local_board[r][c] != 0:
                print("⚠️ 이미 돌이 있는 자리입니다! 다시 두세요.")
                continue
                
            mcts.update_root_game(action)
            is_game_over, winner = mcts.check_game_status()
            local_board[r][c] = turn
            print_board(local_board)
            move_count += 1 # 착수 횟수 증가
            
            if is_game_over:
                print(f"🎉 {color[winner]}이 이겼습니다! (믿기지 않네요)")
                break
            
        # ---------------------------------------------------------
        # 2. AI Turn
        # ---------------------------------------------------------
        else:
            print("🤖 AI가 생각 중입니다...", end="")
            
            for i in range(NUM_MCTS_SIMS):
                leaf_state = mcts.select_leaf()
                if leaf_state is None: 
                    continue
                
                state_tensor = torch.tensor(leaf_state, dtype=torch.float32).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    pi_logits, value = model(state_tensor)
                
                probs = F.softmax(pi_logits, dim=1).cpu().numpy().flatten()
                val = value.item()
                
                mcts.backpropagate(probs, val)
                
                if i % 200 == 0: print(".", end="", flush=True)

            print(" 완료!")
            
            _, pi = mcts.get_action_probs(0.0) # 탐욕적 선택
            ai_action = np.argmax(pi)
            
            r, c = ai_action // BOARD_SIZE, ai_action % BOARD_SIZE
            print(f"🤖 AI가 ({r}, {c})에 두었습니다.")
            
            mcts.update_root_game(ai_action)
            is_game_over, winner = mcts.check_game_status()
            local_board[r][c] = turn
            print_board(local_board)
            move_count += 1 # 착수 횟수 증가
            
            if is_game_over:
                print("💀 AI가 이겼습니다. 더 수련하고 오세요.")
                break

        # 턴 교체
        turn *= -1

if __name__ == "__main__":
    main()