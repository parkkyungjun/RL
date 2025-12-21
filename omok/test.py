import mcts_core
import numpy as np

def print_board(board, size=8):
    print("   " + " ".join([f"{i}" for i in range(size)]))
    for r in range(size):
        row = []
        for c in range(size):
            val = board[r * size + c]
            if val == 1: row.append("O") # 흑
            elif val == -1: row.append("X") # 백
            else: row.append(".")
        print(f"{r:2d} " + " ".join(row))

def test_game_logic():
    print("🚀 Logic Verification Start (8x8 Random Play)")
    mcts = mcts_core.MCTS()
    mcts.reset()
    
    board_size = 8
    steps = 0
    history = [] # (player, action)
    
    while True:
        # 1. 현재 턴 플레이어 확인
        current_player = mcts.get_current_player()
        
        # 2. 랜덤 액션 선택
        temp = 1.0
        # MCTS 시뮬레이션 (최소한으로)
        for _ in range(50):
            mcts.select_leaf()
            # 정책/가치는 랜덤으로 줌 (로직 테스트니까)
            dummy_pi = np.ones(board_size*board_size, dtype=np.float32)
            mcts.backpropagate(dummy_pi, 0.0)
            
        _, pi_probs = mcts.get_action_probs(temp)
        action = np.random.choice(len(pi_probs), p=pi_probs)
        
        # 기록
        history.append((current_player, action))
        
        # 3. 수 두기
        mcts.update_root_game(action)
        steps += 1
        
        # 4. 상태 확인
        is_over, winner = mcts.check_game_status()
        
        if is_over or steps >= 64:
            print(f"\n🛑 Game Over at step {steps}")
            print(f"🏆 Winner from C++: {winner} (1=Black, -1=White, 0=Draw)")
            
            # 최종 보드 출력 (C++ 상태 가져오기)
            # mcts.get_action_probs는 (state, pi)를 리턴함. state를 파싱해야 함
            # 하지만 간단하게 mcts_core에 get_board 같은게 없으므로
            # 우리가 직접 기록한 history로 재구성해보자.
            
            final_board = [0] * (board_size * board_size)
            for p, a in history:
                final_board[a] = p
            
            print_board(final_board, board_size)
            
            # 검증: 실제 5목이 있는지 체크
            # (Winner가 0이 아닌데 5목이 안 보이면 버그)
            print("-" * 30)
            print(f"Checking consistency for Winner {winner}...")
            
            if winner != 0:
                # 위 프린트된 보드에서 winner의 돌이 5개 연결되었는지 눈으로 확인하세요.
                pass
            
            # 데이터 라벨링 시뮬레이션
            print("\n📊 Checking Z-value labeling logic:")
            for i, (h_player, h_action) in enumerate(history):
                # DataWorker 로직 복사
                if winner == 0: z = 0.0
                elif h_player == winner: z = 1.0
                else: z = -1.0
                
                # 마지막 수(이긴 수)에 대한 Z값 확인
                if i == len(history) - 1:
                    print(f"Last Move by P{h_player} -> Z={z}")
                    if z != 1.0 and winner != 0:
                        print("❌ CRITICAL BUG: 이긴 사람의 마지막 수 z값이 1.0이 아님!")
                    elif z == 1.0:
                        print("✅ Logic OK: 이긴 사람의 마지막 수 z값이 1.0임.")
            break

if __name__ == "__main__":
    test_game_logic()