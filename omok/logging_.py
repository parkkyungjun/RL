import matplotlib
matplotlib.use('Agg') # [중요] GUI 없는 리눅스에서 에러 방지
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

BOARD_SIZE = 15
def save_game_log(worker_id, game_idx, actions, winner, board_size):
    if not os.path.exists("debug_games"):
        os.makedirs("debug_games")
    
    # 파일명: worker_0_game_123_winner_1.txt
    filename = f"debug_games/worker_{worker_id}_game_{game_idx}_winner_{winner}.txt"
    
    board = [['.' for _ in range(board_size)] for _ in range(board_size)]
    
    log_content = []
    log_content.append(f"Worker ID: {worker_id}")
    log_content.append(f"Game Index: {game_idx}")
    log_content.append(f"Winner: {winner} (1:Black, -1:White, 0:Draw)")
    log_content.append(f"Total Moves: {len(actions)}")
    log_content.append("-" * 20)

    # 대국 재현
    players = ["X", "O"] # X가 흑(선공), O가 백(후공)
    current_p_idx = 0 
    
    move_record = []

    for action in actions:
        row = action // board_size
        col = action % board_size
        
        stone = players[current_p_idx]
        board[row][col] = stone
        move_record.append(f"{stone}: ({row}, {col})")
        
        current_p_idx = (current_p_idx + 1) % 2

    # 보드 출력 (시각화)
    log_content.append("Final Board State:")
    log_content.append("   " + " ".join([str(i) for i in range(board_size)]))
    for r in range(board_size):
        row_str = f"{r:2} " + " ".join(board[r])
        log_content.append(row_str)
    
    log_content.append("-" * 20)
    log_content.append("Move History:")
    log_content.append("\n".join(move_record))

    with open(filename, "w", encoding='utf-8') as f:
        f.write("\n".join(log_content))
        
def save_debug_files(step, s_batch, pi_batch, z_batch):
    """
    배치 데이터 중 첫 번째 샘플을 텍스트와 이미지로 저장합니다.
    """
    # 텐서 -> 넘파이 변환 (필요시)
    if hasattr(s_batch, 'cpu'): s_batch = s_batch.cpu().numpy()
    if hasattr(pi_batch, 'cpu'): pi_batch = pi_batch.cpu().numpy()
    if hasattr(z_batch, 'cpu'): z_batch = z_batch.cpu().numpy()

    # 첫 번째 샘플만 추출
    state = s_batch[0]   # (3, BOARD_SIZE, BOARD_SIZE)
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
        
        f.write("   " + " ".join([f"{i%10}" for i in range(BOARD_SIZE)]) + "\n")
        for r in range(BOARD_SIZE):
            row_str = f"{r:2d} "
            for c in range(BOARD_SIZE):
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
            r, c = divmod(idx, BOARD_SIZE)
            f.write(f"  Pos({r},{c}): {pi[idx]:.4f}\n")

    # ---------------------------------------------------------
    # [2] 이미지 파일 저장 (채널별 시각화)
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(BOARD_SIZE, 5))
    
    # 내 돌 (Channel 0)
    axes[0].imshow(state[0], cmap='Blues', vmin=0, vmax=1)
    axes[0].set_title(f"My Stones (Ch0)\nTurn info: {state[2][0][0]}")
    
    # 상대 돌 (Channel 1)
    axes[1].imshow(state[1], cmap='Reds', vmin=0, vmax=1)
    axes[1].set_title("Opponent Stones (Ch1)")
    
    # Policy 분포 (Heatmap)
    pi_grid = pi.reshape(BOARD_SIZE, BOARD_SIZE)
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