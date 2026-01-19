import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import mcts_core  # C++ 모듈

# =============================================================================
# [1] 설정 및 모델 클래스
# =============================================================================
# streamlit run app.py
BOARD_SIZE = 15
NUM_RES_BLOCKS = 8
NUM_CHANNELS = 128
MODEL_PATH = "models/checkpoint_340000.pth"
NUM_MCTS_SIMS = 1600  # 반응 속도를 고려하여 조정

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

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlphaZeroNet().to(device)
    if os.path.exists(MODEL_PATH):
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ 모델 로드 완료: {MODEL_PATH}")
        except Exception as e:
            st.error(f"모델 로드 실패: {e}")
    else:
        st.warning(f"모델 파일 없음: {MODEL_PATH}. 초기화된 모델을 사용합니다.")
    model.eval()
    return model, device

model, device = load_model()

# =============================================================================
# [2] 게임 로직 및 UI 설정
# =============================================================================
st.set_page_config(page_title="AlphaZero Omok", layout="centered")

st.markdown("""
    <style>
    div.stButton > button {
        width: 38px; height: 38px; padding: 0px;
        font-size: 20px; border-radius: 5px; margin: 0px;
    }
    div[data-testid="column"] {
        width: auto !important; flex: 0 0 auto !important;
        min-width: 0 !important; padding: 1px !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("⚪ AlphaZero Omok AI ⚫")

# --- 무르기(Undo) 함수 구현 ---
def undo_last_move():
    """
    가장 최근의 수순(Human + AI)을 취소하고 상태를 재구축합니다.
    """
    # 기록이 없으면 무시
    if not st.session_state.history:
        return

    # 보통 '나의 실수'를 되돌리려면 [내 수 + AI 수] 2개를 빼야 내 차례가 됩니다.
    # 하지만 게임이 끝났거나, AI가 두기 전 등 상황에 따라 1개만 뺄 수도 있습니다.
    # 여기서는 간단하게: "현재 턴이 사람 턴이면 2개(AI, 나) 삭제", "AI 턴이면(혹은 종료시) 로직에 맞게 삭제"
    
    # 전략: History에서 2개를 pop하고, 처음부터 다시 둔다.
    # (AI가 선공이라 History 길이가 1인 경우 등 예외 처리 필요)
    
    to_pop = 2
    if len(st.session_state.history) < 2:
        to_pop = len(st.session_state.history)
        # 만약 AI가 선공이라 처음에 1개(AI)만 있는데 무르기를 하면? -> 그냥 초기화와 같음
    
    # 1. 기록 삭제
    for _ in range(to_pop):
        if st.session_state.history:
            st.session_state.history.pop()
            
    # 2. 보드 및 MCTS 완전 초기화
    st.session_state.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
    st.session_state.game_over = False
    st.session_state.winner = None
    st.session_state.last_move = None
    st.session_state.mcts.reset() # C++ MCTS 객체 리셋
    
    # 3. 턴 초기화 (흑부터 시작)
    st.session_state.turn = 1 
    
    # 4. History 재실행 (Replay)
    for idx in st.session_state.history:
        r, c = idx // BOARD_SIZE, idx % BOARD_SIZE
        
        # 보드에 착수
        st.session_state.board[r][c] = st.session_state.turn
        
        # MCTS 트리에 착수 반영
        st.session_state.mcts.update_root_game(idx)
        
        # 마지막 수 갱신
        st.session_state.last_move = (r, c)
        
        # 턴 넘기기
        st.session_state.turn *= -1
        
    st.success("⏪ 무르기 완료!")

# --- 사이드바 설정 ---
with st.sidebar:
    st.header("게임 설정")
    
    user_color_choice = st.radio("당신의 돌을 선택하세요:", ("흑 (선공)", "백 (후공)"))
    human_color = 1 if "흑" in user_color_choice else -1
    
    sims = st.slider("AI 생각 깊이 (Simulations)", 100, 2000, NUM_MCTS_SIMS, step=100)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 재시작", type="primary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    with col2:
        # 무르기 버튼 추가
        if st.button("⏪ 무르기"):
            undo_last_move()
            st.rerun()

# --- 상태 초기화 ---
if 'board' not in st.session_state:
    st.session_state.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
    st.session_state.turn = 1
    st.session_state.game_over = False
    st.session_state.winner = None
    st.session_state.last_move = None
    st.session_state.history = [] # [NEW] 착수 기록 저장용 리스트
    
    mcts = mcts_core.MCTS()
    mcts.reset()
    st.session_state.mcts = mcts

    # AI 선공(흑) 처리
    if human_color == -1:
        center = 7 * BOARD_SIZE + 7
        st.session_state.mcts.update_root_game(center)
        st.session_state.board[7][7] = 1
        st.session_state.turn = -1
        st.session_state.last_move = (7, 7)
        st.session_state.history.append(center) # [NEW] 기록 추가

# --- AI 착수 로직 ---
def run_ai_turn():
    if st.session_state.game_over:
        return

    mcts = st.session_state.mcts
    progress_bar = st.progress(0, text="AI가 생각 중입니다...")
    
    for i in range(sims):
        leaf_state = mcts.select_leaf()
        if leaf_state is None: continue
            
        state_tensor = torch.tensor(leaf_state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pi_logits, value = model(state_tensor)
            
        probs = F.softmax(pi_logits, dim=1).cpu().numpy().flatten()
        val = value.item()
        mcts.backpropagate(probs, val)
        
        if i % (sims // 10) == 0:
            progress_bar.progress((i + 1) / sims, text=f"AI 생각 중... ({i}/{sims})")
            
    progress_bar.empty()

    _, pi = mcts.get_action_probs(0.0)
    ai_action = int(np.argmax(pi))
    r, c = ai_action // BOARD_SIZE, ai_action % BOARD_SIZE
    
    # 상태 업데이트
    st.session_state.mcts.update_root_game(ai_action)
    st.session_state.board[r][c] = st.session_state.turn
    st.session_state.last_move = (r, c)
    st.session_state.history.append(ai_action) # [NEW] 기록 추가
    
    is_over, winner = st.session_state.mcts.check_game_status()
    if is_over:
        st.session_state.game_over = True
        st.session_state.winner = winner
    else:
        st.session_state.turn *= -1
        st.rerun()

# --- 메인 보드 UI ---
st.write(f"현재 차례: **{'흑 (⚫)' if st.session_state.turn == 1 else '백 (⚪)'}**")

if st.session_state.game_over:
    winner_text = "흑 (⚫)" if st.session_state.winner == 1 else "백 (⚪)"
    if st.session_state.winner == 0:
        st.info("🏁 무승부입니다!")
    else:
        msg = "승리! 🎉" if st.session_state.winner == human_color else "패배... 💀"
        st.success(f"{msg} {winner_text} 승.")

for r in range(BOARD_SIZE):
    cols = st.columns(BOARD_SIZE)
    for c in range(BOARD_SIZE):
        idx = r * BOARD_SIZE + c
        val = st.session_state.board[r][c]
        
        label = " "
        if val == 1: label = "⚫"
        elif val == -1: label = "⚪"
        
        if st.session_state.last_move == (r, c):
            label = "🔴" if val == 1 else "⭕"

        is_disabled = st.session_state.game_over or (st.session_state.turn != human_color)
        
        if cols[c].button(label, key=f"btn_{r}_{c}", disabled=is_disabled):
            if val == 0:
                # [사람 착수]
                st.session_state.board[r][c] = st.session_state.turn
                st.session_state.mcts.update_root_game(idx)
                st.session_state.last_move = (r, c)
                st.session_state.history.append(idx) # [NEW] 기록 추가
                
                is_over, winner = st.session_state.mcts.check_game_status()
                if is_over:
                    st.session_state.game_over = True
                    st.session_state.winner = winner
                    st.rerun()
                else:
                    st.session_state.turn *= -1
                    st.rerun()

if not st.session_state.game_over and st.session_state.turn != human_color:
    run_ai_turn()