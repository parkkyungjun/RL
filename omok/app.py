import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import mcts_core  # C++ 모듈

# =============================================================================
# [1] 설정 및 모델 클래스 (기존 코드와 동일)
# =============================================================================
BOARD_SIZE = 15
NUM_RES_BLOCKS = 8
NUM_CHANNELS = 128
MODEL_PATH = "models/checkpoint_100000.pth"  # 경로 확인 필요
NUM_MCTS_SIMS = 400  # 웹 반응 속도를 위해 400~800회 추천 (1600은 조금 느릴 수 있음)

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
# [2] 리소스 캐싱 (모델 로딩 최적화)
# =============================================================================
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
# [3] 게임 로직 및 UI
# =============================================================================
st.set_page_config(page_title="AlphaZero Omok", layout="centered")

# CSS 스타일링: 버튼을 정사각형으로 만들고 간격을 좁힘
st.markdown("""
    <style>
    div.stButton > button {
        width: 38px;
        height: 38px;
        padding: 0px;
        font-size: 20px;
        border-radius: 5px;
        margin: 0px;
    }
    /* 버튼 간격 최소화 */
    div[data-testid="column"] {
        width: auto !important;
        flex: 0 0 auto !important;
        min-width: 0 !important;
        padding: 1px !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("⚪ AlphaZero Omok AI ⚫")

# --- 사이드바 설정 ---
with st.sidebar:
    st.header("게임 설정")
    
    # 선공/후공 선택
    user_color_choice = st.radio("당신의 돌을 선택하세요:", ("흑 (선공)", "백 (후공)"))
    human_color = 1 if "흑" in user_color_choice else -1
    
    # 난이도(시뮬레이션 횟수) 조절
    sims = st.slider("AI 생각 깊이 (Simulations)", 100, 2000, 400, step=100)
    
    if st.button("🔄 새 게임 시작", type="primary"):
        # 세션 초기화
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# --- 게임 상태 초기화 (Session State) ---
if 'board' not in st.session_state:
    st.session_state.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
    st.session_state.turn = 1  # 1: 흑, -1: 백
    st.session_state.game_over = False
    st.session_state.winner = None
    st.session_state.last_move = None
    
    # MCTS 초기화
    mcts = mcts_core.MCTS()
    mcts.reset()
    st.session_state.mcts = mcts # MCTS 객체를 세션에 저장

    # 만약 AI가 선공(흑)이라면 첫 수(7,7) 강제 착수
    if human_color == -1: # 인간이 백이면, AI는 흑
        center = 7 * BOARD_SIZE + 7
        st.session_state.mcts.update_root_game(center)
        st.session_state.board[7][7] = 1
        st.session_state.turn = -1
        st.session_state.last_move = (7, 7)

# --- 헬퍼 함수: AI 착수 로직 ---
def run_ai_turn():
    if st.session_state.game_over:
        return

    mcts = st.session_state.mcts
    
    # 진행바 표시
    progress_bar = st.progress(0, text="AI가 생각 중입니다...")
    
    # MCTS 시뮬레이션
    for i in range(sims):
        leaf_state = mcts.select_leaf()
        if leaf_state is None:
            continue
            
        state_tensor = torch.tensor(leaf_state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pi_logits, value = model(state_tensor)
            
        probs = F.softmax(pi_logits, dim=1).cpu().numpy().flatten()
        val = value.item()
        mcts.backpropagate(probs, val)
        
        # 진행바 업데이트 (너무 자주하면 느려지므로 10%마다)
        if i % (sims // 10) == 0:
            progress_bar.progress((i + 1) / sims, text=f"AI 생각 중... ({i}/{sims})")
            
    progress_bar.empty() # 진행바 제거

    # 행동 선택 (Greedy)
    _, pi = mcts.get_action_probs(0.0)
    ai_action = int(np.argmax(pi))
    
    r, c = ai_action // BOARD_SIZE, ai_action % BOARD_SIZE
    
    # 상태 업데이트
    st.session_state.mcts.update_root_game(ai_action)
    st.session_state.board[r][c] = st.session_state.turn
    st.session_state.last_move = (r, c)
    
    # 승패 체크
    is_over, winner = st.session_state.mcts.check_game_status()
    if is_over:
        st.session_state.game_over = True
        st.session_state.winner = winner
    else:
        st.session_state.turn *= -1 # 턴 변경
        st.rerun() # 화면 갱신하여 턴 넘김

# --- 메인 보드 UI ---
st.write(f"현재 차례: **{'흑 (⚫)' if st.session_state.turn == 1 else '백 (⚪)'}**")

# 게임 종료 메시지
if st.session_state.game_over:
    winner_text = "흑 (⚫)" if st.session_state.winner == 1 else "백 (⚪)"
    if st.session_state.winner == 0: # 무승부
        st.info("🏁 무승부입니다!")
    else:
        if st.session_state.winner == human_color:
            st.success(f"🎉 승리! {winner_text}이 이겼습니다.")
        else:
            st.error(f"💀 패배... {winner_text}이 이겼습니다.")

# 보드 그리기 (15x15)
# columns 간격을 최소화하기 위해 gap="small" 사용 불가 (columns 자체가 좁아야 함)
# 하지만 st.columns는 반응형이라 완벽한 정사각은 CSS로 제어함

for r in range(BOARD_SIZE):
    cols = st.columns(BOARD_SIZE)
    for c in range(BOARD_SIZE):
        idx = r * BOARD_SIZE + c
        val = st.session_state.board[r][c]
        
        # 버튼 라벨 결정
        label = " "
        if val == 1: label = "⚫"
        elif val == -1: label = "⚪"
        
        # 마지막 둔 수 강조 (빨간 테두리 느낌은 텍스트로 대체 or CSS)
        if st.session_state.last_move == (r, c):
            label = "🔴" if val == 1 else "⭕" # 강조 표시

        # 버튼 생성 (키는 유니크해야 함)
        # 게임이 끝났거나 AI 턴이면 버튼 비활성화 (disabled=True)
        is_disabled = st.session_state.game_over or (st.session_state.turn != human_color)
        
        if cols[c].button(label, key=f"btn_{r}_{c}", disabled=is_disabled):
            if val == 0: # 빈 칸일 때만
                # 1. 사람 착수 처리
                st.session_state.board[r][c] = st.session_state.turn
                st.session_state.mcts.update_root_game(idx)
                st.session_state.last_move = (r, c)
                
                # 승패 체크
                is_over, winner = st.session_state.mcts.check_game_status()
                if is_over:
                    st.session_state.game_over = True
                    st.session_state.winner = winner
                    st.rerun()
                else:
                    st.session_state.turn *= -1
                    st.rerun()

# --- AI 턴 자동 실행 ---
# 화면이 다시 그려진 후, 현재 턴이 AI라면 로직 실행
if not st.session_state.game_over and st.session_state.turn != human_color:
    run_ai_turn()