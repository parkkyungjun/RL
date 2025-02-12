import gym
import numpy as np
import imageio

# ----------------------------
# 1. 하이퍼파라미터 설정
# ----------------------------
env_id = "CliffWalking-v0"
num_episodes = 500        # 학습 에피소드 수
alpha = 0.1               # 학습률
gamma = 0.99              # 감가율
epsilon = 1.0             # 초기 epsilon
epsilon_min = 0.01        # 최소 epsilon
epsilon_decay = 0.995     # 에피소드마다 epsilon을 곱해 줄 비율
max_steps_per_episode = 200  # 각 에피소드 최대 스텝(안정적 학습을 위해 설정)

# 환경 생성 (Gym 0.26 이상 기준: render_mode='rgb_array' 가능)
env = gym.make(env_id)

# Q 테이블 초기화
Q = np.zeros((env.observation_space.n, env.action_space.n))

def epsilon_greedy_action(state, Q, epsilon):
    """epsilon-greedy 정책으로 액션 선택"""
    if np.random.rand() < epsilon:
        return env.action_space.sample()  # 랜덤 액션
    else:
        return np.argmax(Q[state])

# ----------------------------
# 2. Q-learning으로 학습
# ----------------------------
for episode in range(num_episodes):
    state, _ = env.reset()  # Gym 0.26+에서는 (state, info) 리턴
    done = False

    for t in range(max_steps_per_episode):
        action = epsilon_greedy_action(state, Q, epsilon)
        next_state, reward, done, truncated, info = env.step(action)

        # Q-learning 업데이트
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )
        
        state = next_state
        
        if done or truncated:
            break
    
    # epsilon 감소
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

env.close()

# ----------------------------
# 3. 학습된 정책으로 시연 & 동영상 저장
# ----------------------------
# 학습이 끝난 후, 최적 정책(= argmax Q)을 이용해서 1 에피소드 실행
# 실행 과정에서 매 스텝의 프레임을 저장 후 mp4로 만든다.

# 렌더링을 위해 새 환경 생성 (render_mode='rgb_array'로 프레임을 얻는다)
test_env = gym.make(env_id, render_mode='rgb_array')

frames = []
state, _ = test_env.reset()
done = False

for _ in range(max_steps_per_episode):
    # 현재 상태에서 Q가 가장 높은 액션 선택
    action = np.argmax(Q[state])
    next_state, reward, done, truncated, info = test_env.step(action)

    # 프레임 수집 (Gym 0.26+에서는 render_mode='rgb_array'로 만든 env에 대해)
    frame = test_env.render()
    frames.append(frame)  # frame: (H, W, 3) 형식의 NumPy 배열
    
    state = next_state
    if done or truncated:
        break

test_env.close()

# frames를 mp4로 저장
# 만약 'moviepy' 대신 'imageio'를 이용하려면 아래와 같이 사용
output_filename = "cliffwalking_result.mp4"
imageio.mimsave(output_filename, frames, fps=30)

print(f"학습 결과 동영상을 '{output_filename}' 파일로 저장했습니다.")
