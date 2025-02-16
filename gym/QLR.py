import argparse
import yaml
import gym
import numpy as np
import imageio
import os
import math

# import code; code.interact(local=locals())

def train_q_learning(
    env_id: str,
    num_episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    epsilon_min: float,
    epsilon_decay: float,
    max_steps_per_episode: int,
    demo_episodes: int,
    render_fps: int
):
    env = gym.make(env_id)
    
    if env_id == 'Blackjack-v1':
        n_states = [env.observation_space[i].n for i in range(len(env.observation_space))]
        n_actions = env.action_space.n
        
        Q = np.zeros(n_states + [n_actions])
    elif env_id == 'MountainCar-v0':
        num_states = (env.observation_space.high - env.observation_space.low)*np.array([10, 100])
        num_states = np.round(num_states, 0).astype(int) + 1

        Q = np.random.uniform(low = -1, high = 0, size = (num_states[0], num_states[1], env.action_space.n))
    else:
        n_states = env.observation_space.n
        n_actions = env.action_space.n

        Q = np.zeros((n_states, n_actions))

    def epsilon_greedy_action(state, Q, epsilon):
        if np.random.rand() < epsilon:
            return env.action_space.sample()
        else:
            return np.argmax(Q[state])

    rewards = []
    max_steps_per_episode = 9999999 if max_steps_per_episode == 'inf' else max_steps_per_episode

    for episode in range(num_episodes):
        state, _ = env.reset()  
        done = False

        if env_id == 'Blackjack-v1':
            state = (state[0], state[1], int(state[2]))
        elif env_id == 'MountainCar-v0':
            state = (state - env.observation_space.low)*np.array([10, 100])
            state = tuple(np.round(state, 0).astype(int))

        for t in range(max_steps_per_episode):
            action = epsilon_greedy_action(state, Q, epsilon)
            next_state, reward, done, truncated, info = env.step(action)

            if env_id == 'Blackjack-v1':
                next_state = (next_state[0], next_state[1], int(next_state[2]))
            elif env_id == 'MountainCar-v0':
                if next_state[0] >= 0.5:
                    reward = 2
                else:
                    reward = (next_state[0] + 1.2)/1.8 - 1

                next_state = (next_state - env.observation_space.low)*np.array([10, 100])
                next_state = tuple(np.round(next_state, 0).astype(int))

                epsilon *= epsilon_decay
        
                if done and next_state[0] >= 0.5:
                    Q[state + (action,)] = reward
                else:
                    delta = alpha*(reward + np.max(Q[next_state]) - Q[state + (action,)])
                    Q[state + (action,)] += delta
            else:
                epsilon = max(epsilon - epsilon_decay, epsilon_min)

            if env_id != 'MountainCar-v0':
                if done or truncated:
                    td_target = reward
                else:
                    td_target = reward + gamma * np.max(Q[next_state])

                Q[state, action] += alpha * (td_target - Q[state, action])

            state = next_state

            if done or truncated:
                break

        rewards.append(reward)
        
        if episode % 1000 == 0:
            print((sum(rewards[-100:]) / 100) * 100)

    env.close()

    test_env = gym.make(env_id, render_mode='rgb_array')
    frames = []
    
    for demo_ep in range(demo_episodes):
        state, _ = test_env.reset()

        if env_id == 'Blackjack-v1':
            state = (state[0], state[1], int(state[2]))
        elif env_id == 'MountainCar-v0':
            state = (state - env.observation_space.low)*np.array([10, 100])
            state = tuple(np.round(state, 0).astype(int))

        while True:
            action = np.argmax(Q[state])
            next_state, reward, done, truncated, info = test_env.step(action)

            if env_id == 'Blackjack-v1':
                next_state = (next_state[0], next_state[1], int(next_state[2]))
            elif env_id == 'MountainCar-v0':
                next_state = (next_state - env.observation_space.low)*np.array([10, 100])
                next_state = tuple(np.round(next_state, 0).astype(int))
                
            frame = test_env.render()  
            frames.append(frame)
            state = next_state
            if done or truncated:
                break

        for _ in range(2 + render_fps // 2):
            frames.append(frame)
    
    test_env.close()
    
    os.makedirs(env_id, exist_ok=True)
    imageio.mimsave(os.path.join(env_id, env_id + '.mp4'), frames, fps=render_fps)

    imageio.mimsave(os.path.join(env_id, env_id + '.gif'), frames, fps=render_fps)
    print(f"[{env_id}] 학습 완료! {demo_episodes} 에피소드 시연 영상을 '{env_id + 'mp4'}'로 저장했습니다.")

    return Q


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Taxi-v3", help="학습할 Gym 환경 ID")
    parser.add_argument("--config", type=str, default="QLR.yaml", help="하이퍼파라미터가 정리된 YAML 파일 경로")
    args = parser.parse_args()

    # UTF-8 인코딩으로 YAML 파일 로딩
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # environments 섹션에서 해당 env ID를 찾기
    env_params = config["environments"].get(args.env)
    if env_params is None:
        raise ValueError(
            f"지정한 env [{args.env}]가 {args.config} 내에 없습니다. "
            f"가능한 환경: {list(config['environments'].keys())}"
        )
    
    num_episodes = env_params["num_episodes"]
    alpha = env_params["alpha"]
    gamma = env_params["gamma"]
    epsilon = env_params["epsilon"]
    epsilon_min = env_params["epsilon_min"]
    epsilon_decay = env_params["epsilon_decay"]
    max_steps_per_episode = env_params["max_steps_per_episode"]
    demo_episodes = env_params["demo_episodes"]
    render_fps = env_params['render_fps']
    
    train_q_learning(
        env_id=args.env,
        num_episodes=num_episodes,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        max_steps_per_episode=max_steps_per_episode,
        demo_episodes=demo_episodes,
        render_fps=render_fps,
    )
