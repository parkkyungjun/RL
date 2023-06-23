import numpy as np
import gym # 0.25.2
import matplotlib.pyplot as plt

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()

# Define Q-learning function
def QLearning(env, learning, discount, epsilon, min_eps, episodes):
    # Determine size of discretized state space
    num_states = (env.observation_space.high - env.observation_space.low)*\
                    np.array([10, 50])
    num_states = np.round(num_states, 0).astype(int) + 1

    # Initialize Q table
    Q = np.random.uniform(low = -1, high = 0,
                          size = (num_states[0], num_states[1],
                                  env.action_space.n))

    # Initialize variables to track rewards
    reward_list = []
    ave_reward_list = []

    eps1 = epsilon

    # Calculate episodic reduction in epsilon
    first = episodes + 1

    # Run Q learning algorithm
    for i in range(episodes):
        # Initialize parameters
        done = False
        tot_reward, reward = 0,0
        state = env.reset()

        # Discretize state
        state_adj = (state - env.observation_space.low)*np.array([10, 50])
        state_adj = np.round(state_adj, 0).astype(int)
        
        while done != True:
            # Render environment for last five episodes
            if i >= (episodes - 5):
                env.render()

            # Determine next action - epsilon greedy strategy
            if np.random.random() < 1 - epsilon:
                action = np.argmax(Q[state_adj[0], state_adj[1]])
            else:
                action = np.random.randint(0, env.action_space.n)

            # Get next state and reward
            state2, reward, done, info = env.step(action)
            
            reward = newreward(state2[0])
            # Discretize state2
            state2_adj = (state2 - env.observation_space.low)*np.array([10, 50])
            state2_adj = np.round(state2_adj, 0).astype(int)

            #Allow for terminal states
            if done and state2[0] >= 0.5:
                Q[state_adj[0], state_adj[1], action] = reward

            # Adjust Q value for current state
            else:
                delta = learning*(reward +
                                 np.max(Q[state2_adj[0],
                                                   state2_adj[1]]) -
                                 Q[state_adj[0], state_adj[1],action])
                Q[state_adj[0], state_adj[1],action] += delta

            if state2[0]>= 0.5 and i<first:
              first=i
              print('clear first time : ', i)
            # Update variables
            tot_reward += reward
            state_adj = state2_adj
              

        # Decay epsilon
        if epsilon > min_eps:
            epsilon *= eps1

        # Track rewards
        reward_list.append(tot_reward)

        if (i+1) % 100 == 0:
            ave_reward = np.mean(reward_list)
            ave_reward_list.append(ave_reward)
            reward_list = []
            print('Episode {} Average Reward: {}'.format(i+1, ave_reward))

    env.close()
    return ave_reward_list, Q

def newreward(x):
  if x >= 0.5:
    return 2
  else:
    return (x + 1.2)/1.8 - 1

# Run Q-learning algorithm
rewards, Q = QLearning(env, 0.2, 0.9, 0.8, 0, 1000)

# env = gym.make("MountainCar-v0", render_mode='human')
# for i in range(3):
#     state = env.reset()

#     state_adj = (state - env.observation_space.low)*np.array([10, 50])
#     state_adj = np.round(state_adj, 0).astype(int)
#     done = False

#     while done != True:
#         action = np.argmax(Q[state_adj[0], state_adj[1]])
#         state2, reward, done, info = env.step(action)

#         # Discretize state2
#         state2_adj = (state2 - env.observation_space.low)*np.array([10, 50])
#         state2_adj = np.round(state2_adj, 0).astype(int)

#         state_adj = state2_adj

# Plot Rewards
# plt.plot(100*(np.arange(len(rewards)) + 1), rewards)
# plt.xlabel('Episodes')
# plt.ylabel('Average Reward')
# plt.title('Average Reward vs Episodes')
# plt.savefig('rewards.jpg')
# plt.close()