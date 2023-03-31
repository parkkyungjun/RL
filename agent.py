import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import QTrainer, CNN_QNet
from helper import plot
import os

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = CNN_QNet(3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        map = [[0 for i in range(-1, 17)] for j in range(-1, 17)]

        # if not game.game_over:
        for m in game.snake:
                map[int(m.y//20 + 1)][int(m.x//20 + 1)] = 1

        map[int(game.snake[0].y//20 + 1)][int(game.snake[0].x//20 + 1)] = 2
        map[int(game.snake[-1].y//20 + 1)][int(game.snake[-1].x//20 + 1)] = 3

        map[int(game.food.y//20 + 1)][int(game.food.x//20 + 1)] = 4

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        dir = [dir_l, dir_r, dir_u, dir_d]
        # map.append(dir_l)
        # map.append(dir_r)
        # map.append(dir_u)
        # map.append(dir_d)

        return np.array(map, dtype=int), np.array(dir, dtype=int)

    def remember(self, state, action, reward, next_state, done, direction, direction_new):
        self.memory.append((state, action, reward, next_state, done, direction, direction_new)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones, directions, direction_news = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones, directions, direction_news)
        # for state, action, reward, next_state, done, direction, direction_new in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done, direction, direction_new)

    def train_short_memory(self, state, action, reward, next_state, done, direction, direction_new):
        self.trainer.train_step(state, action, reward, next_state, done, direction, direction_new)

    def get_action(self, state, direction):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 500 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 2000) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            direction0 = torch.tensor(direction, dtype=torch.float)

            state0 = torch.unsqueeze(state0, 0)
            direction0 = torch.unsqueeze(direction0, 0)

            prediction = self.model(state0, direction0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old, direction = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old, direction)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new, direction_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done, direction, direction_new)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done, direction, direction_new)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                # agent.model.save(n_games=agent.n_games, optimizer=agent.trainer.optimizer)

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()