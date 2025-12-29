import numpy as np
from tqdm import tqdm

TABLE_SIZE = [8, 8]
EPISODE = 1000
Q_TABLE = np.zeros(TABLE_SIZE + [4], dtype=float)

for epi in tqdm(range(EPISODE)):
    y, x = 0, 0
    
    direction = [(1,0), (0,1), (-1,0), (0,-1)]
    while True:
        value = Q_TABLE[y][x][:]
        e = sum([np.exp(i) for i in value])
        prob = [np.exp(i) / e for i in value]
        action = np.random.choice(len(prob), p=prob)
        
        ny = y+direction[action][0]
        nx = x+direction[action][1]
        
        if nx >= 0 and ny >= 0 and nx < TABLE_SIZE[1] and ny < TABLE_SIZE[0]:
            nq = np.max(Q_TABLE[ny][nx])
        else:
            # continue
            nq = -100
        
        target = -1 + nq
        Q_TABLE[y][x][action] = 0.1*target + 0.9*Q_TABLE[y][x][action]
        
        if nx >= 0 and ny >= 0 and nx < TABLE_SIZE[1] and ny < TABLE_SIZE[0]:
            y, x = ny, nx
        
        if [ny+1, nx+1] == TABLE_SIZE:
            break

y, x = 0, 0
step = 0
direction = [(1,0), (0,1), (-1,0), (0,-1)]

while True:
    step += 1
    action = np.argmax(Q_TABLE[y][x])
    
    ny = y+direction[action][0]
    nx = x+direction[action][1]
    
    if nx >= 0 and ny >= 0 and nx < TABLE_SIZE[1] and ny < TABLE_SIZE[0]:
        y, x = ny, nx
    
    if [ny+1, nx+1] == TABLE_SIZE:
        break

print('success' if step == sum(TABLE_SIZE)-2 else 'fail')