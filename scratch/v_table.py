import numpy as np
from tqdm import tqdm

episode = 1000

map_size = (4,4)
value_table = np.zeros(map_size, dtype=float)
mc = True
for e in tqdm(range(episode)):
    done = False
    x, y = 0, 0
    reward = 0
    history = []
    while not done:
        direction = []
        if x > 0: direction.append((y, x-1))
        if y > 0: direction.append((y -1, x))
        if x < map_size[1]-1: direction.append((y, x+1))
        if y < map_size[0]-1: direction.append((y+1, x))
        
        v_list = []
        for y_, x_ in direction:
            v_list.append(value_table[y_][x_])
        
        e = sum([np.exp(i) for i in v_list])
        prob = [np.exp(i)/e for i in v_list]
        
        action = np.random.choice(len(prob), p=prob)
        ny, nx = direction[action]
        
        history.append((y, x, ny,nx))
        if (ny, nx) == (map_size[0]-1, map_size[1]-1):
            if mc:
                G = 0
                for h in history[::-1]:
                    y, x, ny, nx = h
                    G -= 1
                    value_table[y][x] = G*0.1 + 0.9*value_table[y][x]
            break
        else:
            if mc == False:
                target = -1 + value_table[ny][nx]
                value_table[y][x] = target + 0.1*value_table[y][x]
        
        y, x = ny, nx
        
np.set_printoptions(precision=2, suppress=True)
print(value_table)
        
step = 0
x, y = 0, 0
while not done:
    direction = []
    if x > 0: direction.append((y, x-1))
    if y > 0: direction.append((y -1, x))
    if x < map_size[1]-1: direction.append((y, x+1))
    if y < map_size[0]-1: direction.append((y+1, x))
    
    v_list = []
    for y_, x_ in direction:
        v_list.append(value_table[y_][x_])
    
    e = sum([np.exp(i) for i in v_list])
    prob = [np.exp(i)/e for i in v_list]
    
    max_val = np.max(prob)
    candidates = [i for i, v in enumerate(prob) if v == max_val]
    action = np.random.choice(candidates)
    y, x = direction[action]
    
    # print(y,x)
    step += 1
    if (y, x) == (map_size[0]-1, map_size[1]-1):
        if step == map_size[0] + map_size[1] -2 :
            print('clear!')
        break

