import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

TABLE_SIZE = [8, 8]
EPISODE = 1000
MAX_STEPS = 200 # [추가] 한 에피소드 당 최대 시도 횟수 제한

# 장치 설정: 이 정도 규모는 무조건 CPU가 빠릅니다.
device = 'cpu' 

class Q_manifold(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = TABLE_SIZE[0] * TABLE_SIZE[1]
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 4) 
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x
        
model = Q_manifold().to(device)
frozen = Q_manifold().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
frozen.load_state_dict(model.state_dict())
    
for epi in tqdm(range(EPISODE)):
    TABLE = np.zeros(TABLE_SIZE, dtype=np.float32)
    y, x = 0, 0
    TABLE[y][x] = 1
    
    direction = [(1,0), (0,1), (-1,0), (0,-1)]
    
    # [수정] 스텝 제한을 위한 변수
    step_count = 0 
    
    while step_count < MAX_STEPS: # [수정] 무한루프 방지
        step_count += 1
        
        # CPU 연산이므로 .to('cuda') 제거
        input_tensor = torch.tensor(TABLE.flatten(), device=device)
        
        with torch.no_grad():
            # CPU에서 바로 연산하므로 .to('cpu') 불필요
            value = model(input_tensor).numpy()
        
        exp_v = np.exp(value - np.max(value))
        e = sum(exp_v)
        prob = exp_v / e
        action = np.random.choice(len(prob), p=prob)
        
        ny = y + direction[action][0]
        nx = x + direction[action][1]
        
        # Gradient 추적 필요
        q = model(input_tensor)[action]
        
        if nx >= 0 and ny >= 0 and nx < TABLE_SIZE[1] and ny < TABLE_SIZE[0]:
            TABLE[y][x] = 0
            TABLE[ny][nx] = 1
            
            # 다음 상태
            next_input = torch.tensor(TABLE.flatten(), device=device)
            
            with torch.no_grad():
                value = frozen(next_input)
                value2 = model(next_input)
            
            nq = value2[torch.argmax(value).item()] # max라고 말한애의 값을 그대로 쓰기 보다 그 가치의 인덱스로 다른 네트워크에서 값을 뽑아쓰면 둘다 동시에 컸을때만 maximization bias가 생기니 더 나은방법으로 평가됨
            r = -1
            target = r*0.1 + nq*0.9
        else:
            target = torch.tensor(-10, dtype=torch.float32, device=device)
        
        if [ny+1, nx+1] == TABLE_SIZE:
            target = torch.tensor(10, dtype=torch.float32, device=device)
        
        optimizer.zero_grad()
        loss = F.mse_loss(q, target)
        loss.backward()
        optimizer.step()
        
        if nx >= 0 and ny >= 0 and nx < TABLE_SIZE[1] and ny < TABLE_SIZE[0]:
            y, x = ny, nx
        
        if [ny+1, nx+1] == TABLE_SIZE:
            break
    
    frozen.load_state_dict(model.state_dict())
    

TABLE = np.zeros(TABLE_SIZE, dtype=np.float32)
y, x = 0, 0
TABLE[y][x] = 1
step_count = 0
model.eval()

while step_count < MAX_STEPS: # [수정] 무한루프 방지
    step_count += 1
    
    # CPU 연산이므로 .to('cuda') 제거
    input_tensor = torch.tensor(TABLE.flatten(), device=device)
    
    with torch.no_grad():
        # CPU에서 바로 연산하므로 .to('cpu') 불필요
        value = model(input_tensor).numpy()
    
    action = np.argmax(value)
    
    ny = y + direction[action][0]
    nx = x + direction[action][1]
    
    # Gradient 추적 필요
    q = model(input_tensor)[action]
    
    if nx >= 0 and ny >= 0 and nx < TABLE_SIZE[1] and ny < TABLE_SIZE[0]:
        TABLE[y][x] = 0
        TABLE[ny][nx] = 1
    else:
        print('fail')
        break
    
    if nx >= 0 and ny >= 0 and nx < TABLE_SIZE[1] and ny < TABLE_SIZE[0]:
        y, x = ny, nx
    
    if [ny+1, nx+1] == TABLE_SIZE:
        break
    
print('success' if step_count == sum(TABLE_SIZE)-2 else 'fail', step_count)