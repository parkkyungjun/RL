import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class CNN_QNet(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.linear = nn.Linear(256, output_size)
        # self.linear2 = nn.Linear(4, 64)
        # self.linear3 = nn.Linear(64, output_size)
        # self.ln1 = nn.LayerNorm(64)
        # self.ln2 = nn.LayerNorm(64)
    def forward(self, x, d):
        s = x.shape
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(s[0], -1)
        # x = F.relu(self.ln1(self.linear(x)) + self.ln2(self.linear2(d)))
        return F.softmax(self.linear(x), 1)

    def save(self, file_name='model.pth', n_games=0, optimizer=None):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save({
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'n_games': n_games,
                    }, file_name)
        # torch.save(self.state_dict(), file_name)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.model = model
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done, direction, direction_new):
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        direction = torch.tensor(direction, dtype=torch.float)
        direction_new = torch.tensor(direction_new, dtype=torch.float)

        if len(state.shape) == 2:
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)
            direction = torch.unsqueeze(direction, 0)
            direction_new = torch.unsqueeze(direction_new, 0)
            

        # 1. predicted Q values with current state
            pred = self.model(state, direction) # shape : (1, 3), (2, 3) 

            target = pred.clone()
            Q_new = reward
            if not done:
                Q_new = reward + self.gamma * torch.max(self.model(next_state, direction_new))
                
            target[0][torch.argmax(action).item()] = Q_new

        else:
            state = torch.unsqueeze(state, 1)
            pred = self.model(state, direction) # shape : (1, 3), (2, 3) 

            target = pred.clone()
            done = (done, )
            for idx in range(len(done)):
                Q_new = reward[idx]
                if not done[idx]:
                    n = torch.unsqueeze(next_state[idx], 0)
                    d = torch.unsqueeze(direction_new[idx], 0)
                    Q_new = reward[idx] + self.gamma * torch.max(self.model(n, d))
                    
                target[idx][torch.argmax(action[idx]).item()] = Q_new
        # 2. Q_new = r + y * max(next_predicted Q value)
        # pred.clone()
        # preds[argmax(action)] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()

        