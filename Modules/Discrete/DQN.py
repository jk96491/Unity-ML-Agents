import torch
import torch.nn as nn
from Modules.CNN_Layer import CNN
import torch.nn.functional as F
import Utils
import numpy as np


class visual_obs_dqn(nn.Module):
    def __init__(self, action_space, learning_rate, device):
        super(visual_obs_dqn, self).__init__()
        self.learning_rate = learning_rate
        self.action_space = action_space
        self.device = device

        self.cnnLayer = CNN()

        self.fc1 = nn.Linear(420 * 256, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, self.action_space)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.to(self.device)

    def forward(self, obs):
        obs = (obs - (255.0 / 2)) / (255.0 / 2)
        obs = obs.to(self.device)
        x = self.cnnLayer(obs)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_val = self.fc3(x)
        return q_val

    def get_action(self, obs):
        obs = torch.FloatTensor(obs)
        obs = obs.to(self.device)
        q_val = self.forward(obs)
        action = np.asscalar(torch.argmax(q_val[0]).detach().cpu().clone().numpy())
        return action

    def Learn(self, target_model, train_batch, dis):
        Q_val_List = []
        Q_target_val_List = []

        for state, action, reward, next_state, done in train_batch:
            reward = torch.FloatTensor(reward)
            q_val = self.forward(torch.FloatTensor(state))
            target_q_val = target_model.forward(torch.FloatTensor(next_state))

            maxQ1 = torch.max(q_val.data)

            if done:
                q_val[0, action] = reward
            else:
                q_val[0, action] = reward + torch.mul(maxQ1, dis)

            Q_val_List.append(q_val)
            Q_target_val_List.append(target_q_val)

        Q_val_List = torch.stack(Q_val_List).squeeze(1)
        Q_target_val_List = torch.stack(Q_target_val_List).squeeze(1)
        loss = torch.mean((Q_val_List - Q_target_val_List) ** 2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss


class vector_obs_dqn(nn.Module):
    def __init__(self, obs_size, action_space, learning_rate, device):
        super(vector_obs_dqn, self).__init__()
        self.learning_rate = learning_rate
        self.obs_size = obs_size
        self.device = device

        self.fc1 = nn.Sequential(nn.Linear(obs_size, 128),
                                 nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(128, 128),
                                 nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(128, action_space))

        self.to(device)

    def forward(self, obs):
        obs = Utils.convertToTensorInput(obs, self.obs_size, obs.shape[0])
        x = self.fc1(obs)
        x = self.fc2(x)
        q = self.fc3(x)

        return q