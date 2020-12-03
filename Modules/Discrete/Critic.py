import torch
import torch.nn as nn
import torch.nn.functional as F
from Modules.CNN_Layer import CNN
import Utils


class visual_obs_critic(nn.Module):
    def __init__(self, action_space, learning_rate, device):
        super(visual_obs_critic, self).__init__()
        self.learning_rate = learning_rate
        self.device = device

        self.cnnLayer = CNN()

        self.fc1 = nn.Linear(779 * 256, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, action_space)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, obs):
        obs = (obs - (255.0 / 2)) / (255.0 / 2)
        obs.to(self.device)
        x = self.cnnLayer(obs)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_val = self.fc3(x)

        return q_val

    def predict(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)
        q_val = self.forward(obs)
        return q_val

    def Learn(self, states, td_target):
        predict = self.predict(states)
        loss = torch.mean((predict - td_target) ** 2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))




class vector_obs_critic(nn.Module):
    def __init__(self, state_size, action_size, learning_rate):
        super(vector_obs_critic, self).__init__()
        self.learning_rate = learning_rate
        self.state_size = state_size
        self.action_size = action_size

        self.layer1 = nn.Sequential(nn.Linear(state_size, 128),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(130, 128),
                                    nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(128, 128),
                                    nn.ReLU())
        self.layer4 = nn.Sequential(nn.Linear(128, 1))

    def forward(self, obs, action):
        obs = convertToTensorInput(obs, self.obs_size, obs.shape[0])
        x = self.layer1(obs)

        x = torch.cat([x, action], dim=-1)

        x = self.layer2(x)
        x = self.layer3(x)
        q_val = self.layer4(x)

        return q_val

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))
