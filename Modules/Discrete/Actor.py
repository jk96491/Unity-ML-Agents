import torch
import torch.nn as nn
import torch.nn.functional as F
from Modules.CNN_Layer import CNN
import Utils


class visual_obs_actor(nn.Module):
    def __init__(self, action_space, learning_rate, device):
        super(visual_obs_actor, self).__init__()
        self.learning_rate = learning_rate
        self.action_space = action_space
        self.device = device

        self.cnnLayer = CNN()

        self.fc1 = nn.Linear(779 * 256, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Sequential(nn.Linear(128, self.action_space ),
                                 nn.Softmax())

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, obs):
        obs = (obs - (255.0 / 2)) / (255.0 / 2)
        obs = obs.to(self.device)
        x = self.cnnLayer(obs)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action = self.fc3(x)
        return action

    def get_action(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)
        action = self.forward(obs)
        return action

    def Learn(self, obs, actions, advantages):
        advantages = torch.FloatTensor(advantages).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        advantages = torch.gather(advantages, dim=1, index=actions)

        policy = self.get_action(obs)
        log_policy = torch.log(policy)

        loss = torch.mean(- log_policy * advantages)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))


class vector_obs_actor(nn.Module):
    def __init__(self, obs_size, action_space, learning_rate):
        super(vector_obs_actor, self).__init__()
        self.learning_rate = learning_rate
        self.obs_size = obs_size

        self.fc1 = nn.Sequential(nn.Linear(obs_size, 128),
                                 nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(128, 128),
                                 nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(128, action_space),
                                 nn.Softmax())

    def forward(self, obs):
        obs = Utils.convertToTensorInput(obs, self.obs_size, obs.shape[0])
        x = self.fc1(obs)
        x = self.fc2(x)
        q = self.fc3(x)

        return q

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))



