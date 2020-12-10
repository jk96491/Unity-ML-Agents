import torch
import torch.nn as nn
import torch.nn.functional as F
from Modules.Pytoch.CNN_Layer import CNN
import Utils
import numpy as np
from Modules.Pytoch.DNN_Layer import DNN


class visual_obs_actor(nn.Module):
    def __init__(self, action_space, learning_rate, device, env_info, hidden):
        super(visual_obs_actor, self).__init__()
        self.learning_rate = learning_rate
        self.action_space = action_space
        self.device = device
        self.env_info = env_info
        self.hidden = hidden

        self.cnnLayer = CNN(env_info)
        self.DnnLayer = DNN(420 * 256, nn.Softmax(), self.action_space, self.hidden)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.to(self.device)

    def forward(self, obs):
        obs = (obs - (255.0 / 2)) / (255.0 / 2)
        obs = obs.to(self.device)
        x = self.cnnLayer(obs)
        x = x.view(x.size(0), -1)
        action = self.DnnLayer(x)
        return action

    def get_action(self, obs):
        obs = torch.FloatTensor(obs)
        obs = obs.to(self.device)
        action = self.forward(obs)
        action = np.asscalar(torch.argmax(action[0]).detach().cpu().clone().numpy())
        return action

    def Learn(self, obs, actions, advantages):
        obs = torch.FloatTensor(obs).squeeze(1)
        advantages = torch.stack(advantages, dim=0)
        actions = torch.LongTensor(actions).to(self.device).squeeze(1)

        advantages = torch.gather(advantages.squeeze(1).to(self.device), dim=1, index=actions)

        policy = self.forward(obs)
        log_policy = torch.log(policy)

        loss = torch.mean(- log_policy * advantages.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save_weights(self, path):
        torch.save(self.state_dict(), '{0}.th'.format(path))

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))


class vector_obs_actor(nn.Module):
    def __init__(self, obs_size, action_space, learning_rate, device):
        super(vector_obs_actor, self).__init__()
        self.learning_rate = learning_rate
        self.obs_size = obs_size
        self.device = device

        self.fc1 = nn.Sequential(nn.Linear(obs_size, 128),

                                 nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(128, 128),
                                 nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(128, action_space),
                                 nn.Softmax())

        self.to(self.device)

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



