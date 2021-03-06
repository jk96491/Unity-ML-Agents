import torch
import torch.nn as nn
from Modules.Pytoch.CNN_Layer import CNN
from Modules.Pytoch.DNN_Layer import DNN


class visual_obs_critic(nn.Module):
    def __init__(self, args, action_space, learning_rate, device, env_info, hidden):
        super(visual_obs_critic, self).__init__()
        self.args = args
        self.learning_rate = learning_rate
        self.device = device
        self.env_info = env_info
        self.action_space = action_space
        self.hidden = hidden

        self.cnnLayer = CNN(self.env_info)
        self.DnnLayer = DNN(28 * 256, None, self.action_space, self.hidden)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        self.to(self.device)

    def forward(self, obs):
        obs = (obs - (255.0 / 2)) / (255.0 / 2)
        obs.to(self.device)
        x = self.cnnLayer(obs)

        x = x.view(x.size(0), -1)

        q_val = self.DnnLayer(x)

        return q_val

    def predict(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)
        q_val = self.forward(obs)

        return q_val

    def get_loss(self, states, td_targets):
        states = torch.FloatTensor(states).squeeze(1)
        td_target = torch.stack(td_targets, dim=0)

        predict = self.predict(states)
        loss = torch.mean((predict - td_target) ** 2)

        return loss

    def Learn(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))


class vector_obs_critic(nn.Module):
    def __init__(self, state_size, action_size, learning_rate, device):
        super(vector_obs_critic, self).__init__()
        self.learning_rate = learning_rate
        self.state_size = state_size
        self.action_size = action_size

        self.device = device

        self.layer1 = nn.Sequential(nn.Linear(state_size, 128),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(130, 128),
                                    nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(128, 128),
                                    nn.ReLU())
        self.layer4 = nn.Sequential(nn.Linear(128, action_size))

        self.to(self.device)

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
