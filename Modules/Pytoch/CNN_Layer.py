import torch.nn as nn
import numpy as np


class CNN(nn.Module):
    def __init__(self, env_info):
        super(CNN, self).__init__()

        self.shape = np.shape(env_info.visual_observations[0])
        self.channel = self.shape[3]

        self.conv1 = nn.Conv2d(self.channel, 32, kernel_size=8, stride=4, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)

    def forward(self, data):
        x = self.conv1(data)
        x = self.conv2(x)
        x = self.conv3(x)

        return x