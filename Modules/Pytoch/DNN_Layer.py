import torch.nn as nn


class DNN(nn.Module):
    def __init__(self, input_size, activation, action_space, hidden):
        super(DNN, self).__init__()

        self.activation = activation
        self.action_space = action_space

        self.layers = nn.ModuleList()
        cur_dim = input_size

        hidden_count = len(hidden)

        for i in range(hidden_count):
            self.layers.append(nn.Sequential(nn.Linear(cur_dim, hidden[i]),
                                             nn.ReLU()))
            cur_dim = hidden[i]
        self.layers.append(nn.Linear(cur_dim, self.action_space))

    def forward(self, data):
        x = data
        for layer in self.layers[:-1]:
            x = layer(x)

        x = self.layers[-1](x)

        if self.activation is not None:
            x = self.activation(x)

        return x