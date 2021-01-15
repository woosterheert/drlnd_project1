import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    Simple DQN network using only a linear feedforward network with Relu activation
    and no activation after the final layer.
    Linear layers to be defined as input
    """
    def __init__(self, obs_space_size, action_space_size, seed, hidden_layers):
        super(DQN, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.layers = nn.ModuleList([nn.Linear(obs_space_size, hidden_layers[0])])
        self.layers.extend([nn.Linear(hidden_layers[i-1], hidden_layers[i]) for i in range(1, len(hidden_layers))])
        self.layers.append(nn.Linear(hidden_layers[-1], action_space_size))

    def forward(self, state):
        x = F.relu(self.layers[0](state))
        for i in range(1, len(self.layers)-1):
            x = F.relu(self.layers[i](x))
        return self.layers[-1](x)


class DQN_CNN(nn.Module):
    """
    Simple CNN network using two conv-conv-pull architectures followed
    by three linear layers with, the first two with relu activation.
    """
    def __init__(self, obs_space_size, action_space_size, seed):
        super(DQN_CNN, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.conv1 = nn.Conv2d(3, 8, 8, 1)
        self.conv2 = nn.Conv2d(8, 8, 8, 1)
        self.pool = nn.MaxPool2d(2, 2, 1)
        self.conv3 = nn.Conv2d(8, 8, 8, 1)
        self.conv4 = nn.Conv2d(8, 8, 8, 1)
        self.fc1 = nn.Linear(8 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, action_space_size)

    def forward(self, state):
        x = self.conv1(state)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = x.view(-1, 8 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
