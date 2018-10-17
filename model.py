import torch
import torch.nn as nn
import torch.nn.functional as F


class BananaModel(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc_state = nn.Linear(128, 1)
        self.fc_action = nn.Linear(128, action_size)

    def forward(self, state):
        X = self.fc1(state)
        X = F.elu(X)
        state = self.fc2(X)
        state = F.elu(state)
        state = self.fc_state(state)
        action = self.fc3(X)
        action = F.elu(action)
        action = self.fc_action(action)
        action = action / action.mean(-1, keepdim=True)
        return state + action


class BananaPixelsModel(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        h, w, c = state_size
        h_o = h
        self.conv1 = nn.Conv2d(c, 16, (5, 5))
        h_o = h_o // 2
        self.conv2 = nn.Conv2d(16, 32, (5, 5))
        h_o = h_o // 2
        self.conv3 = nn.Conv2d(32, 32, (5, 5))
        h_o = h_o // 2
        self.conv4 = nn.Conv2d(32, 32, (5, 5))
        h_o = h_o // 2
        self.fc1 = nn.Linear(h_o*h_o*32, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc_state = nn.Linear(32, 1)
        self.fc_action = nn.Linear(32, action_size)

    def forward(self, state):
        X = self.conv1(state)
        X = F.elu(X)
        X = F.max_pool2d(X, (2,2), (2,2))
        X = self.conv2(X)
        X = F.elu(X)
        X = F.max_pool2d(X, (2,2), (2,2))
        X = self.conv3(X)
        X = F.elu(X)
        X = F.max_pool2d(X, (2,2), (2,2))
        X = self.conv4(X)
        X = F.elu(X)
        X = F.max_pool2d(X, (2,2), (2,2))
        size = 1
        for dim in X.size()[1:]:
            size *= dim
        X = X.view((-1, size))
        X = self.fc1(X)
        X = F.elu(X)
        state = self.fc2(X)
        state = F.elu(state)
        state = self.fc_state(state)
        action = self.fc3(X)
        action = F.elu(action)
        action = self.fc_action(action)
        action = action / action.mean(-1, keepdim=True)
        return state + action
