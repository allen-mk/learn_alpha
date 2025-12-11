# 极小的神经网络

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, board_size: int = 3):
        super().__init__()
        self.board_size = board_size
        num_cells = board_size * board_size
        hidden = 64

        self.fc1 = nn.Linear(num_cells, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

        self.policy = nn.Linear(hidden, num_cells)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        p = F.log_softmax(self.policy(x), dim=-1)
        v = torch.tanh(self.value(x))
        return p, v
