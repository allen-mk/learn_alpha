
import torch
import torch.nn as nn
import torch.optim as optim


x = torch.rand((16, 3))
# y_true = (x[:, 0] + x[:, 1] + x[:, 2]).view(-1, 1)
y1 = x.sum(dim=1)
y2 = x.sum(dim=1, keepdim=True)
print(x.shape)
print(y1.shape)
print(y2.shape)
