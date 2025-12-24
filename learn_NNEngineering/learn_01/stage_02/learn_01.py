import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(3, 5),
    nn.ReLU(),
    nn.Linear(5, 2)
)

# x = torch.rand(3)
x = torch.tensor([1.0, 2.0, 3.0])
y = model(x)
print(y)

    