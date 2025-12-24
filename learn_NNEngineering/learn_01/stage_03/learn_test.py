
# 请你自己写一份训练代码，任务如下：
# 🎯 训练目标
# y = 2*x1 - x2 + 0.5*x3
# 要求:
# 输入 3 维
# 输出 1 维
# 使用 MLP（至少 1 个 ReLU）
# 使用 MSELoss
# 使用 Adam
# 训练 1000 步
# 每 200 步打印 loss


import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


model = nn.Sequential(
    nn.Linear(3, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)

# 定义损失函数
# loss = (y_pred - y_true)^2
criterion = nn.MSELoss()

# 定义优化器
# model.parameters()：所有可训练参数（权重 + 偏置）
# Adam : 最常用、最稳定的优化器
# lr : 学习率(步子大小)
optimizer = optim.Adam(model.parameters(), lr=0.01)

COUNT = 10000

for step in range(COUNT):
    # 生成训练数据
    x = torch.rand((64, 3))                # batch size = 64

    y_true = (2*x[:, 0] - x[:, 1] + 0.5*x[:, 2]).view(-1, 1)

    y_pred = model(x)

    loss = criterion(y_pred, y_true)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(f"step {step}, loss = {loss.item():.8f}")

for p in model.parameters():
    print(p)
    