# 训练 = 不断修正权重，让模型输出更接近正确答案
# 参数 = 参数 - 学习率 × 梯度

# 我现在需要做一个目标函数 y = x1 + x2 + x3

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# 定义模型
model = nn.Sequential(
    nn.Linear(3, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)

# 定义损失函数  loss = (y_pred - y_true)^2
criterion = nn.MSELoss()

# 定义优化器
# model.parameters()：所有可训练参数（权重 + 偏置）
# Adam : 最常用、最稳定的优化器
# lr : 学习率(步子大小)
optimizer = optim.Adam(model.parameters(), lr=0.01)


# loss history
loss_history = []


# 训练主函数
def train():
    COUNT = 5000
    for step in range(COUNT):
        
        # 生成训练数据
        x = torch.rand((64, 3))                # batch size = 64

        # 计算正确答案 y_true = x1 + x2 + x3
        y_true = x.sum(dim=1, keepdim=True)    # 正确答案
        
        # 前向传播 进行预测 模型给出预测值
        y_pred = model(x)
        
        # 计算损失
        loss = criterion(y_pred, y_true)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            print(f"step {step}, loss = {loss.item():.8f}")
            loss_history.append(loss.item())

    #将训练的模型储存到文件中
    torch.save(model.state_dict(), "model.pth")

    plt.figure()
    plt.plot(loss_history)
    plt.yscale("log")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.show()
        



# 测试计算
def test():
    # 加载训练好的模型
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    
    # 计算 x1 + x2 + x3
    x1 = torch.tensor(1.0)
    x2 = torch.tensor(2.0)
    x3 = torch.tensor(3.0)
    y_pred = model(torch.stack([x1, x2, x3]))
    print(f"x1 + x2 + x3 = {y_pred.item():.6f}")

if __name__ == "__main__":
    train()
    # test()
