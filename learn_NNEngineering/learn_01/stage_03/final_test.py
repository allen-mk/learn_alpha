import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(
    nn.Linear(3, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)

# 创建损失函数
criterion = nn.MSELoss()

# 创建优化
optimizer = optim.Adam(model.parameters(),lr= 0.01)

COUNT = 10000

def train():
    for step in range(COUNT):
        x = torch.rand(16,3)
        y_true = (x[:,0] + 2 * x[:,1] + 3 * x[:,2]).view(-1,1)
        y_step = model(x)
        # 损失
        loss = criterion(y_step,y_true)

        #
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"step{step} , loss = {loss.item():.8f}")


if __name__ == "__main__":
    train()

