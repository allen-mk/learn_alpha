# 阶段 B：从“表格” → 神经网络（真正 RL）
# 现在问题来了：
# 状态太多，表格放不下怎么办？
# 答案：
# 用神经网络当 Q 表

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


T = 2
LR = 0.1
GAMMA = 0.9

class Q_TABLE():
    def __init__(self, target, lr, gamma):
        self.target = target
        self.lr = lr
        self.gamma = gamma
        self.Q = np.zeros((target+1, 2))

    def step(self,s,a):
        s_next = max(0, min(self.target, s + (1 if a == 1 else -1)))
        reward = 1 if s_next == self.target else 0
        td_target = reward + GAMMA * max(self.Q[s_next])
        td_error = td_target - self.Q[s][a]
        self.Q[s][a] += LR * td_error


    def printQ(self, prefix):
        print(prefix + "\n", self.Q)


class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        x = self.model(x)
        return x

class QAgent():
    def __init__(self, target, lr, gamma):
        self.target = target
        self.lr = lr
        self.gamma = gamma
        self.Q = QNet()
        self.optimizer = optim.Adam(self.Q.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def step(self,s,a):
        with torch.no_grad():
            s_next = max(0, min(self.target, s + (1 if a == 1 else -1)))
            reward = 1 if s_next == self.target else 0
            s_next = torch.tensor([[float(s_next)]])
            target = reward + GAMMA * torch.max(self.Q(s_next))
        pred = self.Q(s)[0,a]
        loss = self.loss_fn(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def printQ(self, prefix):
        print(prefix + "\n")
        with torch.no_grad():
            for i in range(self.target+1):
                q = self.Q(torch.tensor([[float(i)]]))
                print(f"s={i}, q={q.numpy()}")


def main1():
    print("========================== main1")
    q_table = Q_TABLE(T, LR, GAMMA)
    s = 1
    a = 1
    q_table.printQ("Before")
    q_table.step(s, a)
    q_table.printQ("After")

def main2():
    print("========================== main2")
    q_agent = QAgent(T, LR, GAMMA)
    s = torch.tensor([[1.0]])
    a = 1
    # print("Before : \n", q_table.Q)
    q_agent.printQ("Before")
    q_agent.step(s, a)
    # print("After : \n", q_table.Q)
    q_agent.printQ("After")



if __name__ == "__main__":
    main1()
    main2()