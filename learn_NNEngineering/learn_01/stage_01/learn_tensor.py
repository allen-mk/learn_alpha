import torch

# 基础创建
a = torch.tensor([1, 2, 3])    # 从Python列表创建张量
b = torch.zeros((2, 3))         # 创建一个2x3的全零张量
c = torch.ones((2, 3))          # 创建一个2x3的全一张量
d = torch.rand((3, 3))         # 创建一个3x3的随机张量（元素在0到1之间均匀分布）

# GPU创建
e = torch.tensor([1,2,3], device="cuda")