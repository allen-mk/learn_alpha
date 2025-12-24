# 🚀 第 1 小节：PyTorch Tensor 基础（我们从这里开始）

## 1. 什么是 Tensor？

张量（tensor）就是 **带维度的数字集合**，你可以理解为：

* 标量：0 维 Tensor（比如 3）
* 向量：1 维 Tensor（比如 [1,2,3]）
* 矩阵：2 维 Tensor（比如 3x3）
* 高维 Tensor：比如 32x3x224x224（图像批次）

> 学神经网络，你最先得学会“张量的 shape”。

---

## 2. 创建 Tensor（记住这几个就够了）

```python
import torch

# 基础创建
a = torch.tensor([1, 2, 3])
b = torch.zeros((2, 3))
c = torch.ones((2, 3))
d = torch.rand((3, 3))  # 0-1 随机数

# GPU 创建
a_gpu = torch.tensor([1,2,3], device="cuda")
```

---

## 3. Tensor 的“形状”（shape）和操作

### 常用操作：

```python
x = torch.rand((2, 3))   # shape = [2, 3]

x.shape
x.reshape(3, 2)
x.view(-1)               # 拉成一维
x.t()                    # 转置（矩阵）
```

---

## 4. 自动求导（神经网络的灵魂）

PyTorch 的自动求导可以帮你计算梯度。只需要：

```python
x = torch.tensor(2.0, requires_grad=True)
y = x * x  # y = x^2
y.backward()
print(x.grad)  # = 4
```

这是神经网络训练最基础的机制。

---

# 📌 小练习（请你做一下）

非常简单的一个练习，让我确认你理解并能上手。

### ✏️ 练习内容：

请你写一段 Python 代码，要求：

1. 创建一个张量 `x = [[1,2],[3,4]]`（2x2）
2. 把它 reshape 成 (4,)
3. 对这个 reshape 后的张量每个元素做平方得到 y
4. 计算 y 的总和，并调用 backward
5. 打印 x 的梯度

你只需要回复代码，我会帮你检查是否正确。

---

# 回复格式示例（你只要这么写就好）：

```python
# your code
```

你来试试吧！
