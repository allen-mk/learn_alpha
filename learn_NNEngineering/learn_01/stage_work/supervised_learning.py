# 手搓一个简单的监督学习，以线性函数为例子

import random



# 学习率
lr = 0.01

# 训练轮次
epochs = 100

w_t = 5
b_t = 8



def test1():
    # 1. 初始化参数 乱猜
    w = random.uniform(-1, 1)
    b = random.uniform(-1, 1)

    print(f"w = {w:.4f}, b = {b:.4f}")
    # 2. 训练 出题
    data = [(x, w_t * x + b_t) for x in range(-10,11)]

    for epoch in range(epochs):
        total_loss = 0
        for x, y_true in data:
            # 预测
            y_pred = w * x + b
            # 损失
            loss = (y_pred - y_true) ** 2
            total_loss += loss


            # 相当于 backward
            # 反向传播
            dw = 2 * (y_pred - y_true) * x
            db = 2 * (y_pred - y_true)


            # 相当于 optimizer
            # 更新参数
            w -= lr * dw
            b -= lr * db

            print(f"epoch {epoch}, loss = {total_loss:.4f}, w = {w:.4f}, b = {b:.4f} , dw = {dw:.4f}, db = {db:.4f}")

            # if epoch % 100 == 0:
            #     print(f"epoch {epoch}, loss = {total_loss:.4f}, w = {w:.4f}, b = {b:.4f}")

    print("训练完成")
    print(f"w = {w}, b = {b}")


def test2():
    # 1. 初始化参数 乱猜
    w = random.uniform(-1, 1)
    b = random.uniform(-1, 1)
    print(f"w = {w:.4f}, b = {b:.4f}")

    for epoch in range(epochs):
        total_loss = 0
        for x in range(-10, 11):
            y_true = w_t * x + b_t
            y_pred = w * x + b

            loss = (y_pred - y_true) ** 2
            total_loss += loss

            dw = 2 * (y_pred - y_true) * x
            db = 2 * (y_pred - y_true)

            w -= lr * dw
            b -= lr * db

            print(f"epoch {epoch}, loss = {total_loss:.4f}, w = {w:.4f}, b = {b:.4f}, dw = {dw:.4f}, db = {db:.4f}")

    print("训练完成")
    print(f"w = {w}, b = {b}")






if __name__ == "__main__":
    test1()
    # test2()