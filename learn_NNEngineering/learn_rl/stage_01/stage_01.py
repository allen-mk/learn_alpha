# 简单世界训练
# 一维世界：位置 0 ~ n
# 起点在 0，终点在 n
# 动作：0 = 左，1 = 右
# 到终点 reward = +1

import random

T = 20
N = 100
MAX = 10000

lr = 0.1
gamma = 0.9

# min Q-Learning
def Q_Learning_walk(t,n):
    Q = [[0,0] for _ in range(t+1)]
    # lr = 0.1
    # gamma = 0.9
    for i in range(n):
        s = 0
        while s != t:
            a = random.choice([0, 1])
            s_next = max(0, min(t, s + (1 if a == 1 else -1)))
            reward = 1 if s_next == t else 0
            
            # 计算 TD 目标
            td_target = reward + gamma * max(Q[s_next])
            # 计算 TD 误差
            td_error = td_target - Q[s][a]
            
            # 打印计算过程
            # print(f"Episode {i}, State {s}, Action {'右' if a else '左'}: Next State {s_next}, Reward {reward}, TD Target {td_target:.8f}, TD Error {td_error:.8f}, Old Q {Q[s][a]:.8f}")

            Q[s][a] += lr * td_error
            s = s_next

    print("Q-Table:")
    for i, q_values in enumerate(Q):
        print(f"位置 {i}: 左={q_values[0]:.8f}, 右={q_values[1]:.8f}")
    return Q


# my Q-Learning
def M_Q_Learning_walk(t,n):
    Q = [[0,0] for _ in range(t+1)]
    # lr = 0.1
    # gamma = 0.9
    for i in range(n):
        s = 0
        while s != t:
            for a in range(2):
                s_next = max(0, min(t, s + (1 if a == 1 else -1)))
                reward = 1 if s_next == t else 0
                
                # 计算 TD 目标
                td_target = reward + gamma * max(Q[s_next])
                # 计算 TD 误差
                td_error = td_target - Q[s][a]
                
                # 打印计算过程
                # print(f"Episode {i}, State {s}, Action {'右' if a else '左'}: Next State {s_next}, Reward {reward}, TD Target {td_target:.8f}, TD Error {td_error:.8f}, Old Q {Q[s][a]:.8f}")
                Q[s][a] += lr * td_error
            s += 1


    print("M-Q-Table:")
    for i, q_values in enumerate(Q):
        print(f"位置 {i}: 左={q_values[0]:.8f}, 右={q_values[1]:.8f}")
    return Q


# 方向运算 Q-Learning
# def D_Q_Learning_walk(t,n):
#     Q = [[0,0] for _ in range(t+1)]
#     # lr = 0.1
#     # gamma = 0.9
#     for i in range(n):
#         s = 0
#         while s != t:
#             a = random.choice([0, 1])
#             s_next = max(0, min(t, s + (1 if a == 1 else -1)))
#             reward = 999999 if s_next == t else 0
#             reward += (s_next - s)/t
#             reward /= 1000000 
            
#             # 计算 TD 目标
#             td_target = reward + gamma * max(Q[s_next])
#             # 计算 TD 误差
#             td_error = td_target - Q[s][a]
            
#             # 打印计算过程
#             # print(f"Episode {i}, State {s}, Action {'右' if a else '左'}: Next State {s_next}, Reward {reward}, TD Target {td_target:.8f}, TD Error {td_error:.8f}, Old Q {Q[s][a]:.8f}")
#             Q[s][a] += lr * td_error
#             s = s_next

#     print("D-Q-Table:")
#     for i, q_values in enumerate(Q):
#         print(f"位置 {i}: 左={q_values[0]:.8f}, 右={q_values[1]:.8f}")
#     return Q


# 方向运算 Q-Learning
def D_Q_Learning_walk(t,n):
    Q = [[0,0] for _ in range(t+1)]
    # lr = 0.1
    # gamma = 0.9
    for i in range(n):
        s = 0
        while s != t:
            for i in range(8):
                a = i % 2
                s_next = max(0, min(t, s + (1 if a == 1 else -1)))
                reward = 99 if s_next == t else 0
                reward += (s_next - s)/t
                reward /= 100 
            
                # 计算 TD 目标
                td_target = reward + gamma * max(Q[s_next])
                # 计算 TD 误差
                td_error = td_target - Q[s][a]
                
                # 打印计算过程
                # print(f"Episode {i}, State {s}, Action {'右' if a else '左'}: Next State {s_next}, Reward {reward}, TD Target {td_target:.8f}, TD Error {td_error:.8f}, Old Q {Q[s][a]:.8f}")
                Q[s][a] += lr * td_error
            s += 1

    print("D-Q-Table:")
    for i, q_values in enumerate(Q):
        print(f"位置 {i}: 左={q_values[0]:.8f}, 右={q_values[1]:.8f}")
    return Q


# 检查数据是否正确
def check_Q_table(Q):
    ok = True
    p = -1
    for i in range(len(Q)):
        if Q[i][0] > Q[i][1]:
            ok = False
            p = i
            break

    print("Q-Table 检查结果: " + ("正确" if ok else "错误"))
    if not ok:
        print("错误位置: " + str(p))
    return ok

Q_TABLE = Q_Learning_walk(T, N)
check_Q_table(Q_TABLE)
# M_Q_TABLE = M_Q_Learning_walk(T, N)
# check_Q_table(M_Q_TABLE)
D_Q_TABLE = D_Q_Learning_walk(T, N)
check_Q_table(D_Q_TABLE)



# 随机走动的函数
def random_walk(t):
    p = 0
    c = 0
    while p != t:
        a = random.choice([0, 1])
        p = max(0, min(t, p + (1 if a == 1 else -1)))
        c += 1
        if c > MAX:
            break

    print(f"随机走动的步数: {c}")

    return c

# ai walk
def ai_walk(t,Q):
    p = 0
    c = 0
    while p != t:
        a = random.choice([0, 1]) if Q[p][0] == Q[p][1] else (0 if Q[p][0] > Q[p][1] else 1)
        p = max(0, min(t, p + (1 if a == 1 else -1)))
        c += 1
        if c > MAX:
            break

    print(f"ai走动的步数: {c}")
    return c

if __name__ == "__main__":
    print("===============")
    # Q_Learning_walk(T, N)
    # random_walk(T)
    # ai_walk(T,Q_TABLE)
    # ai_walk(T,M_Q_TABLE)
    # ai_walk(T,D_Q_TABLE)



    
