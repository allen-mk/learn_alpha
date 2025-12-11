import math
from typing import Tuple

import numpy as np
import torch

from .tictactoe import TicTacToe


def clone_env(env: TicTacToe) -> TicTacToe:
    """浅克隆 TicTacToe 环境以便在 MCTS 中进行模拟。"""
    new_env = TicTacToe(board_size=env.board_size, win_length=env.win_length)
    new_env.board = env.board.copy()
    new_env.player = env.player
    return new_env


class MCTS:
    """简化版蒙特卡洛树搜索（MCTS）实现，用于 AlphaZero 风格自我对弈。"""

    def __init__(self, net: torch.nn.Module, simulations: int = 30, cpuct: float = 1.4):
        """
        参数:
            net: 策略/价值网络。
            simulations: 每个根节点的搜索次数。
            cpuct: UCB 探索系数。
        """
        self.net = net
        self.simulations = simulations
        self.cpuct = cpuct
        # 状态键 -> 动态数组，长度取决于棋盘尺寸
        self.Q = {}  # 平均价值
        self.N = {}  # 访问次数
        self.P = {}  # 先验策略

    def _state_key(self, env: TicTacToe) -> Tuple[int, Tuple[int, ...]]:
        # 将棋盘和当前执子方编码为可哈希键
        return env.player, tuple(env.board.reshape(-1))

    def _is_terminal(self, env: TicTacToe) -> Tuple[bool, float]:
        """返回 (是否结束, 当前玩家视角下的价值)。"""
        winner = env.check_winner()
        if winner != 0:
            # 当前要落子的玩家若不是赢家，则为输家
            return True, 1.0 if winner == env.player else -1.0
        if len(env.legal_moves()) == 0:
            return True, 0.0
        return False, 0.0

    def search(self, env: TicTacToe) -> float:
        done, terminal_value = self._is_terminal(env)
        if done:
            return terminal_value

        s = self._state_key(env)

        # 未扩展过的叶子节点：用网络估计策略/价值
        if s not in self.P:
            with torch.no_grad():
                state_tensor = torch.tensor(
                    env.board.reshape(-1) * env.player, dtype=torch.float32
                ).unsqueeze(0)
                logp, v = self.net(state_tensor)
                p = torch.exp(logp).cpu().numpy()[0]
            legal = env.legal_moves()
            num_cells = env.board_size * env.board_size
            mask = np.zeros(num_cells, dtype=np.float32)
            legal_actions = [i * env.board_size + j for i, j in legal]
            mask[legal_actions] = 1
            p = p * mask
            if p.sum() <= 0:
                p = mask / mask.sum()
            else:
                p = p / p.sum()

            self.P[s] = p
            self.N[s] = np.zeros(num_cells, dtype=np.float32)
            self.Q[s] = np.zeros(num_cells, dtype=np.float32)
            return float(v.item())

        # 选择阶段：按 UCB 选择动作
        legal = env.legal_moves()
        actions = [i * env.board_size + j for i, j in legal]
        total_n = np.sum(self.N[s]) + 1.0

        def ucb_score(a: int) -> float:
            prior = self.P[s][a]
            return self.Q[s][a] + self.cpuct * prior * math.sqrt(total_n) / (1 + self.N[s][a])

        best_a = max(actions, key=ucb_score)

        # 模拟
        child = clone_env(env)
        reward, done = child.step((best_a // env.board_size, best_a % env.board_size))
        if done:
            v = float(reward)
        else:
            v = -self.search(child)

        # 反向传播
        self.N[s][best_a] += 1
        self.Q[s][best_a] += (v - self.Q[s][best_a]) / self.N[s][best_a]
        return v

    def get_action_probs(self, env: TicTacToe, temperature: float = 1.0) -> np.ndarray:
        """对根节点重复模拟，返回归一化访问计数作为策略。"""
        for _ in range(self.simulations):
            self.search(clone_env(env))

        s = self._state_key(env)
        counts = self.N.get(s)
        if counts is None:
            # 若状态未初始化（极少发生），补一次
            self.search(clone_env(env))
            counts = self.N[s]

        legal = env.legal_moves()
        actions = [i * env.board_size + j for i, j in legal]
        selected_counts = np.array([counts[a] for a in actions], dtype=np.float32)

        num_cells = env.board_size * env.board_size

        if temperature <= 1e-3:
            # 近似贪心：将概率全部放在访问次数最多的动作上
            best_idx = actions[int(np.argmax(selected_counts))]
            probs = np.zeros(num_cells, dtype=np.float32)
            probs[best_idx] = 1.0
            return probs

        selected_counts = selected_counts ** (1.0 / temperature)
        norm = selected_counts.sum()
        probs = np.zeros(num_cells, dtype=np.float32)
        if norm <= 0:
            probs[actions] = 1.0 / len(actions)
        else:
            probs[actions] = selected_counts / norm
        return probs
