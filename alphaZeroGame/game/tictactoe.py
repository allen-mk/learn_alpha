# 这是一个极简井字棋游戏 环境

import numpy as np


class TicTacToe:
    def __init__(self, board_size: int = 3, win_length: int = 3):
        """
        参数:
            board_size: 棋盘边长 (N)，棋盘为 N x N。
            win_length: 连续多少子获胜。
        """
        if board_size < 3:
            raise ValueError("board_size 必须 >= 3")
        if win_length < 3 or win_length > board_size:
            raise ValueError("win_length 必须满足 3 <= win_length <= board_size")
        self.board_size = board_size
        self.win_length = win_length
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.player = 1  # 当前玩家，1或-1

    def legal_moves(self):
        # 返回所有合法的移动（即棋盘上所有空的格子）
        return [(i, j) for i in range(self.board_size) for j in range(self.board_size) if self.board[i, j] == 0]

    def step(self, move):
        # 执行一步移动
        i, j = move
        if not (0 <= i < self.board_size and 0 <= j < self.board_size):
            raise ValueError("落子越界")
        if self.board[i, j] != 0:
            raise ValueError("该位置已被占用")
        self.board[i, j] = self.player  # 当前玩家在指定位置落子
        winner = self.check_winner()  # 检查是否有赢家
        # 判断游戏是否结束：有赢家或棋盘已满
        done = winner != 0 or len(self.legal_moves()) == 0
        # 计算奖励：从当前玩家视角，赢家得1，输家得-1，平局得0
        reward = winner * self.player
        self.player *= -1  # 切换到下一个玩家
        return reward, done  # 返回奖励和游戏结束状态

    def check_winner(self):
        """检查是否有玩家获胜，支持任意 board_size 与 win_length。"""
        n = self.board_size
        k = self.win_length
        board = self.board
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for i in range(n):
            for j in range(n):
                player = board[i, j]
                if player == 0:
                    continue
                for di, dj in directions:
                    end_i = i + (k - 1) * di
                    end_j = j + (k - 1) * dj
                    if not (0 <= end_i < n and 0 <= end_j < n):
                        continue
                    win = True
                    for step in range(1, k):
                        if board[i + step * di, j + step * dj] != player:
                            win = False
                            break
                    if win:
                        return int(player)
        return 0  # 没有赢家
