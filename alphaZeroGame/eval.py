"""
对比两个模型的自博弈评估脚本：
- 交替先后手，统计胜/负/平
- 输出胜率与估算 Elo 差
示例：
    uv run python eval.py --model-a tictactoe_b5_w4_epoch0.pt --model-b tictactoe_b5_w4_epoch5.pt --games 400 --simulations 60 --temperature 0.05
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from typing import Literal

import numpy as np
import torch

from game.mcts import MCTS
from game.minnet import Net
from game.tictactoe import TicTacToe

LOGGER = logging.getLogger("eval")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_game_config(path: str) -> dict:
    """从 JSON 配置文件加载棋盘大小与连线规则。文件不存在则返回空字典。"""
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("配置文件内容应为 JSON 对象")
    return data


def estimate_elo(win_rate: float) -> float | None:
    """
    根据胜率估算 Elo 差值（A 相对 B）。
    win_rate = (win + 0.5 * draw) / total
    """
    if win_rate <= 0.0 or win_rate >= 1.0:
        return None
    return -400.0 * math.log10(1.0 / win_rate - 1.0)


def play_one_game(
    net_a: Net,
    net_b: Net,
    board_size: int,
    win_length: int,
    simulations: int,
    temperature: float,
    start_player: int,
) -> Literal["a", "b", "draw"]:
    """
    单局对战，交替先后手：
    - start_player = 1 表示 A 执先；-1 表示 B 执先。
    - 返回胜者归属或平局。
    """
    env = TicTacToe(board_size=board_size, win_length=win_length)
    mcts_a = MCTS(net_a, simulations=simulations)
    mcts_b = MCTS(net_b, simulations=simulations)

    a_side = start_player
    b_side = -start_player

    while True:
        # 选择当前执子的模型
        if env.player == a_side:
            mcts = mcts_a
        else:
            mcts = mcts_b

        pi = mcts.get_action_probs(env, temperature=temperature)
        action_index = int(np.random.choice(env.board_size * env.board_size, p=pi))
        move = (action_index // env.board_size, action_index % env.board_size)
        _, done = env.step(move)

        if done:
            winner = env.check_winner()
            if winner == a_side:
                return "a"
            if winner == b_side:
                return "b"
            return "draw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估两个 AlphaZero 井字棋模型的对战表现。")
    parser.add_argument("--model-a", type=str, required=True, help="模型 A 路径（作为对比基线或新模型）")
    parser.add_argument("--model-b", type=str, required=True, help="模型 B 路径（作为对手模型）")
    parser.add_argument("--config", type=str, default="config.json", help="JSON 配置文件路径，含 board_size 与 win_length")
    parser.add_argument("--board-size", type=int, default=None, help="棋盘大小 N，棋盘为 N x N（若配置文件有值可不填）")
    parser.add_argument("--win-length", type=int, default=None, help="获胜所需连续棋子数（若配置文件有值可不填）")
    parser.add_argument("--games", type=int, default=200, help="评估对局数（建议 200-500）")
    parser.add_argument("--simulations", type=int, default=60, help="每一步 MCTS 模拟次数")
    parser.add_argument("--temperature", type=float, default=0.1, help="对战采样温度，越低越接近贪心")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--log-level", type=str, default="info", choices=["debug", "info", "warning", "error", "critical"], help="日志等级")
    parser.add_argument("--log-file", type=str, default=None, help="可选，输出日志到文件路径")
    return parser.parse_args()


def setup_logging(level: str, log_file: str | None) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level, args.log_file)
    set_seed(args.seed)

    config_data = load_game_config(args.config) if args.config else {}
    if args.config and not config_data:
        LOGGER.info("配置文件未找到或为空，将使用命令行/默认值: %s", args.config)

    board_size = config_data.get("board_size", 3)
    win_length = config_data.get("win_length", 3)
    if args.board_size is not None:
        board_size = args.board_size
    if args.win_length is not None:
        win_length = args.win_length

    if board_size < 3:
        LOGGER.error("board_size 必须 >= 3，当前为 %d", board_size)
        return
    if win_length < 3 or win_length > board_size:
        LOGGER.error(
            "win_length 必须满足 3 <= win_length <= board_size，当前 win_length=%d board_size=%d",
            win_length,
            board_size,
        )
        return

    LOGGER.info(
        "评估开始 | 棋盘=%dx%d 连线=%d | games=%d | sims=%d | temp=%.3f | 模型A=%s | 模型B=%s",
        board_size,
        board_size,
        win_length,
        args.games,
        args.simulations,
        args.temperature,
        args.model_a,
        args.model_b,
    )

    net_a = Net(board_size=board_size)
    net_b = Net(board_size=board_size)
    net_a.load_state_dict(torch.load(args.model_a, map_location="cpu"))
    net_b.load_state_dict(torch.load(args.model_b, map_location="cpu"))
    net_a.eval()
    net_b.eval()

    results = {"a": 0, "b": 0, "draw": 0}

    for game_idx in range(1, args.games + 1):
        # 奇数局 A 先，偶数局 B 先
        start_player = 1 if game_idx % 2 == 1 else -1
        outcome = play_one_game(
            net_a,
            net_b,
            board_size=board_size,
            win_length=win_length,
            simulations=args.simulations,
            temperature=args.temperature,
            start_player=start_player,
        )
        results[outcome] += 1
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug("Game %d outcome=%s | 累计 A:%d B:%d Draw:%d", game_idx, outcome, results["a"], results["b"], results["draw"])

    total = args.games
    score_a = results["a"] + 0.5 * results["draw"]
    win_rate = score_a / total if total > 0 else 0.0
    elo = estimate_elo(win_rate)

    LOGGER.info("评估完成 | A 胜:%d | B 胜:%d | 平局:%d | 胜率(A)=%.4f", results["a"], results["b"], results["draw"], win_rate)
    if elo is None:
        LOGGER.info("Elo 估计：无法计算（胜率为 0 或 1）")
    else:
        LOGGER.info("Elo 估计：A 相对 B ≈ %.2f", elo)

    print("========== 评估结果 ==========")
    print(f"棋盘: {board_size}x{board_size}, 连线: {win_length}")
    print(f"总局数: {total}")
    print(f"A 胜: {results['a']} | B 胜: {results['b']} | 平局: {results['draw']}")
    print(f"A 胜率 (含平局0.5): {win_rate:.4f}")
    if elo is None:
        print("Elo 估计: 无法计算（胜率为 0 或 1）")
    else:
        print(f"Elo 估计: A 相对 B ≈ {elo:.2f}")


if __name__ == "__main__":
    main()

