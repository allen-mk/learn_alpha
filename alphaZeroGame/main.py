"""
极简 AlphaZero 式井字棋训练脚本：神经网络 + MCTS + 自我对弈 + 训练循环。
运行示例:
    uv run python main.py --epochs 5 --games-per-epoch 30 --simulations 40
"""

import argparse
import json
import os
import logging
import random
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from game.mcts import MCTS
from game.minnet import Net
from game.tictactoe import TicTacToe

State = np.ndarray
Policy = np.ndarray
Outcome = float
LOGGER = logging.getLogger("alphazerogame")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def setup_logging(level: str = "info", log_file: str | None = None) -> None:
    """配置日志输出，支持控制台与可选的文件输出。"""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )


def format_board(board: np.ndarray) -> str:
    symbols = {1: "X", -1: "O", 0: "."}
    rows = [" ".join(symbols[int(cell)] for cell in row) for row in board]
    return "\n".join(rows)


def log_move(
    logger: logging.Logger,
    game_idx: int,
    move_count: int,
    player: int,
    move: Tuple[int, int],
    pi: np.ndarray,
    board: np.ndarray,
    temperature: float,
) -> None:
    """记录单步动作及 MCTS 概率，便于理解决策过程。"""
    board_size = board.shape[0]
    player_symbol = "X" if player == 1 else "O"
    selected_idx = move[0] * board_size + move[1]
    selected_prob = float(pi[selected_idx])
    top_actions = np.argsort(pi)[-3:][::-1]
    top_desc = ", ".join(
        f"({idx // board_size},{idx % board_size})={float(pi[idx]):.2f}" for idx in top_actions
    )
    logger.info(
        "[第 %d 局] 步 %d | 玩家: %s | 动作: (%d,%d) | 动作概率: %.2f | 温度: %.2f | 前3概率: %s",
        game_idx,
        move_count,
        player_symbol,
        move[0],
        move[1],
        selected_prob,
        temperature,
        top_desc,
    )
    logger.debug("[第 %d 局] 棋盘状态:\n%s", game_idx, format_board(board))


def prompt_mode() -> str:
    """交互式选择运行模式。"""
    print("请选择模式：")
    print("1) 自对弈训练（从零开始）")
    print("2) 自对弈训练（加载已有模型继续）")
    print("3) 人机对战（使用训练好的模型）")
    print("4) 双 AI 对决演示（使用模型互博一局）")
    choice = input("输入 1/2/3/4，然后回车：").strip()
    if choice == "1":
        return "train_new"
    if choice == "2":
        return "train_load"
    if choice == "3":
        return "play"
    if choice == "4":
        return "ai_vs_ai"
    raise ValueError("输入有误，请输入 1/2/3/4。")


def prompt_human_side() -> int:
    """交互式选择执子方，返回 1 表示 X，-1 表示 O。"""
    print("请选择你的棋子：")
    print("1) X（先手）")
    print("2) O（后手）")
    choice = input("输入 1 或 2，然后回车：").strip()
    if choice == "1":
        return 1
    if choice == "2":
        return -1
    raise ValueError("输入有误，请输入 1 或 2。")


def load_model(net: Net, path: str, logger: logging.Logger) -> bool:
    """尝试加载模型参数，成功返回 True。"""
    if not os.path.exists(path):
        logger.error("未找到模型文件: %s", path)
        return False
    state = torch.load(path, map_location="cpu")
    net.load_state_dict(state)
    logger.info("已加载模型参数: %s", path)
    return True


def load_game_config(path: str) -> dict[str, int]:
    """从 JSON 配置文件加载棋盘大小与连线规则。文件不存在则返回空字典。"""
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("配置文件内容应为 JSON 对象")
        return data
    except Exception as exc:
        LOGGER.error("读取配置文件失败: %s | error=%s", path, exc)
        raise


def self_play(
    net: Net,
    games: int,
    simulations: int,
    temperature: float,
    board_size: int,
    win_length: int,
    trace_moves: bool = False,
    logger: logging.Logger | None = None,
) -> Tuple[List[Tuple[State, Policy, Outcome]], dict[str, int]]:
    """通过自我对弈收集 (状态, MCTS 策略, 最终胜负值) 数据，并记录对局日志。"""
    logger = logger or LOGGER
    net.eval()
    dataset: List[Tuple[State, Policy, Outcome]] = []
    results = {"x": 0, "o": 0, "draw": 0}

    logger.info(
        "开始自对弈: 局数=%d, 每步模拟=%d, 温度=%.3f, 棋盘=%dx%d, 连线=%d, 记录动作=%s",
        games,
        simulations,
        temperature,
        board_size,
        board_size,
        win_length,
        trace_moves,
    )

    for game_idx in range(1, games + 1):
        env = TicTacToe(board_size=board_size, win_length=win_length)
        mcts = MCTS(net, simulations=simulations)
        history: List[Tuple[State, Policy, int]] = []  # 记录当前玩家
        move_count = 0
        logger.info("[第 %d 局] 开局，先手玩家：X", game_idx)

        while True:
            move_count += 1
            pi = mcts.get_action_probs(env, temperature=temperature)
            state = (env.board * env.player).astype(np.float32)  # 以当前玩家视角编码
            history.append((state, pi, env.player))

            current_player = env.player
            action_index = int(np.random.choice(env.board_size * env.board_size, p=pi))
            move = (action_index // env.board_size, action_index % env.board_size)
            _, done = env.step(move)

            if trace_moves:
                log_move(logger, game_idx, move_count, current_player, move, pi, env.board, temperature)

            if done:
                winner = env.check_winner()
                if winner == 1:
                    results["x"] += 1
                elif winner == -1:
                    results["o"] += 1
                else:
                    results["draw"] += 1

                for s, p, player in history:
                    outcome = 0.0 if winner == 0 else (1.0 if winner == player else -1.0)
                    dataset.append((s, p, outcome))

                outcome_label = "平局" if winner == 0 else ("X 胜" if winner == 1 else "O 胜")
                logger.info(
                    "[第 %d 局] 结束，结果: %s，步数: %d，样本累计: %d",
                    game_idx,
                    outcome_label,
                    move_count,
                    len(dataset),
                )
                if trace_moves:
                    logger.debug("[第 %d 局] 最终棋盘:\n%s", game_idx, format_board(env.board))
                break

    logger.info(
        "自对弈完成: X 胜 %d, O 胜 %d, 平局 %d，样本总计 %d",
        results["x"],
        results["o"],
        results["draw"],
        len(dataset),
    )
    return dataset, results


def human_vs_ai(
    net: Net,
    simulations: int,
    temperature: float,
    board_size: int,
    win_length: int,
    human_side: int | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """人与模型对战，使用训练好的网络与 MCTS。"""
    logger = logger or LOGGER
    net.eval()
    human_player = human_side if human_side is not None else prompt_human_side()
    human_symbol = "X" if human_player == 1 else "O"
    ai_symbol = "O" if human_player == 1 else "X"
    env = TicTacToe(board_size=board_size, win_length=win_length)
    mcts = MCTS(net, simulations=simulations)
    logger.info("开始人机对战，你执子 %s，AI 执子 %s", human_symbol, ai_symbol)
    print(f"对战开始，你执子 {human_symbol}，AI 执子 {ai_symbol}")
    print(format_board(env.board))

    move_count = 0
    while True:
        move_count += 1
        if env.player == human_player:
            # 玩家回合
            while True:
                user_input = input(f"请输入你的落子 (行 列，范围 1-{env.board_size} 1-{env.board_size})：").strip()
                try:
                    row_str, col_str = user_input.split()
                    row, col = int(row_str) - 1, int(col_str) - 1
                except Exception:
                    print("格式有误，请输入两个数字，例如 1 3")
                    continue
                if (row, col) not in env.legal_moves():
                    print("该位置不可落子，请重新选择。")
                    continue
                _, done = env.step((row, col))
                logger.info("玩家落子: (%d,%d)", row + 1, col + 1)
                break
        else:
            # AI 回合
            pi = mcts.get_action_probs(env, temperature=temperature)
            action_index = int(np.random.choice(env.board_size * env.board_size, p=pi))
            move = (action_index // env.board_size, action_index % env.board_size)
            _, done = env.step(move)
            logger.info(
                "AI 落子: (%d,%d) | 概率: %.3f | 温度: %.2f",
                move[0] + 1,
                move[1] + 1,
                float(pi[action_index]),
                temperature,
            )
            top_actions = np.argsort(pi)[-3:][::-1]
            logger.debug(
                "AI 概率Top3: %s",
                ", ".join(
                    f"({idx // env.board_size + 1},{idx % env.board_size + 1})={float(pi[idx]):.2f}"
                    for idx in top_actions
                ),
            )

        print(format_board(env.board))

        if done:
            winner = env.check_winner()
            if winner == human_player:
                print("你赢了！")
                logger.info("对战结束：玩家胜利")
            elif winner == 0:
                print("平局。")
                logger.info("对战结束：平局")
            else:
                print("AI 获胜。")
                logger.info("对战结束：AI 胜利")
            break


def ai_vs_ai(
    net: Net,
    simulations: int,
    temperature: float,
    board_size: int,
    win_length: int,
    logger: logging.Logger | None = None,
) -> None:
    """让两个 AI（同一网络）对决一局，打印每步棋盘。"""
    logger = logger or LOGGER
    net.eval()
    env = TicTacToe(board_size=board_size, win_length=win_length)
    mcts = MCTS(net, simulations=simulations)
    logger.info("开始双 AI 对决，双方使用同一模型")
    print("双 AI 对决开始：X vs O")
    print(format_board(env.board))

    move_count = 0
    while True:
        move_count += 1
        pi = mcts.get_action_probs(env, temperature=temperature)
        action_index = int(np.random.choice(env.board_size * env.board_size, p=pi))
        move = (action_index // env.board_size, action_index % env.board_size)
        player_symbol = "X" if env.player == 1 else "O"
        _, done = env.step(move)

        logger.info(
            "AI(%s) 落子: (%d,%d) | 概率: %.3f | 温度: %.2f | 步数: %d",
            player_symbol,
            move[0] + 1,
            move[1] + 1,
            float(pi[action_index]),
            temperature,
            move_count,
        )
        top_actions = np.argsort(pi)[-3:][::-1]
        logger.debug(
            "AI(%s) 概率Top3: %s",
            player_symbol,
            ", ".join(
                f"({idx // env.board_size + 1},{idx % env.board_size + 1})={float(pi[idx]):.2f}"
                for idx in top_actions
            ),
        )

        print(f"AI({player_symbol}) 落子: {move[0] + 1} {move[1] + 1}")
        print(format_board(env.board))

        if done:
            winner = env.check_winner()
            if winner == 0:
                print("对局结束：平局。")
                logger.info("双 AI 对局结束：平局")
            elif winner == 1:
                print("对局结束：X 胜。")
                logger.info("双 AI 对局结束：X 胜")
            else:
                print("对局结束：O 胜。")
                logger.info("双 AI 对局结束：O 胜")
            break


def train_epoch(
    net: Net,
    optimizer: torch.optim.Optimizer,
    batch: Sequence[Tuple[State, Policy, Outcome]],
    batch_size: int,
    logger: logging.Logger | None = None,
) -> float:
    """用收集到的数据训练一轮，返回平均 loss。"""
    logger = logger or LOGGER
    net.train()
    total_loss = 0.0
    steps = 0
    total_steps = max((len(batch) + batch_size - 1) // max(batch_size, 1), 1)

    for start in range(0, len(batch), batch_size):
        chunk = batch[start : start + batch_size]
        states_np = np.asarray([b[0].reshape(-1) for b in chunk], dtype=np.float32)
        target_pi_np = np.asarray([b[1] for b in chunk], dtype=np.float32)
        target_v_np = np.asarray([b[2] for b in chunk], dtype=np.float32)

        states = torch.from_numpy(states_np)
        target_pi = torch.from_numpy(target_pi_np)
        target_v = torch.from_numpy(target_v_np)

        logp, v = net(states)
        policy_loss = -(target_pi * logp).sum(dim=1).mean()
        value_loss = F.mse_loss(v.squeeze(-1), target_v)
        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        steps += 1
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "训练批 %d/%d | 策略损失: %.4f | 价值损失: %.4f | 总损失: %.4f",
                steps,
                total_steps,
                float(policy_loss.item()),
                float(value_loss.item()),
                float(loss.item()),
            )

    return total_loss / max(steps, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用 AlphaZero 思路自我对弈训练井字棋。")
    parser.add_argument("--config", type=str, default="config.json", help="JSON 配置文件路径，含 board_size 与 win_length")
    parser.add_argument("--board-size", type=int, default=None, help="棋盘大小 N，棋盘为 N x N（若配置文件有值可不填）")
    parser.add_argument("--win-length", type=int, default=None, help="获胜所需连续棋子数（若配置文件有值可不填）")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮次")
    parser.add_argument("--games-per-epoch", type=int, default=30, help="每轮自我对弈局数")
    parser.add_argument("--simulations", type=int, default=40, help="每一步 MCTS 模拟次数")
    parser.add_argument("--temperature", type=float, default=0.1, help="策略抽样温度，越低越贪心")
    parser.add_argument("--batch-size", type=int, default=32, help="训练批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam 学习率")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--log-level", type=str, default="info", choices=["debug", "info", "warning", "error", "critical"], help="日志等级")
    parser.add_argument("--log-file", type=str, default=None, help="可选，输出日志到文件路径")
    parser.add_argument("--trace-moves", action="store_true", help="记录自对弈每一步动作（日志量大，建议搭配 debug 等级）")
    parser.add_argument("--mode", type=str, choices=["train_new", "train_load", "play", "ai_vs_ai"], default=None, help="运行模式：新训练、加载模型训练、人机对战或双 AI 对决。若未指定，将在运行时交互选择。")
    parser.add_argument("--model-path", type=str, default=None, help="模型保存/加载路径。默认根据棋盘设置生成，例如 tictactoe_b3_w3.pt")
    parser.add_argument("--play-temperature", type=float, default=0.1, help="人机对战或演示时 AI 采样温度，越低越贪心")
    parser.add_argument("--human-side", type=str, choices=["X", "O"], default=None, help="人机对战时执子（X 先手 / O 后手）。未指定则交互选择。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level, args.log_file)
    LOGGER.info("启动，参数: %s", args)
    try:
        mode = args.mode or prompt_mode()
    except ValueError as exc:
        LOGGER.error(str(exc))
        return

    set_seed(args.seed)
    LOGGER.info("随机种子已设为 %d", args.seed)

    config_data = load_game_config(args.config) if args.config else {}
    if args.config and not config_data:
        LOGGER.info("配置文件未找到或为空，将使用命令行/默认值: %s", args.config)

    default_board_size = 3
    default_win_length = 3
    board_size = config_data.get("board_size", default_board_size)
    win_length = config_data.get("win_length", default_win_length)

    # CLI 显式传入则覆盖配置
    if args.board_size is not None:
        board_size = args.board_size
    if args.win_length is not None:
        win_length = args.win_length

    if board_size < 3:
        LOGGER.error("board_size 必须 >= 3，当前为 %d", board_size)
        return
    if win_length < 3 or win_length > board_size:
        LOGGER.error("win_length 必须满足 3 <= win_length <= board_size，当前 win_length=%d board_size=%d", win_length, board_size)
        return

    model_path = args.model_path or f"tictactoe_b{board_size}_w{win_length}.pt"
    LOGGER.info("模型文件: %s", model_path)

    net = Net(board_size=board_size)
    if mode in ("train_load", "play", "ai_vs_ai"):
        if not load_model(net, model_path, LOGGER):
            return

    if mode in ("train_new", "train_load"):
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        for epoch in range(1, args.epochs + 1):
            LOGGER.info(
                "======= 第 %d 轮 ======= | 自对弈局数: %d | 模拟次数: %d | 温度: %.2f",
                epoch,
                args.games_per_epoch,
                args.simulations,
                args.temperature,
            )
            data, results = self_play(
                net,
                games=args.games_per_epoch,
                simulations=args.simulations,
                temperature=args.temperature,
                board_size=board_size,
                win_length=win_length,
                trace_moves=args.trace_moves,
                logger=LOGGER,
            )
            LOGGER.info(
                "[第 %d 轮] 自对弈结果统计 -> X 胜: %d | O 胜: %d | 平局: %d",
                epoch,
                results["x"],
                results["o"],
                results["draw"],
            )
            avg_loss = train_epoch(net, optimizer, data, batch_size=args.batch_size, logger=LOGGER)
            LOGGER.info("[第 %d 轮] 自对弈样本数: %d | 平均损失: %.4f", epoch, len(data), avg_loss)

        # 训练完成后可保存模型
        torch.save(net.state_dict(), model_path)
        LOGGER.info("模型已保存到 %s", model_path)
    elif mode == "play":
        human_side = None
        if args.human_side:
            human_side = 1 if args.human_side.upper() == "X" else -1
        human_vs_ai(
            net,
            simulations=args.simulations,
            temperature=args.play_temperature,
            board_size=board_size,
            win_length=win_length,
            human_side=human_side,
            logger=LOGGER,
        )
    elif mode == "ai_vs_ai":
        ai_vs_ai(
            net,
            simulations=args.simulations,
            temperature=args.play_temperature,
            board_size=board_size,
            win_length=win_length,
            logger=LOGGER,
        )


if __name__ == "__main__":
    main()
