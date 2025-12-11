# AlphaZero 井字棋（可配置棋盘/连线规则）

一个极简的 AlphaZero 式井字棋实现，包含自对弈数据生成（MCTS）、策略/价值网络和训练循环。支持自定义棋盘大小与连线胜利条件，可针对不同规则训练/保存对应模型。

## 安装
- 推荐：`uv sync`
- 备选：`pip install -e .`

## 配置
- JSON 配置文件（默认 `config.json`，示例见 `config.example.json`）：
  ```json
  {
    "board_size": 3,
    "win_length": 3
  }
  ```
  - `board_size`: 棋盘边长 N（N x N，需 >=3）
  - `win_length`: 连续多少子获胜（3 <= win_length <= board_size）
- 命令行可覆盖配置文件中的值：`--board-size`、`--win-length`

## 运行模式
- 交互启动：`uv run python main.py`，提示选择
  - 1) 新训练（train_new）
  - 2) 继续训练（train_load）
  - 3) 人机对战（play）
  - 4) 双 AI 对决演示（ai_vs_ai）

## 常用命令示例
- 新训练（5x5、连4，免交互）  
  `uv run python main.py --mode train_new --board-size 5 --win-length 4 --epochs 5 --games-per-epoch 40 --simulations 60`

- 继续训练已有模型  
  `uv run python main.py --mode train_load --model-path tictactoe_b5_w4.pt`

- 人机对战（执 X，温度更贪心）  
  `uv run python main.py --mode play --model-path tictactoe_b5_w4.pt --human-side X --play-temperature 0.1`

- 双 AI 演示  
  `uv run python main.py --mode ai_vs_ai --model-path tictactoe_b5_w4.pt --play-temperature 0.1`

- 仅测试导入  
  `uv run python -c "from game.tictactoe import TicTacToe; print(TicTacToe(board_size=3, win_length=3).legal_moves())"`

## 重要参数
- `--config`: 配置文件路径（默认 `config.json`）
- `--board-size`, `--win-length`: 若提供则覆盖配置文件
- `--games-per-epoch`: 每轮自对弈局数
- `--simulations`: MCTS 每步模拟次数
- `--temperature`: 自对弈采样温度；`--play-temperature`: 对战/演示温度
- `--batch-size`, `--lr`, `--epochs`: 训练超参
- `--model-path`: 模型文件路径；未指定时自动生成 `tictactoe_b{board_size}_w{win_length}.pt`
- `--log-level`, `--log-file`, `--trace-moves`: 日志与追踪
- `--human-side {X,O}`: 人机对战执子

## 日志与可观测性
- 默认 `info`，可用 `--log-level debug` 查看批次损失、MCTS Top3 概率等。
- `--trace-moves` 记录自对弈每一步落子和概率分布（日志量大，建议与 debug 搭配）。
- 日志可通过 `--log-file` 同步到文件。

## 目录结构
- `main.py`：训练/对战入口与参数解析
- `game/tictactoe.py`：可配置棋盘与连线规则的环境
- `game/mcts.py`：基于网络先验的 MCTS
- `game/minnet.py`：极小策略/价值网络（输入输出随棋盘大小自适应）
- `config.example.json`：配置示例

## 说明
- 默认使用 CPU，需 GPU 可自行扩展模型/训练逻辑。***
