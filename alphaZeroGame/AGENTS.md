# 仓库指南

## 项目结构与模块组织
- 核心代码在 `game/`：`tictactoe.py` 提供 3x3 井字棋环境，`mcts.py` 实现基于网络输出的蒙特卡洛树搜索，`minnet.py` 定义极小的策略/价值网络。
- 根目录含打包文件（`pyproject.toml`, `uv.lock`）和预留的 `main.py`，用于后续入口脚本。可将实验或临时脚本放在自建的 `scripts/` 中，通用逻辑放 `game/`。
- 生成的模型、数据、检查点等不要提交；在提交前先更新 `.gitignore`。

## 构建、测试与开发命令
- `uv sync`（推荐）或 `pip install -e .` 安装 Python 3.11+ 依赖。
- `uv run python -c "from game.tictactoe import TicTacToe; print(TicTacToe().legal_moves())"` 快速确认导入和依赖正常。
- `uv run python main.py` 运行自定义入口（实现后再用）。
- `uv run pytest` 运行测试套件。

## 代码风格与命名规范
- 遵循 PEP 8，4 空格缩进；函数/方法用 `snake_case`，类用 `PascalCase`。
- 公共函数写类型标注与简洁 docstring，说明输入/输出。
- 张量/数组命名体现形状或角色（如 `board_flat`, `policy_logits`）；基准或调试时使用 `torch.manual_seed`、`random.seed`、`np.random.seed` 固定随机性。

## 测试指南
- 使用 `pytest`，测试放 `tests/`，文件/函数命名为 `test_*.py`，结构与模块对应（如 `tests/test_mcts.py`）。
- 编写快速、可重复的检查：环境状态转移、MCTS 只选择合法动作、`Net` 返回 `(batch, 9)` 策略 logits 和 `(batch, 1)` 价值。
- 修复缺陷时添加回归测试；多用 fixture，避免冗长的手动搭建。

## 提交与 PR 指南
- 提交信息用清晰的现在时；鼓励 Conventional Commits（如 `feat: add training loop`, `fix: guard illegal moves`）。
- 若有 issue，请在 PR 中引用；描述范围、风险、覆盖面。
- PR 备注中列出验证步骤（运行过的命令和结果），并说明对性能或依赖的影响。

## 安全与配置提示
- 精简依赖，避免提交大体积模型或数据集。
- 默认为 CPU 路径，除非显式支持 GPU；实验特性用命令行参数或环境变量开关。
