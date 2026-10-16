# 05 · 运行方式与测试

## 5.1 环境要求

- Python 3.8（项目默认 SDK，见 [.idea/misc.xml](../.idea/misc.xml)）
- 具备图形显示环境（Pygame 需要创建窗口）
- 依赖见 [requirements.txt](../requirements.txt)

## 5.2 安装依赖

```bash
cd /workspace
pip install -r requirements.txt
```

依赖清单：`matplotlib==3.3.2`、`torch==1.8.1+cpu`、`numpy==1.20.2`、`pygame==2.0.1`。

## 5.3 运行

入口为 [play.py](../play.py)，直接执行：

```bash
python play.py
```

### 运行模式切换

模式由 `main()` 内部选择（见 [play.py](../play.py#L919-L924)）。当前默认启用的是 **DQN 模式**：

```python
def main():
    play = Play2048(width=4, debug=False)
    env = ENV(play)
    # human_play(env)      # 人类手动游玩（被注释）
    # ai_play(env)         # AI 算法评测 + 绘图（被注释）
    dqn_algorithm(env)     # 当前实际运行：DQN 训练
```

**切换方式**：修改 `main()`，按需放开对应调用：

| 想要的效果 | 修改 |
| --- | --- |
| 人类手动游玩 | 注释 `dqn_algorithm(env)`，启用 `human_play(env)` |
| 多算法评测对比 | 注释 `dqn_algorithm(env)`，启用 `ai_play(env)` |
| DQN 训练（默认） | 保持现状 |

### `ai_play` 的算法配置

`ai_play` 内部的 `configs` 决定评测哪些算法（见 [play.py](../play.py#L869-L882)）：

```python
configs = []
configs.append((expectimax_algorithm, 4))          # 默认启用：Expectimax，深度 4
# configs.append((heuristic_algorithm, [1, 1, 1, 1]))
# configs.append((heuristic_algorithm, [4, 3, 3, 1]))
# configs.append((heuristic_algorithm_with_one_step_heuristic, ['one']))
```

按需取消注释即可加入更多算法，运行结束后会弹出四张柱状图（min / max / mean / cv）。

### 交互操作（人类模式）

| 按键 | 动作 |
| --- | --- |
| `←` / `→` / `↑` / `↓` | 向左 / 右 / 上 / 下移动 |
| `R` | 重新开始 |
| 关闭窗口 | 退出程序 |

## 5.4 调参与可配置项

| 参数 | 位置 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `MAX_EPISODE` | [play.py L13](../play.py#L13) | 10 | 评测模式每算法局数；DQN 模式内部改写为 10000 |
| `width` | `main()` | 4 | 棋盘边长（渲染与逻辑适配） |
| `debug` | `Play2048.__init__` | `False` | 打印棋盘与生成/移动日志 |
| Expectimax 深度 | `ai_play` 中 `max_depth` | 4 | 越大越强但越慢 |
| 启发式权重 | `heuristic_algorithm` 的 `weights` | `[1,1,1,1]` | 四特征权重 |
| DQN 超参数 | `dqn_algorithm` 内 | 见 [03-ai-algorithms.md](./03-ai-algorithms.md) | 学习率、折扣、回放池等 |

## 5.5 无图形环境运行（headless）

由于所有模式都依赖 Pygame 窗口主循环，在无显示设备的服务器上直接运行会失败。可采用：

```bash
SDL_VIDEODRIVER=dummy python play.py
```

> 注意：即便使用 dummy 驱动，算法仍是长时间运行的循环，且默认 DQN 需训练 10000 局，请评估耗时后再运行。

## 5.6 测试现状

| 文件 | 状态 | 说明 |
| --- | --- | --- |
| [test.py](../test.py) | **空文件** | 仓库当前**没有任何单元测试或自动化测试** |

也就是说，项目未提供测试套件；验证方式主要是"运行后观察控制台输出的每局得分/步数/最大方块"以及 `ai_play` 的统计图表。

## 5.7 运行观察点（控制台输出）

各算法每局结束会打印一行状态，便于人工评估：

- 启发式：`[heuristic<权重>]episode:xxx score:xxx step:xxx max_block:xxx`
- 一步启发式：`[heuristic__one_step]episode:...`
- Expectimax：`[expectimax<深度>]episode:...`
- DQN：`[dqn]episode:...`

`@coast_time` 装饰的算法在整体结束时额外打印一行 `func <name> coast time: ... s`。

## 5.8 快速排障

| 现象 | 可能原因 | 处理 |
| --- | --- | --- |
| 无法创建窗口 / 报显示错误 | 无图形环境 | 使用 `SDL_VIDEODRIVER=dummy` 或本地图形环境运行 |
| 中文字体显示异常 | 依赖系统字体 `simsunnsimsun` | 安装/替换为可用中文字体 |
| `numpy` / `torch` 安装失败 | Python 版本与固定版本不兼容 | 匹配到 Python 3.8 环境或放宽版本约束 |
| 程序长时间不结束 | 算法为无限循环，需跑满 `MAX_EPISODE`（DQN 默认 10000 局） | 调小 `MAX_EPISODE` 或改用人类模式 |
| 运行结果无统计图 | 未启用 `ai_play` | 在 `main()` 中启用 `ai_play(env)` |
