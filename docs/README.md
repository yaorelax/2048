# 2048 项目 Code Wiki

> 本 Wiki 面向开发者，系统梳理该 2048 项目的整体架构、模块职责、关键类与函数、依赖关系及运行方式。

## 目录

| 文档 | 说明 |
| --- | --- |
| [01-architecture.md](./01-architecture.md) | 项目整体架构、分层设计与核心数据流 |
| [02-core-modules.md](./02-core-modules.md) | 游戏核心模块：`ENV`、`Play2048` 及私有工具方法 |
| [03-ai-algorithms.md](./03-ai-algorithms.md) | AI 算法模块：启发式、Expectimax、DQN 与评测入口 |
| [04-dependencies.md](./04-dependencies.md) | 外部依赖与内部模块依赖关系 |
| [05-run-and-test.md](./05-run-and-test.md) | 环境搭建、运行方式与测试现状 |

## 项目一句话概述

一个基于 Python 的 2048 游戏实现，使用 Pygame 提供图形化交互界面，并在其上实现了三种自动游玩（AI）策略：**加权启发式**、**Expectimax 搜索**、**DQN 深度强化学习**。

## 关键信息速览

| 项 | 值 |
| --- | --- |
| 语言 / 运行时 | Python 3.8（见 [.idea/misc.xml](../.idea/misc.xml)） |
| 入口文件 | [play.py](../play.py) |
| 代码规模 | 单文件约 927 行，包含全部逻辑 |
| 核心类 | `ENV`、`Play2048`，以及算法内部类 `Brain`、`Net`、`DQN` |
| 主要外部库 | `numpy`、`pygame`、`torch`、`matplotlib` |
| 棋盘尺寸 | 默认 `4 x 4`（`Play2048(width=4)`） |
| 单局训练局数 | `MAX_EPISODE = 10`（DQN 模式内部改为 10000） |

## 阅读建议

1. 先读 [01-architecture.md](./01-architecture.md) 建立整体认知；
2. 再读 [02-core-modules.md](./02-core-modules.md) 理解游戏规则实现；
3. 若关注 AI 策略，继续读 [03-ai-algorithms.md](./03-ai-algorithms.md)；
4. 最后按 [05-run-and-test.md](./05-run-and-test.md) 搭建环境并运行。
