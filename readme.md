# COH2 Match Analysis / 英雄连2 比赛数据分析

这是一个基于 *Company of Heroes 2* (COH2) 比赛数据的分析项目。本项目包含从数据爬取、胜负预测模型训练，到玩家实力（Tier）评估与可视化的完整流程。

## 📋 功能模块

项目主要包含三个核心任务：

1.  **数据获取 (Data Collection)**: 批量下载并解析玩家的比赛历史数据。
2.  **胜负预测 (Win/Loss Prediction)**: 基于比赛数据特征（如杀敌、资源获取效率等）训练机器学习模型预测比赛结果。
3.  **实力评估 (Player Skill Assessment)**: 基于神经网络（MLP）模型，通过分析玩家表现数据，评估其对应的实力分段（Tier 0 - Tier 19），并生成可视化分析报告。

### 任务一：数据获取与基础分析

**第一步：获取数据** 使用 `fetch_player_all_matches.py` 抓取特定玩家的比赛记录。数据将保存为 `.jsonl` 格式。

```
# 请在脚本中配置好 steam_id 或 profile_id 后运行
python fetch_player_all_matches.py
```

*注：抓取的数据会默认保存在 `data/players/` 目录下。*

**第二步：训练胜负预测模型** 使用 `train_task1.py` 读取数据并训练 Logistic Regression 和 Random Forest 模型。

```
python train_task1.py
```

- 该脚本会自动读取 `data/players/*.jsonl` 下的所有数据。
- 模型会保存到 `models/task1/`。

**第三步：可视化特征权重** 运行 `viz_task1.py` 查看哪些游戏数据（如每分钟伤害、燃油获取率）对胜负影响最大。

```
python viz_task1.py
```

### 任务二：玩家实力评估 (Tier Prediction)

**第一步：训练实力评估模型** 使用 `task2_train_tier20.py` 训练一个神经网络模型。该模型会学习高分段玩家与低分段玩家的数据差异。

```
python task2_train_tier20.py
```

- 脚本会根据 `oldrating` 自动划分 20 个 Tier 等级作为标签进行训练。
- 模型权重保存至 `models/task2_tier20/`。

**第二步：单人实力推理与报告生成** 使用 `task2_infer_and_viz_player.py` 或 `task2_viz_player_report.py` 对特定玩家进行详细分析。

```
# 生成玩家的六边形能力图及详细数据报告
python task2_viz_player_report.py
```

- 该脚本会加载训练好的模型。
- 输出包含：玩家定级预测、各项能力评分（战斗、运营、控图等）与同分段平均水平的对比图。

## 📂 文件结构说明

- `fetch_player_all_matches.py`: 数据爬虫脚本。
- `train_task1.py`: 任务1训练脚本（胜负分类）。
- `viz_task1.py`: 任务1可视化脚本（特征重要性）。
- `task2_train_tier20.py`: 任务2训练脚本（Tier分类模型）。
- `task2_viz_player_report.py`: 任务2核心应用，生成玩家分析报告。
- `task2_features.py`: 特征工程核心代码，定义了如何从原始数据计算 `dmg_per_min` 等特征。
- `task2_tiers.py`: 处理段位（Tier）划分逻辑的工具脚本。

## 📢 数据来源与致谢 (Data Source)

本项目所使用的比赛数据来自于 **[COH2 Stats](https://coh2stats.com/)**。

> **COH2 Match Data / COH2 匹配数据** The data is provided by **COH2 Stats - Open Data**. Special thanks to the team at coh2stats.com for providing accessible match history and statistical data for the community.
>
> 如果您使用本项目的逻辑进行二次开发或数据分析，也请注明数据来源于 coh2stats.com。