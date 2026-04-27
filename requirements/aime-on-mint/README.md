# 在 MinT 上复现 AIME：需求文档

## 1. 目标

本需求文档用于定义本仓库在 AIME 方向上的首期建设目标。

首期目标不是搭建通用训练框架，而是交付一个满足以下条件的实验目录：

- 自包含
- 可训练
- 可评测
- 可自动实验
- 可作为后续 experiment 的模板

本期指定的目标实验为：

- `experiments/dapo-aime24/`

该实验用于承载一条明确的研究问题：

> 在 MinT 上，直接使用 DAPO-Math-17k 做 GRPO 训练，并在冻结的完整 AIME 2024 benchmark 上评测，形成可复现基线。

---

## 2. 范围

### 2.1 本期必须覆盖

本期必须覆盖以下内容：

- experiment-monorepo 目录契约
- `dapo-aime24` 实验目录
- 本地可读的训练数据与评测数据
- 统一训练/评测入口 `train.py`
- `--dry-run`
- `--eval-only`
- checkpoint 或 state 恢复能力
- 稳定的 `METRIC` 输出格式
- `autoresearch.sh` 自动实验入口
- 与 benchmark 口径配套的 `README.md`、`autoresearch.md`、`data/sources.yaml`

### 2.2 本期不要求覆盖

本期不要求覆盖以下内容：

- 多个算法同时复现
- 通用共享训练框架
- 完整等价复刻上游 DAPO verifier stack
- 独立的 `inspect_ai` / `tinker-cookbook` eval-only 实验
- 多 benchmark 并行体系
- 图形题专门处理链路

---

## 3. 仓库级要求

### 3.1 monorepo 结构要求

仓库应采用 experiment-monorepo 结构。

根目录应保留仓库级文件，至少包括：

- `README.md`
- `AGENTS.md`
- `.env.example`
- `experiments/README.md`
- `requirements/`

其中根目录 `.env.example` 的职责是为 experiment-local `.env` 提供模板，而不是作为运行时 fallback。

### 3.2 `requirements/` 目录要求

`requirements/` 目录只用于承载：

- 需求定义
- 约束说明
- benchmark 口径
- 验收标准
- 设计边界

`requirements/` 不应用于记录实现现状，不应用于替代 experiment 目录内的使用说明。

### 3.3 `experiments/` 目录要求

`experiments/` 下每个子目录都应是独立、自包含实验。

每个实验目录必须满足：

- 可单独 `uv sync`
- 可单独 `uv run train.py`
- 可单独 `uv run train.py --dry-run`
- 可单独 `uv run train.py --eval-only`
- 不依赖其他 experiment 目录中的 helper import

---

## 4. `dapo-aime24` 实验目录要求

### 4.1 目标目录

本期必须交付：

- `experiments/dapo-aime24/`

### 4.2 最小文件集合

该目录至少应包含：

```text
experiments/dapo-aime24/
├── README.md
├── pyproject.toml
├── train.py
├── autoresearch.sh
├── autoresearch.md
└── data/
    ├── train.jsonl
    ├── eval.jsonl
    └── sources.yaml
```

该目录可以额外包含本地 `.env` 作为运行配置文件，也可以完全依赖 shell 环境变量。
可以额外包含原始数据或 provenance 文件，但不能替代默认训练/评测入口所依赖的物化数据文件。

### 4.3 设计边界

该目录应是单实验目录，而不是通用框架入口。

该目录应回答的核心问题是：

- 使用 DAPO-Math-17k 做 direct GRPO
- 使用完整 AIME 2024 作为冻结 benchmark
- 输出可复现、可自动化迭代的实验结果

该目录不应承担以下职责：

- 通用算法库
- 跨 experiment 共享工具层
- 多算法调度中心

---

## 5. 数据要求

### 5.1 训练数据要求

训练数据必须满足以下要求：

- 来源对应 `BytedTsinghua-SIA/DAPO-Math-17k`
- 默认训练路径为 experiment-local 的 `data/train.jsonl`
- 默认训练路径应指向可直接消费的最终物化格式
- 每条训练样本至少能提供：样本 ID、题目、标准答案、来源

允许脚本兼容额外输入格式，例如 parquet，但这属于兼容能力，不应替代默认训练路径。

### 5.2 评测数据要求

评测数据必须满足以下要求：

- benchmark 对应完整 AIME 2024
- 默认评测路径为 experiment-local 的 `data/eval.jsonl`
- benchmark 数据必须冻结
- benchmark 数据不得在训练过程中被改写
- benchmark 数据应覆盖全部 30 道题

### 5.3 训练集与评测集解耦要求

必须明确区分：

- `data/train.*` 只用于训练
- `data/eval.*` 只用于 benchmark

不得通过临时脚本在运行时拼接 benchmark 数据，也不得把 benchmark 题目混入训练输入路径中而不做说明。

### 5.4 数据来源说明要求

`data/sources.yaml` 必须记录至少以下内容：

- `train_sources`
- `eval_sources`
- 去重规则
- 清洗或字段转换规则
- benchmark 数据来源
- 答案抽取约定
- 备注信息

需求重点是可追溯、可审阅，而不是固定某一套字段命名模板。

---

## 6. `train.py` 主入口要求

### 6.1 单入口要求

`train.py` 必须是 `dapo-aime24` 的唯一主入口。

训练、评测、dry-run、恢复执行都必须通过该脚本触发。

### 6.2 必须支持的能力

`train.py` 必须支持：

- 参数解析
- 训练
- 评测
- `--dry-run`
- `--eval-only`
- 目录驱动的同 run 续跑（`train/checkpoints.jsonl` 最新 `state_path`）与 `--load-checkpoint-path`（仅权重新起训练）
- 输出目录写入
- 结构化指标打印

### 6.3 必须支持的关键参数

至少需要支持以下参数：

- `--train-data`
- `--eval-data`
- `--log-path`
- `--base-model`
- `--rank`
- `--seed`
- `--load-checkpoint-path`
- `--save-state-name`
- `--eval-only`
- `--dry-run`
- `--grpo-steps`
- `--groups-per-batch`
- `--group-size`
- `--rl-learning-rate`
- `--rl-temperature`
- `--rl-max-tokens`
- `--rl-loss`
- `--eval-num-samples`
- `--eval-temperature`
- `--eval-max-tokens`
- `--max-concurrent-requests`

允许增加更多参数，但不应破坏这些基础接口。

### 6.4 运行原则

`train.py` 应遵守以下原则：

- 通过 `import mint` 调用远端 MinT 服务
- 不要求本地启动训练后端
- 如果使用 `.env`，只从 experiment 目录读取，不依赖仓库根目录 fallback
- 也允许完全依赖 shell 环境变量
- 使用 `MINT_BASE_URL` 指定 MinT 远端地址
- 优先使用本地 HF cache
- 在 `--dry-run` 模式下不依赖远端凭证

---

## 7. Benchmark 要求

### 7.1 benchmark 定义要求

`dapo-aime24` 的 benchmark 必须满足以下要求：

- 评测对象是完整 AIME 2024
- prompt 形式应与 DAPO 风格兼容
- 存在稳定的答案抽取逻辑
- 多样本评测时存在明确聚合规则

### 7.2 主指标要求

主指标必须为：

- `eval_accuracy`

自动实验系统应默认以该指标作为主要优化目标。

### 7.3 辅助指标要求

建议输出以下辅助指标：

- `eval_pass_at_k`
- `answer_extract_success`
- `avg_completion_tokens`
- `rl_reward_mean`
- `rl_group_accuracy`
- `rl_answer_extract_success`
- `rl_datums_per_step`

辅助指标用于监控训练稳定性、格式稳定性和采样成本，不替代主指标。

### 7.4 指标输出格式要求

所有可供自动实验系统消费的指标都必须采用如下格式打印：

```text
METRIC eval_accuracy=0.4333
```

如果打印辅助指标，也必须使用相同格式。

### 7.5 多样本评测要求

当 `--eval-num-samples > 1` 时，必须满足以下条件：

- 聚合规则明确
- 聚合规则写入 experiment README
- 主指标口径稳定

对于本期 `dapo-aime24`，推荐默认采用单样本确定性口径；如果启用多样本评测，则推荐采用多数投票口径。

---

## 8. 输出物要求

实验运行后，至少应产出：

- `metrics.json`
- `predictions.jsonl`

如果支持保存训练状态，则还应产出：

- `state_path.txt` 或等价记录文件

这些输出必须落在 experiment-local 的 artifacts 目录中，不应散落在仓库根目录。

---

## 9. `autoresearch` 要求

`dapo-aime24` 必须提供：

- `autoresearch.sh`
- `autoresearch.md`

其中：

- `autoresearch.sh` 必须作为稳定的 current practical-line wrapper
- bare benchmark 确认仍必须可由 `uv run train.py --eval-only ...` 直接重跑；如果 wrapper 不等于该路径，`README.md` 与 `autoresearch.md` 必须写清训练期检查和最终 benchmark 确认命令
- `autoresearch.md` 必须写清主指标、默认配方、实验边界和后续假设

---

## 10. Experiment README 要求

`experiments/dapo-aime24/README.md` 至少应覆盖以下内容：

1. `What this reproduces`
2. `Upstream references`
3. `Training data`
4. `Evaluation data`
5. `Model and MinT setup`
6. `Run training`
7. `Run evaluation`
8. `Expected benchmark output`
9. `Known deviations from upstream`
10. `Reproduction checklist`

本需求文档不替代 experiment README；README 负责说明该实验如何使用和如何复现。

---

## 11. 允许的设计取舍

本期允许以下取舍：

- 奖励函数先采用轻量 exact-match 风格
- 评测脚本采用自包含实现
- benchmark 口径先以稳定和可自动化为优先

这些取舍的前提是：

- benchmark 定义清楚
- 数据来源可追溯
- 偏差在 README 中明确说明

---

## 12. 不允许的情况

本期不允许出现以下情况：

- `--dry-run` 缺失
- `--eval-only` 缺失
- 主指标没有 `METRIC` 输出
- benchmark 数据来源不清楚
- benchmark 数据在训练过程中被修改
- 训练逻辑依赖其他 experiment 目录的共享代码
- 需要先理解仓库其他实验目录才能运行本实验

---

## 13. 验收标准

当以下条件全部满足时，本期需求才算完成。

### 13.1 目录验收

- 存在 `experiments/dapo-aime24/`
- 目录内包含 `README.md`、`pyproject.toml`、`train.py`、`autoresearch.sh`、`autoresearch.md`、`data/`
- `data/` 内存在默认训练数据、默认评测数据和来源说明文件

### 13.2 命令验收

在 `experiments/dapo-aime24/` 目录内，应能执行：

```bash
uv sync
uv run train.py --dry-run
uv run train.py --eval-only
bash autoresearch.sh
```

其中：

- `--dry-run` 不应依赖远端凭证
- `--eval-only` 应固定 benchmark 路径

### 13.3 benchmark 验收

- `train.py` 必须打印 `METRIC eval_accuracy=...`
- benchmark 默认使用冻结的完整 AIME 2024
- 多样本评测的聚合规则必须明确
- 评测结果必须写入 artifacts

### 13.4 工程验收

- `train.py` 支持目录驱动续跑与 `--load-checkpoint-path`；`--eval-only` 通过 `--base-model` 使用 `sampler_path`
- benchmark 入口稳定
- 数据来源可追溯
- experiment 目录可独立理解和运行

---

## 14. 后续阶段方向

在本期需求完成之后，下一阶段可以考虑：

- 多 seed 重复实验
- 比较不同 RL loss
- 引入更严格的数学 verifier
- 基于相同契约新增第二个 AIME experiment

这些属于后续阶段，不应回流修改本期核心目标。

---

## 15. 核心结论

本需求文档只定义一件事：

> 以 `dapo-aime24` 为首个落地对象，规定一个 MinT 上可训练、可评测、可自动实验的 AIME experiment 契约，并以该契约作为后续 experiment 扩展的基础。
