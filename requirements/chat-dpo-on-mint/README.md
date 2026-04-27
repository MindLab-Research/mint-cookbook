# 在 MinT 上搭建聊天偏好 DPO 实验：chat-dpo

## 1. 目标

本需求文档用于定义一个在 MinT 上启动聊天质量偏好优化实验的首期目标。

本 requirement 与仓库级 experiment contract 保持一致，但当前实验明确选择 **直接使用 `mint` 后端**，而不是先走 `tinker` 再迁移。
这不是仓库默认建议，而是当前需求的显式选择。

本期目标实验目录：

- `experiments/chat-dpo/`

该实验要回答的问题是：

> 在 MinT 上，能否把历史 `chat_dpo` 仓库里的聊天质量 DPO 训练思路，重构成 `mint-cookbook` 风格的单文件、eval-first、自包含实验，并以本地 preference pairs 为训练和 held-out preference pairs 为评测基准，形成一个可审查、可扩展、可自动实验的 DPO 基线？

---

## 2. 范围

### 2.1 本期必须覆盖

本期必须覆盖以下内容：

- `requirements/` 下的需求定义
- `experiments/chat-dpo/` 自包含实验目录
- 本地 `data/train.jsonl` / `data/eval.jsonl` 偏好对数据 contract
- 单入口 `train.py`
- `--dry-run`
- `--eval-only`
- DPO 训练模式
- checkpoint / sampler checkpoint 保存能力
- `METRIC ...` 输出
- `autoresearch.sh`
- 与 benchmark 口径一致的 `README.md` / `autoresearch.md` / `data/sources.yaml`

### 2.2 本期不要求覆盖

本期不要求覆盖以下内容：

- 通用 DPO 框架
- 在线 preference 生成
- PM blind eval 工作室整合
- 多 benchmark 并行体系
- 多算法统一入口 (`sft|dpo|grpo`)
- 复杂 async rollout 系统

---

## 3. benchmark 口径

### 3.1 首期 benchmark 不是生成式公开 benchmark

本实验首期 benchmark 不是像 LawBench / AIME 那样的外部公开生成 benchmark。

首期 benchmark 固定为：

- 本地 held-out preference pairs
- 用模型对 `chosen` 和 `rejected` 的 completion logprob 进行对比
- 以 held-out pairwise accuracy 作为主指标

也就是说，当前 experiment 的 `--eval-only` 路径要回答的是：

> 对于未参与训练的偏好对，当前模型是否更偏向 `chosen` 而不是 `rejected`？

### 3.2 首期主指标

首期主指标固定为：

- `METRIC eval_pair_accuracy=...`

陪跑指标至少包括：

- `METRIC eval_margin=...`
- `METRIC eval_chosen_score=...`
- `METRIC eval_rejected_score=...`
- `METRIC eval_num_pairs=...`

其中：

- `eval_pair_accuracy` = held-out preference 对中，模型更偏向 `chosen` 的比例
- `eval_margin` = `chosen_score - rejected_score` 的均值
- `eval_chosen_score` / `eval_rejected_score` = completion logprob 聚合值的均值

### 3.3 benchmark 与训练必须解耦

必须明确区分：

- `data/train.jsonl` 只用于 DPO 训练
- `data/eval.jsonl` 只用于 held-out preference benchmark

任何 train/eval 重叠都必须在 `--dry-run` 中被审计并显式报出。

---

## 4. 数据 contract

### 4.1 canonical 行格式

首期 canonical pair 行格式固定为：

```json
{
  "pair_id": "pair-0001",
  "messages": [{"role": "user", "content": "..."}],
  "chosen": [{"role": "assistant", "content": "..."}],
  "rejected": [{"role": "assistant", "content": "..."}],
  "group_id": "prompt-42",
  "source": "dataset-name",
  "metadata": {}
}
```

字段语义：

- `messages`：共享 prompt 前缀
- `chosen`：偏好更高的 completion
- `rejected`：偏好更低的 completion
- `group_id`：可选，用于把重复 prompt 样本分散到不同 batch
- `pair_id`：稳定行 ID

### 4.2 兼容旧格式

为了平滑迁移历史 `chat_dpo` 数据，本实验可以兼容以下旧格式：

1. 顶层 `prompt` / `chosen` / `rejected`
2. 顶层 `prompt_conversation` / `completion_A` / `completion_B` + `label`
3. 嵌套 `comparison` + `label`

但 README 与 requirement 的 canonical 文档仍以 `messages/chosen/rejected` 为准。

### 4.3 数据要求

至少必须记录：

- 训练数据来源
- held-out eval 数据来源
- 去重与过滤规则
- preference 标签来源
- train/eval 解耦规则
- overlap audit 规则
- `group_id` 生成规则

这些信息统一记录在：

- `experiments/chat-dpo/data/sources.yaml`

---

## 5. 实验目录要求

首期必须交付：

```text
experiments/chat-dpo/
├── README.md
├── pyproject.toml
├── train.py
├── autoresearch.sh
├── autoresearch.md
├── tests/
│   └── test_train.py
└── data/
    ├── train.jsonl
    ├── eval.jsonl
    ├── sources.yaml
    └── README.md
```

该目录应满足：

- 可单独 `uv sync`
- 可单独 `uv run train.py --dry-run`
- 可单独 `uv run train.py --eval-only`
- 可单独 `uv run train.py`
- 不依赖其他 experiment 的 helper import

---

## 6. `train.py` 要求

### 6.1 单入口要求

`train.py` 必须是唯一主入口。

必须通过它支持：

- `--dry-run`
- `--eval-only`
- DPO 训练
- 同目录自动 same-run resume（复用同一个 `--log-path`）
- `--load-checkpoint-path` fresh weight-only start
- checkpoint / sampler checkpoint 保存
- 结构化 metrics 输出

### 6.2 runtime 选择

本实验当前 requirement 明确指定：

- 使用 `mint` Python SDK
- 环境变量使用 `MINT_API_KEY` / `MINT_BASE_URL`
- 超时 flag 使用 `--mint-timeout`

README、CLI、backend compatibility 代码必须保持一致，不允许混写 `tinker` / `mint`。

### 6.3 DPO 训练实现要求

当前实验的 DPO 训练主线允许使用 MinT 的 custom loss 路径。

训练 loop 至少应包含：

1. preference pair -> chosen/rejected datum 构建
2. reference model logprob 计算
3. DPO loss 计算
4. `optim_step`
5. 周期性 held-out eval
6. checkpoint 保存

### 6.4 dry-run 要求

`--dry-run` 必须在不依赖 MinT 凭证的前提下完成：

- schema 校验
- pair 数量统计
- overlap audit
- prompt preview
- grouping 统计
- 基本长度统计

---

## 7. artifacts 要求

必须保留仓库标准 eval artifacts：

- `run.json`
- `console.log`
- `eval/examples.jsonl`
- `eval/predictions.jsonl`
- `eval/metrics.json`

训练时还应增加：

- `train/metrics.jsonl`
- `train/checkpoints.jsonl`
- `eval/periodic.jsonl`
- `train/batches.jsonl`

fresh run 需要清空 append-only JSONL；resume 才允许续写。

---

## 8. 验收标准

首期 requirement 完成的最低标准：

1. `experiments/chat-dpo/` 已创建且自包含
2. `train.py` 支持 dry-run / eval-only / 训练三种模式
3. 训练和评测都走 `mint` 后端
4. requirement / README / CLI / artifact contract 自洽
5. 至少有本地单元测试覆盖核心 contract

---

## 9. 当前结论

本期推荐路线非常明确：

- 不再沿用旧 `chat_dpo` 的 `src/train_dpo_tinker.py + logs/` 结构
- 直接把它重构成 `mint-cookbook` 原生 experiment
- 用 held-out preference eval 作为首期 benchmark contract
- 用单文件 `train.py` 实现 DPO 基线
- 在一个 experiment 跑稳之后，再决定是否抽象出 `scaffolds/profiles/dpo.md`
