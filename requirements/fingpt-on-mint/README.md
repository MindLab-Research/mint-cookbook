# 在 MinT 上跑金融 benchmark：FinGPT

## 1. 目标

本需求文档用于定义一个在 MinT 上启动 `FinGPT` 方向金融 benchmark 实验的首期目标。

本 requirement 与仓库级 experiment contract 保持一致。通用目录结构可参考：

- 仓库根目录 `README.md`
- `experiments/README.md`

但以下内容以本 requirement 为准：

- canonical benchmark 入口是 `uv run train.py --eval-only`
- canonical 训练入口是 `uv run train.py`
- `autoresearch.sh` 是当前 practical line 的自动实验 wrapper；bare benchmark 入口仍是 `uv run train.py --eval-only`
- `train.py --dry-run` 必须在不依赖 MinT 凭证的前提下完成本地数据与 benchmark contract 校验
- 首期主指标采用 `METRIC eval_accuracy=...`
- 首期必须先冻结一个可运行、可审计、可扩展的 FinGPT benchmark slice，再决定是否进入训练

本期目标实验目录：

- `experiments/fingpt/`

本实验要回答的问题是：

> 在 MinT 上，先把 `FinGPT` 论文家族对应的金融 instruction benchmark 固定成一个可跑的 eval-first 实验；首期先选一个官方发布且本地容易冻结的 benchmark slice 跑通 `--dry-run` 和 `--eval-only`，拿到 base-model 起点，再决定是否扩展到多任务 instruction tuning 或监督训练。

---

## 2. benchmark 口径

### 2.1 需要明确区分的三层对象

本 requirement 必须明确区分以下三层对象：

1. `target method family`
   - 指 `FinGPT` 论文所代表的金融 instruction-tuning / benchmark 路线
2. `current runnable benchmark slice`
   - 指首期为了 eval-first 落地而选择的一个具体、冻结、可下载、可审计的 benchmark 子集
3. `future full-scope benchmark ambition`
   - 指后续是否要把实验从一个 slice 扩到更完整的 FinGPT benchmark 多任务集合

首期 requirement 不允许把这三层对象混写成“已经完整复现 FinGPT benchmark”。

### 2.2 首期 benchmark 选择

首期 benchmark 选择如下：

- `target method family`：`FinGPT`（OpenReview 论文）
- `current runnable benchmark slice`：官方 Hugging Face 数据集 `FinGPT/fingpt-fineval`
- benchmark 任务形态：中文金融单项选择题 / instruction-following multiple choice

之所以把 `fingpt-fineval` 设为首期 runnable slice，而不是一开始就实现论文中的完整多任务 benchmark，是因为：

- 当前仓库的实验生命周期要求先固定一个可运行的 eval-only benchmark path
- `fingpt-fineval` 已经具备公开数据集、标准 train/test split、以及统一的 instruction / input / output 结构
- 相比直接把 sentiment / headline classification / NER / relation extraction 一次性拼进一个多任务 eval，Fineval 更适合作为第一个冻结 benchmark slice

### 2.3 这不是对论文范围的缩写替代

本 requirement 必须明确声明：

- 论文中的 `FinGPT` benchmark 范围大于首期 `fingpt-fineval` slice
- 当前把 `fingpt-fineval` 设为首期 benchmark，是一个工程分期与 eval-first 落地选择
- 这不等于声称 `FinGPT` 论文只包含 `fingpt-fineval`

如果后续扩展到更完整的 FinGPT benchmark 集合，必须把 benchmark 版本与 scope 的变化明确写入 README 和 run artifact。

### 2.4 首期执行矩阵：train / eval / profiling 必须写清

首期 requirement 必须把“在哪个数据集上 train、在哪个数据集上 eval、先跑什么 baseline、SFT 后和谁比较”写成明确矩阵，而不是散落在 README 或后续讨论里。

当前首期执行矩阵固定如下：

| 阶段 | train 数据 | eval 数据 | 目的 | 备注 |
| --- | --- | --- | --- | --- |
| M0 `dry-run` | 无正式训练；允许读取 `data/smoke_train.jsonl` 仅做结构校验 | `data/fingpt-fineval/test.jsonl` 优先；缺失时允许 `data/smoke_eval.jsonl` 仅做本地验证 | 校验数据结构、prompt、grader、overlap audit | smoke 只用于本地验证，不可报告结果 |
| M1 `base-model eval` | 不训练 | `data/fingpt-fineval/test.jsonl` | 建立 official benchmark slice 的 zero-shot / direct-inference 起点 | 这是首期必须先拿到的可报告 baseline |
| M2 `slice-local SFT`（如启用） | `data/fingpt-fineval/train.jsonl` | `data/fingpt-fineval/test.jsonl` | 先回答“只在 Fineval slice 上做 SFT 能提升多少” | 若启用这一路线，结果只解释为 Fineval-slice SFT，不等于完整 FinGPT multi-task reproduction |
| M3 `full-scope multi-task SFT`（后续） | Fineval 之外再加入其他 FinGPT benchmark tasks | 至少保留 `fingpt-fineval/test.jsonl` 的独立评测 | 逼近更完整的 FinGPT 论文路线 | 一旦进入这条路线，必须升级 requirement scope，而不是沿用当前 M0/M1 文案 |

因此首期 requirement 的默认答案应当非常明确：

- 首期 eval：跑 `FinGPT/fingpt-fineval` 的 `test` split
- 首期若开 SFT：先跑 `FinGPT/fingpt-fineval` 的 `train` split，eval 仍固定在同一 official `test` split
- 首期不允许一边说“在 Fineval 上 eval”，一边暗中把 sentiment / NER / RE / headline 数据混入训练而不更新 requirement

### 2.5 首期必须补清楚的模型结果表

对于 `fingpt-fineval`，需求文档里必须把结果分成三类，而不是笼统写“Qwen 效果如何”：

1. `publicly reported results`
   - 公开资料里已经能直接引用的结果
2. `repo-local required baselines`
   - 官方没给、但本项目必须自己先跑出来的 baseline
3. `repo-local required post-SFT results`
   - 如果项目决定开 SFT，本项目必须自己补齐的训练后结果

按当前已核对的公开资料，首期 requirement 应明确写成：

- `FinGPT` 论文与官方 repo **公开了** Qwen-7B 在 SA / NER / HC / RE 上的 task-specific 和 multi-task 指标
- 官方 repo **公开了** `fingpt-mt_qwen-7b_lora` 这个 multi-task LoRA 模型
- 但截至当前 requirement 版本，**没有在官方论文或官方 README 中查到** `fingpt-fineval` 上的 Qwen base-model zero-shot 分数表
- 同样，截至当前 requirement 版本，**没有在官方论文或官方 README 中查到** `fingpt-fineval` 上的 Qwen SFT 后分数表

这意味着 requirement 不能把“官方已经给了 Fineval 上的 Qwen 基线和 SFT 结果”写成既成事实。
相反，应该明确区分：

- `公开已知`：Qwen 在论文其他任务上的 SFT / multi-task 结果存在
- `当前未知`：Qwen 在 `fingpt-fineval` 上的官方 zero-shot 与 SFT 后结果未见公开表格
- `本项目必须补齐`：在本地 benchmark contract 下先跑 Qwen baseline，再决定是否做 Fineval-local SFT

### 2.6 requirement 中必须固定的结果目标字段

从本 requirement 起，`experiments/fingpt/README.md` 与 run artifact 至少要维护下表。即使某些字段暂时为空，也必须明确写成 `pending`，而不是省略：

| 结果槽位 | 数据集 | 模型 | 训练状态 | 指标 | 当前状态 |
| --- | --- | --- | --- | --- | --- |
| Public baseline | `fingpt-fineval/test` | Qwen-related base model | zero-shot | `eval_accuracy` | `pending / not found in official sources` |
| Repo baseline #1 | `fingpt-fineval/test` | `Qwen/Qwen3-4B-Instruct-2507` | zero-shot | `eval_accuracy` | 必跑 |
| Repo baseline #2 | `fingpt-fineval/test` | `Qwen/Qwen2.5-7B-Instruct` 或另一条中文强基线 | zero-shot | `eval_accuracy` | 建议跑 |
| Repo post-SFT #1 | `fingpt-fineval/train -> test` | 与 Repo baseline #1 同 base model | slice-local SFT | `eval_accuracy` | 如启用 SFT 则必跑 |
| Repo post-SFT #2 | `full-scope multi-task train -> fingpt-fineval/test` | 与 repo 选定训练主线一致 | multi-task SFT | `eval_accuracy` + 其他任务指标 | 后续 scope 升级时再定义 |

只要这些字段没有写清楚，requirement 就不能算完成。

---

## 3. 搭建原则

`experiments/fingpt/` 仍应按当前仓库 experiment contract 搭建，但这里额外强调以下几点：

- experiment 目录必须自包含
- canonical eval 入口必须是 `uv run train.py --eval-only`
- canonical train 入口必须是 `uv run train.py`
- `autoresearch.sh` 必须是当前 practical line 的自动实验 wrapper；如果它不等于 bare benchmark 入口，`README.md` 与 `autoresearch.md` 必须写清两者分工
- 首期默认是 `eval-first`
- 训练在 benchmark contract 冻结之前必须保持关闭或明确 gated
- 任何 smoke / toy / hand-written 示例数据都只能用于本地验证，不得当成 benchmark 结果

当前本地 contract 已经分成两条线：`Fineval` 是 benchmark anchor，`sentiment` 是当前维护的 practical wrapper line。两条线的数据、指标、reportable rerun 路径和自动化 wrapper 目标都必须在 experiment 文档中分开写清。

如果 repo 默认脚手架中的 `data/train.jsonl` / `data/eval.jsonl` 与官方 slice 的原生组织方式冲突，本实验应优先保留 benchmark-friendly 的本地数据布局，并在 `README.md` 与 `data/sources.yaml` 中写清映射关系。

---

## 4. 启动原则

本实验遵守 eval-first 原则：

1. 先实现 `--dry-run`
2. 先实现 `--eval-only`
3. 先固定首期 benchmark slice 的本地数据布局与 prompt / grading contract
4. 先跑至少一轮 base model eval
5. 先记录起点指标与主要失败模式
6. 再决定是继续保持 eval-only，还是进入训练

首期建议优先尝试：

- `Qwen/Qwen3-4B-Instruct-2507`
- 如资源允许，可补一条更强中文指令模型 profiling 线

在以下条件满足前，不应宣称训练有效：

- benchmark 路径已冻结
- benchmark 数据已本地物化
- prompt / output parser / grader 已固定
- smoke fallback 与 official benchmark 的区别已写清
- 已经有 official benchmark 的 base model 起点结果

---

## 5. baseline 来源与参考实现边界

首期 requirement 依赖以下三类公开来源：

1. `FinGPT` 论文（OpenReview）
2. `AI4Finance-Foundation/FinGPT` 官方仓库
3. `FinGPT` 官方 Hugging Face 数据集与模型发布页

但对本仓库而言，这些来源的职责必须写清：

- OpenReview 论文用于定义 target method family 和 benchmark 研究背景
- 官方 GitHub 仓库用于核对公开 benchmark / dataset / model 发布资产
- 官方 Hugging Face 数据集用于本地物化首期 runnable benchmark slice
- canonical 运行入口始终是 `experiments/fingpt/` 下的 `uv run train.py --eval-only` 与 `uv run train.py`
- canonical 运行路径不得依赖运行时在线浏览 repo 文档才能理解 benchmark contract

至少应在 `README.md` 与 `data/sources.yaml` 中记录以下内容：

- OpenReview 论文链接
- 官方 GitHub 仓库链接
- 官方 Hugging Face benchmark slice 链接
- 本地 train / test JSONL 的生成方式
- 当前 benchmark slice 和论文完整 benchmark family 的边界
- 是否存在 smoke fallback，以及它的用途

---

## 6. benchmark contract

对于首期 `fingpt-fineval` 路线，benchmark contract 至少需要固定以下内容：

- task / row ID
- train / test split 定义
- 行数据字段结构（至少 `instruction`、`input`、`output`）
- prompt 渲染模板
- multiple-choice 输出解析规则
- metric 聚合方式
- eval 并发策略
- official benchmark 与 smoke fallback 的优先级规则

如果以上任一项发生变化，应视为 benchmark contract 变化，不得与旧结果直接横向对比；如需比较，必须在文档和 run artifact 中显式标注 benchmark 版本或 scope 的变化。

### 6.1 首期任务语义

首期 benchmark 任务语义固定为：

- 输入：金融领域中文 instruction + question/options
- 输出：单个正确选项，允许模型输出完整选项文本或单个选项字母
- grader：优先按选项字母判分，必要时再回退到归一化文本匹配

### 6.2 首期主指标

首期主指标默认是：

- `METRIC eval_accuracy=...`

建议同时输出：

- `METRIC eval_correct=...`
- `METRIC eval_total=...`

如果后续扩展到多任务 FinGPT benchmark，允许引入 per-task metric 与 aggregated benchmark score，但必须保留首期 official slice 的独立主指标口径。

---

## 7. 数据要求

### 7.1 official benchmark 数据要求

首期 official benchmark 数据必须满足：

- 来源对齐 `FinGPT/fingpt-fineval`
- 默认 benchmark 输入保存在 `experiments/fingpt/data/fingpt-fineval/` 下
- 至少包含：
  - `train.jsonl`
  - `test.jsonl`
- 默认 `--eval-only` 不得在运行时临时在线抓取 benchmark
- benchmark 范围、字段映射、下载入口必须记录在 `data/sources.yaml`

### 7.2 smoke fallback 数据要求

为了保证 `--dry-run` 和本地脚手架验证可先跑通，允许保留少量 smoke fallback 数据，例如：

- `data/smoke_train.jsonl`
- `data/smoke_eval.jsonl`

但必须满足：

- 这些文件只用于本地验证
- 不得被写成 benchmark 默认来源，除非 official 数据缺失时显式触发 fallback
- 一旦 fallback 被使用，`run.json` 必须明确记录当前 run 不是 official benchmark run
- README 中必须明确写出“smoke 结果不可报告为 benchmark 结果”

### 7.3 训练数据要求

首期训练可以暂时不启用，但 requirement 必须先定义训练开启前提。

训练启用前，至少需要明确：

- 训练是否只使用 `fingpt-fineval` train split，还是扩展到更完整的 FinGPT 多任务 instruction 数据
- train / eval overlap 的审计规则
- 如果扩展到论文范围内的其他任务，如何定义多任务采样与多任务 metric
- 如果训练目标是逼近论文中的 instruction-tuned FinGPT family，而不是只提高 Fineval accuracy，如何记录训练数据来源和作用边界

### 7.4 `data/sources.yaml` 最低要求

`data/sources.yaml` 至少记录：

- `eval_sources`
- `train_sources`
- OpenReview 论文来源
- 官方 GitHub 来源
- 官方 Hugging Face benchmark slice 来源
- 本地物化脚本入口
- official / smoke 的用途边界
- train / eval 解耦规则
- benchmark scope 备注

---

## 8. 指标与验收

首期验收分三层：

### M0：benchmark 打通

- `uv run train.py --dry-run` 成功
- `uv run train.py --eval-only` 成功
- 至少一个 official benchmark slice 能完成首轮 eval
- 能稳定输出 `METRIC eval_accuracy=...`
- `README.md` 与 `data/sources.yaml` 明确声明当前 scope 是 `FinGPT` family 下的 `fingpt-fineval` runnable slice
- `run.json` 能区分 official run 与 smoke fallback run
- `--dry-run` 能完成 train / eval overlap audit，或在当前阶段明确说明为何暂不支持

### M1：base model profiling

- 在 official benchmark slice 上完成至少一条 base model eval
- 最好至少完成两条：
  - `Qwen/Qwen3-4B-Instruct-2507`
  - `Qwen/Qwen2.5-7B-Instruct` 或另一条可公开对齐的中文基线
- 记录 accuracy 与主要失败模式
- 冻结首期 benchmark contract
- 明确记录 prompt contract、grader contract、split contract、并发策略
- 必须把结果写入“base model 效果表”，而不是只在实验日志中零散出现

### M2：训练路径决策

- 明确是保持 eval-only，还是进入监督训练
- 如果进入训练，首条默认训练路线必须先回答：
  - train 是否仅使用 `fingpt-fineval/train.jsonl`
  - eval 是否仍固定在 `fingpt-fineval/test.jsonl`
  - 结果解释是否仅限于 Fineval slice
- 如果进入训练，必须先写清训练数据来源与目标
- 训练结果必须至少形成一张明确对比表：
  - `base model zero-shot on fingpt-fineval/test`
  - `same base model after SFT on fingpt-fineval/train -> test`
- 训练结果必须相对同一 official benchmark contract 下的 base model 起点来解释
- 如果训练范围超出 `fingpt-fineval`，必须把 requirement 升级为更完整的 FinGPT benchmark requirement，而不是静默扩 scope

---

## 9. 输出物要求

实验运行后，至少应产出：

- `predictions.jsonl`
- `eval_metrics.json`
- `run.json`

这些输出必须落在 experiment-local 的 artifacts 目录中，不应散落在仓库根目录。

`run.json` 至少应记录：

- 当前 benchmark scope
- official 或 smoke 模式
- 实际 eval 数据路径
- CLI 参数
- benchmark contract 相关元信息

---

## 10. `autoresearch` 要求

`fingpt` 必须提供：

- `autoresearch.sh`
- `autoresearch.md`

其中：

- `autoresearch.sh` 必须作为稳定的 current practical-line wrapper
- bare benchmark 确认仍必须可由 `uv run train.py --eval-only ...` 直接重跑；如果 wrapper 不等于该路径，`README.md` 与 `autoresearch.md` 必须写清 wrapper 目标、训练期检查所用 eval、以及最终 benchmark / held-out confirmation rerun 的命令
- `autoresearch.md` 必须写清主指标、当前 wrapper recipe、benchmark scope、默认 profiling 命令、以及 smoke fallback 的不可报告性

---

## 11. Experiment README 要求

`experiments/fingpt/README.md` 至少应覆盖以下内容：

1. `What this reproduces`
2. `Current runnable scope`
3. `Why the current scope is narrower than the full FinGPT paper family`
4. `Official benchmark data`
5. `Smoke fallback data`
6. `Run dry-run`
7. `Run evaluation`
8. `Expected benchmark output`
9. `Current training status`
10. `References`

本 requirement 不替代 experiment README；README 负责说明该实验如何使用和如何复现。

---

## 12. 允许的设计取舍

本期允许以下取舍：

- 首期只固定 `fingpt-fineval` 一个 runnable slice
- 首期 grader 先采用 multiple-choice 解析 + exact option 匹配
- 首期训练保持关闭，先以 benchmark 打通与 profiling 为目标
- 允许保留少量 smoke fallback 数据用于本地验证

这些取舍的前提是：

- benchmark 定义清楚
- official 与 smoke 的边界写清
- 偏差在 README 中明确说明

---

## 13. 不允许的情况

本期不允许出现以下情况：

- `--dry-run` 缺失
- `--eval-only` 缺失
- 主指标没有 `METRIC` 输出
- official benchmark 数据来源不清楚
- smoke fallback 被当成 benchmark 结果报告
- 训练逻辑在 benchmark contract 未冻结时提前启用
- 需要先理解仓库其他 experiment 目录才能运行本实验
- 用“已经复现 FinGPT 全 benchmark”描述实际上只有 Fineval slice 的实现

---

## 14. 后续阶段方向

在本期 requirement 完成之后，下一阶段可以考虑：

- 在同一 experiment 中扩到更多 FinGPT benchmark tasks
- 引入更完整的多任务 instruction-tuning 训练路径
- 对比不同中文金融模型在 official slice 上的表现
- 如果 scope 扩到论文中的多任务 benchmark，新增 aggregated benchmark score 与 per-task metrics

这些属于后续阶段，不应回流修改本期核心目标。

---

## 15. 核心结论

本需求文档只定义一件事：

> 以 `experiments/fingpt/` 为落地对象，先在 MinT 上冻结一个属于 `FinGPT` family 的可运行金融 benchmark slice（首期选择 `fingpt-fineval`），并以 eval-first 的方式完成 benchmark 打通、base model profiling、以及后续训练决策前的契约固定。

---

参考资料：

- `https://openreview.net/forum?id=FuOMomaQa8`
- `https://openreview.net/pdf?id=FuOMomaQa8`
- `https://github.com/AI4Finance-Foundation/FinGPT`
- `https://huggingface.co/datasets/FinGPT/fingpt-fineval`
- `https://huggingface.co/FinGPT/datasets`
