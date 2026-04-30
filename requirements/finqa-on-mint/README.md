# 在 MinT 上跑金融 benchmark：FinQA

## 1. 目标

本需求文档用于定义一个在 MinT 上跑金融 benchmark 的实验；首期 benchmark 选择 `FinQA`，首期 baseline 路线采用 `OpenEnv FinQA` 口径并参考 Daytona 的 `openenv-finqa` 指南。

本 requirement 与仓库级 experiment contract 保持一致。通用目录结构可参考：

- 仓库根目录 `README.md`
- `experiments/README.md`

但以下内容以本 requirement 为准：

- canonical benchmark 入口是 `uv run train.py --eval-only --eval-data <full_eval_path>`
- canonical 训练入口是 `uv run train.py`
- `autoresearch.sh` 是当前 practical line 的自动实验 wrapper；bare benchmark 入口仍是 `uv run train.py --eval-only --eval-data <full_eval_path>`
- `train.py --dry-run` 必须在不依赖 MinT 凭证的前提下完成本地数据与 benchmark contract 校验
- 首期主指标采用 `METRIC eval_success_rate=...`

本期目标实验目录：

- `experiments/finqa/`

本实验要回答的问题：

> 在 MinT 上，先冻结首期采用的 `FinQA` benchmark 的 `OpenEnv FinQA` 实现口径，先跑 eval 看当前可用 base model 的表现，再尝试用 GRPO 在不泄漏 benchmark 的前提下提升金融推理 success rate。

---

## 2. benchmark 口径

本期 benchmark 选择 `FinQA`，首期 baseline 路线采用 `OpenEnv FinQA` 口径，而不是复刻原始 `FinQA` program generation 任务。

这两者必须明确区分：

- 原始 `FinQA`：金融数值推理 / program generation 数据集
- `OpenEnv FinQA`：`FinQA` 的环境化 / agentic 实现口径，以 episode success rate 为核心口径

首期中，原始 `FinQA` 只作为：

- 背景资料
- 后续可选 warm-start / SFT 语料
- provenance 中需要记录的上游来源

对于首期实验，`success` 默认指 episode-level success，而不是答案字符串级 exact match。
如果后续实现中引入答案级指标、子任务级指标、或其他辅助指标，必须将其视为补充指标，不能替代首期主指标。

---

## 3. 搭建原则

`experiments/finqa/` 仍应按当前仓库 experiment contract 搭建，但这里的“遵守 contract”主要指单目录、自包含、以及统一的运行入口，而不是强制套用与 baseline 冲突的数据文件命名。

尽管首期 baseline 路线采用 `OpenEnv FinQA` 的 environment / agent 口径，`experiments/finqa/` 仍然必须遵守本仓库现有 experiment contract，而不是额外定义一套平行的 canonical 运行入口。

对本实验而言，需要明确区分两类约束：

- 必须遵守的 repo 级约束：experiment 目录自包含、`uv run train.py` / `uv run train.py --eval-only --eval-data <full_eval_path>` 作为 bare 训练/benchmark 入口、`autoresearch.sh` 作为当前 practical line 的 canonical wrapper、`--dry-run --eval-data <smoke_eval_path>` 可本地校验、指标以 `METRIC ...` 形式稳定输出
- 允许以 baseline 为先的实现项：`data/` 下的数据文件命名、目录组织、snapshot 清单、task manifest、以及 adapter 文件形态

如果 repo 的标准 split layout（例如 `data/train/full.jsonl` 加显式 `--eval-data <...>`）与 baseline 原生组织方式冲突，则本实验应优先遵从 baseline；如需提供额外的本地 adapter / manifest 文件，也应作为对 baseline 数据组织的补充，而不是替代其 canonical 数据定义。

本 requirement 不重复以下通用内容：

- 标准 scaffold 文件集合
- 通用 `train.py` 参数清单
- 通用 MinT 接口约定
- 通用 `autoresearch.sh` 脚手架约定

本实验只额外强调以下几点：

- benchmark 是 `FinQA`
- 首期 baseline 路线采用 `OpenEnv FinQA` 口径，并参考 Daytona 的 `openenv-finqa` 指南
- 必须先完成 base model eval
- train / eval 必须严格解耦
- 训练结果必须相对固定 benchmark 上的 base model 起点来解释
- benchmark contract 一旦冻结，后续结果必须基于同一口径比较；若 contract 变化，应明确视为新 benchmark 版本

---

## 4. 启动原则

本实验遵守 eval-first 原则：

1. 先实现 `--eval-only`
2. 先确保首期 baseline 路线对应的 benchmark 路径能跑通
3. 先跑至少一轮 base model eval
4. 先记录起点指标与主要失败模式
5. 再进入 RL / GRPO

首期建议优先尝试：

- `Qwen/Qwen3-4B-Instruct`
- `Qwen/Qwen3-8B-Base`

在以下条件满足前，不应宣称训练有效：

- benchmark 路径已冻结
- success 定义已固定
- prompt / tool schema 已固定
- train / eval 泄漏风险已排除或已明确写清
- 已有 base model eval 结果作为起点

---

## 5. baseline 来源与参考实现边界

首期 `OpenEnv FinQA` baseline 的代码与环境准备路径参考 Daytona 官方指南：

- baseline 参考目录：`daytona/guides/python/reinforcement-learning/openenv`
- baseline setup 方式：先 clone Daytona 仓库，再进入上述目录
- baseline 环境与数据准备方式：执行 `python build_snapshot.py`
- 该脚本会在构建 snapshot 时拉取 OpenEnv FinQA 环境依赖，并预下载 FinQA 数据集

但对本仓库而言，Daytona / OpenEnv 的定位必须写清：

- Daytona / OpenEnv 仅作为参考实现与 provenance 来源，不作为 `experiments/finqa/` 的 canonical 运行入口或运行时依赖
- canonical 训练入口与 benchmark 入口必须始终是 `experiments/finqa/` 下的 `uv run train.py` / `uv run train.py --eval-only --eval-data <full_eval_path>`
- canonical 运行路径不得要求在运行时 clone Daytona、执行 `build_snapshot.py`、或在线拉取 benchmark 内容
- 若后续将 baseline 逻辑适配到 `experiments/finqa/`，可以保留 baseline 原生的数据命名与目录组织；但本地实验目录仍必须自包含并可审计
- `data/` 目录中的本地文件可以是 baseline 对齐的数据文件、snapshot 清单、任务索引、adapter manifest、或 provenance 镜像

必须在 `README.md` 与 `data/sources.yaml` 中记录至少以下内容：

- 参考的 Daytona / OpenEnv repo、commit 或 tag、以及 guide 版本
- 对应 benchmark / environment snapshot 的标识信息
- 数据集下载来源与 snapshot 构建方式
- 本地物化、引用、镜像、或适配方式
- 与参考实现保持等价的 contract 项
- 与参考实现存在差异的实现项及原因

---

## 6. benchmark contract

对于首期 `OpenEnv FinQA` 路线，episode-level benchmark contract 应优先与 Daytona / OpenEnv baseline 保持一致；`README.md`、`train.py`、以及本地 eval manifest（如 `data/eval/full.jsonl`）的职责是显式记录、适配、和复现该 contract，而不是重新定义另一套 benchmark。

若本地实现相对参考实现存在差异，必须明确区分两类情况：

- 与 benchmark 可比性直接相关的差异（如 success 定义、tool schema、prompt contract、episode termination）默认不允许静默漂移；如确需变更，应视为新的 benchmark 版本
- 与工程组织相关但不改变 benchmark 含义的差异（如代码结构、artifact 路径、MinT 封装方式）可以本地化，但必须记录

至少需要固定并可审计以下内容：

- task / episode ID
- benchmark 范围定义（例如 full benchmark 或固定子集）
- benchmark 版本或环境快照版本
- success 判定口径
- 可用工具集合与 tool schema
- prompt / system prompt 模板
- 单个 episode 的最大步数、超时、以及失败终止条件
- 是否允许 retry / 重新规划 / 多次提交最终答案
- 评测时的固定 seed、episode 遍历顺序、以及会影响结果的并发策略
- 最终 success 是由环境直接返回，还是由本地 grader 二次判定

如果以上任一项发生变化，应视为 benchmark contract 变化，不得与旧结果直接横向对比；如需比较，必须在文档和指标记录中显式标注 benchmark 版本变化。

---

## 7. 数据要求

首期 benchmark 集合的定义应优先对齐 Daytona / OpenEnv baseline，而不是由本仓库自行发明一套不同的数据组织方式。

如果 baseline 的 benchmark 由环境快照、任务源、episode contract、tool schema、或其他运行时定义共同确定，则本实验应保持与 baseline 一致；本仓库中的本地数据文件仅作为索引、manifest、adapter 输入、或 provenance 镜像，不应擅自改变 benchmark 定义。

在这种前提下，若存在本地 eval 相关文件，它们至少应能审计：

- task / episode ID 或其本地引用
- 对应的 baseline benchmark 来源
- benchmark 范围定义与总 episode 数
- success 判定口径
- 环境或 snapshot 来源
- 与 baseline 对齐的版本信息

训练相关文件不要求命名为 `train.jsonl`，eval 相关文件也不要求命名为 `eval.jsonl`。应优先保留 baseline 原生命名与目录组织；若另加本地 adapter 文件，也要明确它与 baseline 数据源的映射关系。

如果首期 benchmark 不是 baseline 的 full benchmark，而是固定子集，则必须在 `README.md` 与 `data/sources.yaml` 中明确写出子集选择规则、固定 episode ID 清单、以及为何将其视为 canonical benchmark。

首期训练输入必须满足“可审计”要求，二选一：

1. 本地物化训练数据
   - 在 `data/` 下保留 baseline 原生命名的数据文件，或保存本地物化的训练 episode / task 样本
2. baseline 一致的训练任务引用 / 生成规则
   - 在 `data/` 下记录训练任务索引、训练样本引用、生成入口、snapshot 清单、或 baseline 对应的数据引用方式
   - `data/sources.yaml` 明确记录生成器来源、环境快照、固定 seed、过滤规则、以及与 eval benchmark 的解耦规则

无论采用哪种方式，都必须能够审计：

- 训练任务来源
- 训练任务生成或筛选规则
- 与 benchmark eval 集的解耦方式
- benchmark 泄漏风险分析
- train / eval episode 是否重叠的校验结果

如果首期暂时没有可审计的训练 split，可以分阶段推进：

- M0：先交付 eval-only benchmark
- M1：拿到清晰 train split 或训练任务生成规则后，再交付可报告的 RL 路径

`data/sources.yaml` 至少记录：

- `eval_sources`
- `train_sources`
- baseline 参考实现来源（如 Daytona guide / OpenEnv repo）
- 数据下载入口或 snapshot 构建入口
- baseline 原生数据文件名 / 目录组织
- 本地 adapter、manifest、或镜像文件与 baseline 的映射关系
- benchmark 版本或快照信息
- train / eval 解耦规则
- 如果使用原始 `FinQA`，它在本实验中的用途
- success 判定约定
- prompt / tool / environment contract 的版本说明
- benchmark 泄漏风险说明

---

## 8. 指标与验收

首期评价指标应尽可能与 Daytona / OpenEnv baseline 保持一致。

首期主指标默认是：

- `METRIC eval_success_rate=...`

这是 `finqa` 实验相对 repo 默认 `eval_accuracy` 的特例，因为首期 benchmark 口径采用 `OpenEnv FinQA` baseline 的 episode success rate，而不是答案级 accuracy。若 baseline 后续公开了更明确的 canonical metric 命名或附加统计口径，应优先与 baseline 对齐，并在本实验 README 与 run artifact 中显式记录差异。

建议同时输出：

- `METRIC eval_success_count=...`
- `METRIC eval_total_episodes=...`

如需兼容 repo 级通用工具，可额外输出：

- `METRIC eval_accuracy=...`

其中该字段在本实验中作为 `eval_success_rate` 的兼容别名，而不是独立指标。

首期验收分三层：

### M0：benchmark 打通

- `uv run train.py --dry-run --eval-data <smoke_eval_path>` 成功
- `uv run train.py --eval-only --eval-data <full_eval_path>` 成功
- 至少一个 base model 能完成首轮 eval
- 能输出稳定的 `METRIC eval_success_rate=...`
- 能清楚记录当前 benchmark contract
- README 与 `data/sources.yaml` 明确声明 Daytona / OpenEnv 仅为参考实现，而不是运行时依赖
- `--dry-run` 能完成 train / eval overlap audit，或在当前阶段明确说明为何暂不支持

### M1：base model profiling

- 完成 `Qwen/Qwen3-4B-Instruct` 与 `Qwen/Qwen3-8B-Base` 的 eval，或明确记录为何某个模型当前不可用
- 记录 success rate 与主要失败模式
- 冻结首期 benchmark 口径
- 明确记录 episode contract、tool schema、prompt contract、环境快照版本、以及 eval seed / 并发等稳定性约束
- 明确记录参考的 Daytona / OpenEnv repo、commit 或 tag、guide 版本、以及与参考实现的关键差异

### M2：GRPO 首跑

- 在不泄漏 benchmark 的前提下启动 GRPO
- 训练结果可以和已记录的 base model 起点对比
- 至少得到一次可审计的训练后结果
- 明确说明训练输入来源与 benchmark 解耦方式
- 明确说明本次训练基于哪个 canonical base model，并保持与对比基线一致

外部资料中的 `50%+` success rate 可以作为参考上界或 stretch goal，但不作为首期硬验收线。

---

参考资料：

- `https://github.com/czyssrs/FinQA`
- `https://arxiv.org/abs/2109.00122`
- `https://www.daytona.io/docs/en/guides/reinforcement-learning/openenv-finqa/`
- `https://snorkel.ai/blog/building-finqa-an-open-rl-environment-for-financial-reasoning-agents/`
- `https://openenv.dev/benchmarks/finqa`

---

## 9. 二期目标：尽可能复现 Daytona 官方 openenv-finqa 实验

在不改动首期 requirement 定位的前提下，二期目标进一步收敛为：

> 在保持 `mint-cookbook` 的 repo-native 入口与单目录自包含约束下，尽可能复现 Daytona 官方文档 `https://www.daytona.io/docs/en/guides/reinforcement-learning/openenv-finqa/` 中的完整 `openenv-finqa` 多轮工具调用 RL 实验。

这意味着二期目标不再满足于：

- 只在同一份 `finqa.csv` 上做本地单轮问答评测
- 只对齐数据文件与主指标命名
- 只做一个“与 OpenEnv 大致相似”的本地适配器

而是要尽量复现 Daytona 文档里的以下实验语义。

### 9.1 二期主要复现对象

二期主要复现对象包括：

- multi-turn episode protocol
- `get_descriptions` / `get_table_info` / `sql_query` / `submit_answer` 四工具交互
- 二值 episode reward（正确 `1.0`，错误 `0.0`）
- Daytona 文档中的 system prompt 与答案格式约束
- GRPO + LoRA 的训练路径
- group-based advantage normalization
- 训练中的 rollout collection、grouping、adapter hot-swap、pipeline overlap 等核心语义

允许使用 MinT 替代 Daytona 文档中的部分底层训练与推理实现，但不应静默改变实验含义。

### 9.2 二期 benchmark / eval 目标

二期的 `uv run train.py --eval-only --eval-data <full_eval_path>` 应尽量逼近 Daytona 文档中的真实多轮工具调用评测，而不是长期停留在：

- 单轮 completion
- 本地 grader 近似 reward
- 无真实 tool-calling trace 的本地 adapter eval

二期至少应尽量固定并审计：

- episode reset / step / done 语义
- tool schema 获取方式
- tool call 解析方式
- 最大步数与失败终止条件
- 最终答案提交格式
- reward 来源是环境直接返回，还是本地近似
- eval seed、episode 顺序与并发策略

### 9.3 二期训练目标

二期的 `uv run train.py` 应尽量复现 Daytona 文档中的训练路径，而不是仅作为占位入口。

重点包括：

- 多轮 rollout collection
- 以同题多采样构造 strict prompt groups
- GRPO 优势标准化
- turn-level policy gradient update
- LoRA adapter 导出与热切换
- rollout 与 update 的 overlap / pipeline 化
- 训练 artifacts 的完整落盘

在资源允许的情况下，优先贴近 Daytona 文档中公开的参考设置：

- base model：`Qwen3-14B`
- LoRA
- GRPO
- 默认训练配置中约 `500` sandboxes、`10` iterations、`group-size 6`

如果本地资源无法完全对齐，必须在 README 与 run artifacts 中明确记录差异来源。

### 9.4 二期结果复现目标

二期不仅关注最终 `eval_success_rate`，还应尽量关注 Daytona 文档中明确指出的行为变化是否出现，包括：

- year-string column quoting 修复
- `SELECT *` 错误消失
- single numeric answer formatting 提升
- parenthetical negative notation 解释为负数
- empty result 后的 adaptive recovery
- company / table-specific convention 学习
- shortcut arithmetic 的出现
- late-stage failures 主要转向 arithmetic / interpretation，而不再是 retrieval

如果结果声称“接近复现 Daytona 文档”，至少应通过样例轨迹或失败分析说明这些行为改进中的若干项确实出现。

### 9.5 二期验收建议

可将二期内部再拆为三个阶段：

#### P2-M0：真实 multi-turn eval

- `--eval-only` 路径具备真实多轮工具调用
- 输出可审计 traces
- prompt / tool / reward / termination contract 冻结

#### P2-M1：训练路径打通

- `uv run train.py` 能启动可审计训练
- 训练结果能够相对 base-model 起点解释
- 输出 per-iteration metrics、rollout summaries、trajectories

#### P2-M2：结果趋势与行为接近 Daytona 文档

- success rate 相比 base model 明显提升
- 训练后 failure modes 发生与 Daytona 文档同方向的迁移
- 若与公开结果差距较大，必须解释差距来自模型、训练预算、环境协议、或底层执行差异

### 9.6 二期公开参考目标

Daytona 文档中的公开参考结果可作为二期的重要对照目标：

- 默认训练参数下，`Qwen3-14B + LoRA` 在约 10 个 iteration 中，accuracy / success rate 从约 `~21%` 提升到约 `~52%`

该结果不应被当作必须逐点精确复刻的硬门槛，但应作为二期最重要的公开参考曲线。
如果最终结果显著偏离，应明确说明是否由于以下原因导致：

- 模型不同
- 训练预算不同
- 多轮环境协议未完全复现
- reward / prompt / rollout / grouping 存在差异
- MinT 与 Daytona 底层实现不同
