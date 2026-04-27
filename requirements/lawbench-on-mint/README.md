# 在 MinT 上复现中国法律 benchmark: LawBench

## 1. 目标

本 requirement 定义 `experiments/lawbench/` 的主线, 并明确区分论文目标线与当前 cookbook 可执行线。

从 `2026-04-23` 这次补充调研起, `experiments/lawbench/` 的论文目标线固定为:

- `LawLLM-7B`
- base model: `Qwen/Qwen2.5-7B-Instruct`
- train data: `DISC-Law-SFT`
- training method: `SFT`

也就是说, 当前 repo 对 LawBench 的主线不是“任意 legal SFT 都可以”, 而是:

- 论文目标: 复现 `DISC-LawLLM` 公开路线
- 目标模型线: `LawLLM-7B`, 即基于 `Qwen/Qwen2.5-7B-Instruct` 的法律 SFT 路线
- 当前 cookbook 可执行线: 先用 `Qwen/Qwen3-4B-Instruct-2507` 跑通 `DISC-Law-SFT -> SFT -> LawBench official eval`

原因很直接:

- 这条线同时公开了模型, 训练数据, 训练配置, 以及可直接运行的 SFT 命令
- 它比 `Qzhou-Law` 更适合作为当前仓库的首个可审计复现目标
- 它和本仓库现有单实验 `train.py` 结构更匹配, 不依赖多 agent 或复杂系统外设
- 当前 `Qwen/Qwen3-4B-Instruct-2507` 执行线不是新的论文目标, 而是为了在 mint-cookbook 里先把同一条 train/eval 流水线低成本跑通

本 requirement 保持以下稳定约束:

- canonical benchmark 入口是 `uv run train.py --eval-only`
- canonical 训练入口是 `uv run train.py`
- `autoresearch.sh` 是当前 practical SFT line 的 canonical search wrapper: 它保留 `data/eval/train_eval_200.jsonl` 作为周期训练评测 slice, 但 wrapper 的最终 post-SFT eval 仍以 `data/eval/full.jsonl` 为准, 且不替代 bare full-benchmark 入口 `uv run train.py --eval-only`
- `train.py --dry-run` 必须在不依赖 MinT 凭证的前提下完成本地数据, prompt, metric, 以及 train/eval overlap 校验
- 首期主指标采用 `METRIC eval_lawbench_avg=...`
- `LawBench` benchmark 本身只用于评测, 不直接回流进训练集

本 requirement 从现在开始固定区分两层:

- `paper target line`: `LawLLM-7B` 这条公开资产最完整的 SFT 路线
- `current cookbook execution line`: `Qwen/Qwen3-4B-Instruct-2507 + DISC-Law-SFT + SFT + LawBench official eval`
- `public upper-bound reference`: `Qzhou-Law` 这类更强但当前公开复现资产不完整的方法线

也就是说, 当前主问题变成:

> 在 MinT 上, 以 `DISC-LawLLM / LawLLM-7B` 作为论文目标线, 同时先用当前可跑通的 `Qwen/Qwen3-4B-Instruct-2507 + DISC-Law-SFT -> SFT -> LawBench official eval` cookbook 执行线把本地流水线、数据物化、训练、checkpoint、和官方评测先稳定下来。

---

## 2. 论文目标线与当前执行线

### 2.1 当前主线分层

当前 requirement 明确采用如下三层结构:

- `paper target line`: `LawLLM-7B`
- `current cookbook execution line`: `Qwen/Qwen3-4B-Instruct-2507 + DISC-Law-SFT + SFT`
- `public comparison baselines`: `Qwen/Qwen2.5-7B-Instruct` 与 `Qwen/Qwen2.5-32B-Instruct` on `LawBench zero-shot`
- `public upper-bound reference`: `Qzhou-Law`

这意味着:

- 论文目标仍然是 `Qwen/Qwen2.5-7B-Instruct + DISC-Law-SFT + SFT`
- 但当前仓库里默认维护和优先验证的 runnable line 是 `Qwen/Qwen3-4B-Instruct-2507 + DISC-Law-SFT + SFT + LawBench eval`
- 对外公开比较优先锚定已公开的通用 `Qwen2.5-Instruct` 结果
- `Qzhou-Law` 继续保留为更强的公开参考上限, 但不再作为首个必须落地的复现对象

### 2.2 为什么把主线切到 LawLLM-7B

截至 `2026-04-23`, 这条线的优势是:

1. 公开资产完整度更高
   - `DISC-LawLLM` 官方 repo 提供了 `DISC-Law-SFT` 数据入口, `LawLLM-7B` 模型, 以及 `lawllm_full_sft.yaml` / `lawllm_lora_sft.yaml` 两套 SFT 配置
2. 训练方式清晰
   - 它不是模糊的“post-training”, 而是明确的 `stage: sft`
3. 当前 7B 版本已经转到 Qwen 系
   - `LawLLM-7B` model card 明确写的是基于 `Qwen2.5-7B-Instruct`
4. 更适合当前仓库的单模型 benchmark 复现流程
   - phase 1 可以先只复现纯模型 SFT, 不把检索增强混进 canonical benchmark 路径

### 2.3 为什么不再把 Qzhou-Law 作为首个必须落地的 target

`Qzhou-Law` 仍然重要, 但它现在更适合作为参考上限, 不适合作为首个必须落地的主线, 原因是:

- 公开 benchmark 结果很强
- 但当前可直接验证的 repo / 训练配置 / 训练数据公开度, 不如 `DISC-LawLLM` 这条线完整
- 对本仓库而言, 第一条主线更需要“可审计可执行”, 而不是只盯最强分数

因此本 requirement 的当前判断是:

- 第一条主复现线: `LawLLM-7B`
- 更强公开上限参考: `Qzhou-Law`

---

## 3. benchmark 口径与公开比较线

### 3.1 LawBench benchmark 口径

根据 LawBench 官方仓库与正式论文, LawBench 是一个面向中国法律体系的大模型评测基准, 覆盖 20 个任务和 3 个认知层级:

- 法律知识记忆
- 法律知识理解
- 法律知识应用

本 requirement 的 canonical benchmark 设置固定为:

- `LawBench full benchmark`
- `zero-shot`
- 使用官方 benchmark 范围, 不自行裁剪任务子集

### 3.2 当前公开比较线

截至 `2026-04-23`, 当前 requirement 保留以下公开比较线:

| 来源 | 公开时间 | 设置 | 模型 | LawBench 分数 |
| --- | --- | --- | --- | --- |
| LawBench 官方仓库 | 2024-11 论文 / 仓库持续更新 | zero-shot | `Qwen-7B-Chat` | `37.00` |
| Unilaw-R1 | 2025-11 | zero-shot | `Qwen/Qwen2.5-7B-Instruct` | `52.3` |
| Unilaw-R1 | 2025-11 | zero-shot | `Qwen/Qwen2.5-32B-Instruct` | `63.8` |
| Qzhou-Law | 2026-01 公开稿 | benchmark-wide AVG | `Qzhou-Law-7B` | `70.56` |
| Qzhou-Law | 2026-01 公开稿 | benchmark-wide AVG | `Qzhou-Law-14B` | `72.31` |
| Qzhou-Law | 2026-01 公开稿 | benchmark-wide AVG | `Qzhou-Law-32B` | `73.75` |
| Qzhou-Law | 2026-01 公开稿 | benchmark-wide AVG | `Qzhou-Law-72B` | `75.79` |

### 3.3 对当前 requirement 最重要的结论

1. 对外公开比较, 当前最稳的通用 Qwen 锚点仍然是:
   - `Qwen/Qwen2.5-7B-Instruct = 52.3`
   - `Qwen/Qwen2.5-32B-Instruct = 63.8`
2. 对当前仓库的论文目标, 最合适的公开路线是 `LawLLM-7B`; 当前 `Qwen3` 执行线只是 mint-cookbook 里的低成本 runnable approximation, 不是新的论文目标
3. `Qzhou-Law` 仍然是值得记录的强参考上限, 但当前 requirement 不再要求第一阶段直接复现它
4. 截至本次补充调研, 我没有在 `DISC-LawLLM` 的当前公开主源里找到一个可直接核验的“新版 `LawLLM-7B` 在 LawBench 上的确切数字”, 因此 requirement 不直接把某个未经核验的 `LawLLM-7B` LawBench 分数写死
5. 因此当前 requirement 的关键交付, 是先把 `base eval -> SFT -> official LawBench eval` 的对照结果自己跑出来

---

## 4. LawLLM-7B 路线的详细研究结论

### 4.1 base model

当前公开的新版 `LawLLM-7B` 使用:

- `Qwen/Qwen2.5-7B-Instruct`

这点来自两处主源:

- `DISC-LawLLM` 官方 repo 在 `2025-05-20` 更新中说明, 因新版 `transformers` 不再方便支持 `Baichuan`, 因此重新构建了基于 `Qwen2.5-Instruct 7B` 的 `LawLLM-7B`
- `LawLLM-7B` 的 Hugging Face model card 明确标记该模型基于 `Qwen2.5-7B-Instruct`

### 4.2 训练方式

当前这条线的核心训练方式是:

- `SFT`

而且不是推断, 是配置中直接声明的:

- `lawllm_full_sft.yaml`: `stage: sft`
- `lawllm_lora_sft.yaml`: `stage: sft`

因此对本 requirement 来说:

- phase 1 不默认引入 `DPO`
- phase 1 不默认引入 `GRPO`
- phase 1 先按公开 SFT recipe 做对齐

### 4.3 训练数据

当前主训练数据是:

- `DISC-Law-SFT`

官方说明把它拆成两块:

- `DISC-Law-SFT-Pair`
- `DISC-Law-SFT-Triplet`

当前 requirement 对这两块的理解固定为:

- `Pair`: 更偏法律推理与任务 supervision
- `Triplet`: 更偏增强模型利用外部知识的能力

### 4.4 数据任务构成

`DISC-Law-SFT` 覆盖的任务包括:

- 法律信息抽取
- 司法事件检测
- 案件分类
- 判决预测
- 类案匹配
- 司法摘要
- 舆情摘要
- 法律问答
- 司法阅读理解
- 法律考试

此外, `Triplet` 部分还混入了通用指令数据:

- `Alpaca-GPT4`
- `Firefly`

这点很重要, 因为它说明这条线不是“只喂纯法律数据”, 而是保留了一定通用 instruction 数据来减少能力塌缩。

### 4.5 数据规模

当前公开主源里有两个需要同时记录的数字:

- repo 任务统计表按子任务相加约 `403K`
- repo 文字说明又说“目前开源近 30 万条训练数据”

因此 requirement 不把“403K 全量已公开”当成硬事实, 而是采用更保守的执行规则:

- 训练时必须记录实际使用的数据版本和样本数
- 如果当前公开下载到的是 `300K+` 子集, 那么 requirement 认可先基于已公开子集复现
- 不允许在文档里把未核验可下载的数据量写成已使用事实

### 4.6 官方公开训练配置

当前 requirement 记录两套官方配置, 但将它们区分为“主复现配置”和“低成本配置”。

#### 4.6.1 主复现配置

优先参考 `lawllm_full_sft.yaml`:

- `stage: sft`
- `finetuning_type: full`
- `cutoff_len: 4096`
- `learning_rate: 1e-5`
- `num_train_epochs: 2.0`
- `lr_scheduler_type: cosine`
- `per_device_train_batch_size: 2`
- `gradient_accumulation_steps: 4`

这说明当前公开的主线不是先 LoRA, 而是先 full SFT。

#### 4.6.2 低成本配置

作为更便宜的 profiling / ablation 线, 可以参考 `lawllm_lora_sft.yaml`:

- `stage: sft`
- `finetuning_type: lora`
- `lora_rank: 8`
- `lora_target: all`
- `cutoff_len: 4096`
- `learning_rate: 1e-4`
- `num_train_epochs: 3.0`
- `lr_scheduler_type: cosine`
- `per_device_train_batch_size: 1`
- `gradient_accumulation_steps: 2`

### 4.7 检索增强的定位

`DISC-LawLLM` 还包含知识库增强路线, 但本 requirement 当前明确把它放在 phase 2 以后, 不放入首个 canonical benchmark 复现路径, 原因是:

- 当前首要目标是复现“纯模型 SFT 后的 LawBench 变化”
- 检索增强会改变系统边界, 使 benchmark 口径更复杂
- 如果后续要做 knowledge-augmented 版本, 必须单独记录系统路径和额外依赖

因此当前 phase 1 的目标固定为:

- `Qwen/Qwen2.5-7B-Instruct`
- `DISC-Law-SFT`
- `SFT`
- `LawBench official eval`

而不是:

- `SFT + retrieval` 的混合系统 benchmark

---

## 5. 当前算法路线决策

### 5.1 phase 1 默认路线

当前 requirement 的默认推进路线固定为:

1. 以 `DISC-LawLLM / LawLLM-7B` 作为论文目标线, 明确目标 base model 是 `Qwen/Qwen2.5-7B-Instruct`
2. 在当前 mint-cookbook 代码里, 先用 `Qwen/Qwen3-4B-Instruct-2507` 跑通同结构的 `DISC-Law-SFT -> SFT -> LawBench official eval` 执行线
3. 记录当前 cookbook 执行线的 base vs post-SFT 对照
4. 在流水线稳定后, 再把同一套 train/eval 合同推进到 `Qwen/Qwen2.5-7B-Instruct / LawLLM-7B` 目标线上

### 5.2 为什么 phase 1 不优先做 DPO 或 GRPO

- `DISC-LawLLM` 当前公开主线本身就是 `SFT`
- 这条线已经足够回答“法律监督数据是否能让 Qwen2.5-7B-Instruct 在 LawBench 上提升”
- 在没有先跑出 SFT 对照之前, 直接引入 `DPO` 或 `GRPO` 只会扩大变量数
- `LawBench` 是混合任务 benchmark, benchmark-wide reward 并不天然统一, 因此 `GRPO` 不应成为当前主线的默认起点

### 5.3 full SFT 与 LoRA 的关系

当前 requirement 的默认判断是:

- 如果资源允许, 论文目标线优先对齐 `LawLLM-7B` 的 `full SFT` 主配置
- 如果当前 MinT 资源不足以直接承接该 7B full SFT, 可以先做一条 `Qwen3-4B + LoRA` cookbook 执行线作为 profiling / 低成本验证
- 但 requirement 中必须清楚区分:
  - `paper-target full SFT` 结果
  - `cookbook execution-line LoRA approximation` 结果

LoRA 不是 `DISC-LawLLM` 论文目标线的主定义, 只是当前 mint-cookbook 执行线的更便宜近似。

---

## 6. 数据要求

### 6.1 eval 数据要求

eval 数据必须满足:

- 来源对齐 LawBench 官方 benchmark
- 保留独立的 eval-side 下载脚本与 eval-side 物化脚本
- 默认 benchmark 输入必须保存在 `experiments/lawbench/data/` 下
- 默认 `--eval-only` 路径不得依赖运行时在线抓取 benchmark
- benchmark 范围, 任务列表, 处理方式必须记录在 `data/sources.yaml`

### 6.2 train 数据要求

phase 1 的 train 数据默认要求是:

- 优先使用当前公开可下载的 `DISC-Law-SFT` 子集
- 保留独立的 train-side 下载脚本与 train-side 物化脚本, 不要和 LawBench eval 的脚本混用
- 必须记录具体来源 URI, 下载日期, 行数, 字段映射, 清洗规则
- 如果 repo 表中的完整统计与实际可下载子集不一致, 必须记录“实际使用的是公开子集, 不是声明中的全量统计表”

当前 train 数据设计必须优先支持:

- 法律知识问答
- 法律条文 / 法规理解
- 法律场景任务
- 法律咨询 / 规范化作答
- 与 LawBench 20 个任务有可解释映射关系的 instruction 数据

并且必须满足:

- 与 LawBench eval benchmark 明确解耦
- 不在 train manifest 物化阶段做基于 eval 的删样过滤
- 如果需要检查 train/eval 风险, 以 `--dry-run` 报告或单独审计为主, 不把它直接写成 train 数据裁剪规则
- 不得把 LawBench 官方 benchmark rows 直接并入训练集

### 6.3 supplemental 数据

如果后续需要补充数据, 当前优先级是:

1. `DISC-Law-SFT` 当前公开子集
2. 其他公开法律 instruction 数据
3. 通用 instruction replay 数据

但任何 supplemental 数据都必须满足:

- 来源可追溯
- 与 benchmark 解耦
- 采样比例有记录
- 不把 benchmark test 改写后回流进训练集

### 6.4 `data/sources.yaml` 最低要求

`data/sources.yaml` 至少记录:

- `eval_sources`
- `train_sources`
- LawBench 官方来源与版本信息
- `DISC-Law-SFT` 具体来源与版本信息
- 实际下载到的 train 子集大小
- 训练数据来源与物化规则
- train / eval 解耦规则
- overlap audit 规则

---

## 7. `train.py` 要求

### 7.1 首期必须支持的模式

`train.py` 首期必须支持:

- `--dry-run`
- `--eval-only`
- 标准 `SFT` 训练模式
- 结构化指标输出
- experiment-local 数据路径
- experiment-local `.env` 读取

### 7.2 当前默认 base model 与训练主线

当前 requirement 对 `train.py` 的要求是:

- 论文目标 base model 是 `Qwen/Qwen2.5-7B-Instruct`
- 当前 checked-in cookbook 默认执行 base model 允许继续使用 `Qwen/Qwen3-4B-Instruct-2507`

并围绕下面这条训练线设计:

- cookbook 执行线: `eval-only -> Qwen3-4B SFT -> LawBench eval`
- 论文目标线: `eval-only -> LawLLM-7B-aligned SFT -> LawBench eval`

如果当前实验实现仍保留 `Qwen3` 默认值, 文档必须把它明确标成 `current cookbook execution line`, 不能把它写成论文目标本身。

### 7.3 首期建议支持的最小参数集

首期建议至少支持:

- `--train-data`
- `--eval-data`
- `--log-path`
- `--base-model`
- `--seed`
- 同目录自动 same-run resume（复用同一个 `--log-path`）
- `--load-checkpoint-path`

除此之外, 对当前 LawLLM-7B 路线最重要的训练参数是:

- `--learning-rate`
- `--batch-size`
- `--num-epochs`
- `--lr-schedule`
- `--cutoff-len` 或等价长度控制参数
- `--finetuning-type` (`full` / `lora`), 如果当前实现需要同时承载两条线

### 7.4 phase 1 默认训练超参参考

如果当前实现需要一个默认参考值, 当前 requirement 推荐优先靠拢官方 full SFT 配置:

- `learning_rate = 1e-5`
- `num_epochs = 2`
- `lr_schedule = cosine`
- `cutoff_len = 4096`

如果当前 Mint 资源不允许直接做 full SFT, 则允许先用 LoRA 近似线:

- `finetuning_type = lora`
- `lora_rank = 8`
- `learning_rate = 1e-4`
- `num_epochs = 3`
- `cutoff_len = 4096`

但文档和结果必须明确标注这是近似线, 不是官方 full SFT 主线。

---

## 8. 验收口径

当前 requirement 分两层验收:

1. 当前 cookbook 执行线必须能稳定跑通:
   - `Qwen/Qwen3-4B-Instruct-2507` 的 `LawBench` baseline
   - 基于 `DISC-Law-SFT` 的 `SFT`
   - post-SFT 的官方 `LawBench` eval
2. 论文目标线继续保持为:
   - `Qwen/Qwen2.5-7B-Instruct / LawLLM-7B`
   - `DISC-Law-SFT`
   - 官方 `LawBench` eval

对当前代码库, phase 1 的最小验收标准是:

1. `Qwen/Qwen3-4B-Instruct-2507` 的 `LawBench` baseline 跑通
2. 基于当前公开 `DISC-Law-SFT` 子集的 `SFT` 训练跑通
3. post-SFT 模型再次跑通官方 `LawBench full benchmark`
4. README 或 run artifact 中明确记录:
   - 当前执行 base model
   - target paper line
   - train dataset version and rows
   - train config
   - eval config
   - final metrics
   - wall-clock timing
5. 无论结果是否提升, 都必须记录 base vs post-SFT 对照

当前 phase 1 不把下面这些内容列为必须完成:

- retrieval-enhanced system benchmark
- DPO
- GRPO
- 直接追平 `Qzhou-Law`

---

## 9. 主源与检索日期

以下主源用于本 requirement 的当前结论, 检索日期统一记为 `2026-04-23`:

- LawBench 官方 repo: `https://github.com/open-compass/LawBench`
- LawBench 正式论文: `https://aclanthology.org/2024.emnlp-main.452/`
- `DISC-LawLLM` 官方 repo: `https://github.com/FudanDISC/DISC-LawLLM`
- `LawLLM-7B` model card: `https://huggingface.co/ShengbinYue/LawLLM-7B`
- `DISC-Law-SFT` dataset card: `https://huggingface.co/datasets/ShengbinYue/DISC-Law-SFT`
- `Qzhou-Law` 公开稿: `https://openreview.net/forum?id=TRuj9IpWOL`
- `Unilaw-R1` 公开结果来源

如果后续发现 `LawLLM-7B` 的官方 LawBench 精确分数, 或者 `DISC-Law-SFT` 的公开子集规模发生变化, 应直接更新本 requirement, 而不是继续沿用旧推断。
