# Single-File Experiment Review Checklist

适用范围：

- `experiments/chat-dpo`
- `experiments/fingpt`
- `experiments/lawbench`
- 未来从 `scaffolds/single_file_experiment` 派生的新实验

目标：

- 保证单文件实验之间的主流程、artifact、resume、测试树保持同构
- 允许 benchmark-specific 逻辑不同，但不允许骨架层无意识漂移
- 为 code review、迁移审查、模板对齐检查提供一份统一清单

## 一、使用方式

这份清单主要用于三类场景：

1. 新实验从 scaffold 落地后，做第一次结构审查
2. 对已有实验做功能改动后，检查是否引入骨架漂移
3. 对照 `scaffolds/` 与具体实验，判断哪些差异是 intentional，哪些是 drift

建议把检查结果分成三类：

- `骨架一致`：与现有单文件实验范式一致
- `可接受漂移`：存在差异，但属于任务特定逻辑
- `建议尽快对齐`：差异会增加维护成本或误导后续实验

## 二、骨架层检查项

### 1. CLI 与主入口

检查 `parse_args()` 与 `main_async()`：

- 是否使用 `--log-path` 作为运行目录参数
- 是否同时支持以下主控制参数：
  - `--dry-run`
  - `--eval-only`
  - `--train-data`
  - `--eval-data`
  - `--load-checkpoint-path`
- `main_async()` 是否仍保持统一顺序：
  1. parse args
  2. load data
  3. overlap audit
  4. dry-run early return
  5. prepare run dir / console log
  6. create service client
  7. eval-only branch
  8. train branch
  9. write outputs / run metadata
- eval-only 是否不会错误进入 training-side 恢复逻辑

### 2. Run 目录与 artifacts

检查 `prepare_run_dir()`、`write_run_metadata()`、`write_outputs()`：

- `prepare_run_dir(log_path)` 是否只负责：
  - 创建目录
  - smoke run 时维护 `latest` 软链
- `write_run_metadata()` 是否仍写入这些基础字段：
  - `experiment`
  - `status`
  - `command`
  - `started_at_unix`
  - `ended_at_unix`
  - `duration_seconds`
  - `args`
  - `env`
  - `error`
- `write_outputs()` 是否仍维护统一 artifact 结构：
  - `eval/examples.jsonl`
  - `eval/predictions.jsonl`
  - `eval/metrics.json`
  - `train/metrics.jsonl`
  - `train/checkpoints.jsonl`
  - `train/batches.jsonl`
- eval-only 写结果时，是否显式避免混入旧 train artifacts
  - 即支持并正确使用 `include_existing_artifacts=False`

### 3. Resume 语义

这是最关键的一组，也是最容易漂移的一组。

检查以下 helper 与控制流：

- `get_last_resumable_checkpoint(log_path)`
  - 是否定义为：扫描 `train/checkpoints.jsonl`，取最后一个带非空 `state_path` 的 row
- `validate_resume_contract(...)`
  - 是否保证同一 run 继续训练时，关键参数不能变化
- `resolve_resume_state(...)`
  - 是否返回：
    - `start_step`
    - `start_epoch_idx`
    - `start_batch_idx`
  - 是否校验 checkpoint row 中 `step/epoch/batch` 的一致性
- `create_training_client(..., resume_state_path=...)`
  - same-run resume 是否恢复 optimizer state
  - `--load-checkpoint-path` 是否只做 weight-only 热启动
- `console.log`
  - fresh run 是否使用 `"w"`
  - resumed run 是否使用 `"a"`
- eval-only rerun 是否不会自动吸收旧 run 的 training `state_path`

### 4. Append-only 日志流

检查 fresh run 与 resumed run 对以下文件的处理：

- `eval/periodic.jsonl`
- `train/checkpoints.jsonl`
- `train/metrics.jsonl`
- `train/batches.jsonl`

要求：

- fresh run 会清空 append-only 文件
- resumed run 会保留 append-only 文件继续追加
- `train/batches.jsonl` 仍作为 per-step lineage / trace 文件存在

### 5. 训练返回结果字段

这块最容易影响跨实验统一汇总。

优先检查：

- 训练返回 dict 是否统一使用：
  - `train_steps`
  - `train_duration_seconds`
  - `train_total_steps`
- 如果实验使用不同字段名，是否有非常明确的理由
- checkpoint / final eval / stdout metric 输出读取的字段是否与返回值一致

### 6. Checkpoint 与 final artifact

检查 periodic checkpoint row 与 final checkpoint row：

- periodic checkpoint row 是否仍记录：
  - `name`
  - `step`
  - `epoch`
  - `batch`
  - `state_path`（若可用）
  - `sampler_path`（若可用）
- final checkpoint row 是否写回：
  - `final: true`
- final eval 是否基于最终 sampler 或保存的 sampler path
- `run.json` 是否仍能关联到最终 `state_path` / `sampler_path`（若该实验设计要求如此）

## 三、测试树检查项

每个真实实验至少应有三条 live smoke：

- `test_eval_only_live_smoke`
- `test_train_live_smoke`
- `test_resume_live_smoke`

review 时逐条确认：

### 1. eval-only smoke

- 能独立运行
- 产出 `run.json`
- 产出 `eval/metrics.json`
- 不错误混入 train artifacts

### 2. train smoke

- 能训练至少 1 step
- 产出 train metrics
- 产出 checkpoints
- 产出 final eval

### 3. resume smoke

- 能中断在 checkpoint 之后
- 能重跑同一命令自动 resume
- stdout / artifact 能证明是续跑而不是重头跑

## 四、Scaffold 对齐检查项

检查具体实验与 scaffold 的关系：

- 是否仍符合 `scaffolds/single_file_experiment` 的最小骨架
- 如果实验已经超出模板能力，是否同步更新了 scaffold 文档说明
- `capability_matrix.md` 是否仍准确描述当前实验状态
- 不允许模板文档长期落后于真实实验行为

## 五、允许不同，不必强行统一

下面这些差异通常属于 benchmark-specific，不应被误判为骨架漂移：

- prompt 形式
- grader / scorer
- metrics 细节
- DPO / SFT / GRPO 的任务逻辑
- smoke 数据内容
- benchmark-specific artifact 字段

这些差异需要说明清楚，但不需要强行对齐。

## 六、触发同步审查的高风险改动

如果改到了以下任一模块，建议顺手检查另外两个实验：

1. `parse_args`
2. `prepare_run_dir`
3. `write_run_metadata`
4. `write_outputs`
5. `get_last_resumable_checkpoint`
6. `validate_resume_contract`
7. `resolve_resume_state`
8. `create_training_client`
9. `main_async`
10. `tests/test_train.py` 中的三条 live smoke

这些模块属于当前仓库里最核心的“骨架同步面”。

## 七、推荐的 review 结论模板

可以直接用下面的格式写 review 结论：

### 骨架一致

- CLI / 主入口
- artifact 结构
- resume 语义
- 三段 smoke 测试树

### 发现漂移

- 字段命名
- 文档状态
- checkpoint schema
- eval-only artifact 行为

### 结论

- `可接受漂移`
- `建议尽快对齐`
- `会导致维护成本上升，应修复`

## 八、当前仓库的已知优先项

基于当前 `chat-dpo` / `fingpt` / `lawbench` / scaffold 状态，优先建议关注：

1. scaffold 文档是否反映最新 resume 状态
2. train metrics 字段名是否保持统一
3. eval-only artifact 行为是否一致
4. 新实验是否沿用了三段 smoke 测试树

## 九、相关参考

- `scaffolds/README.md`
- `scaffolds/single_file_experiment/README.md.tpl`
- `scaffolds/single_file_experiment/capability_matrix.md`
- `scaffolds/profiles/sft.md`
- `skills/helper-contract-alignment/SKILL.md`
