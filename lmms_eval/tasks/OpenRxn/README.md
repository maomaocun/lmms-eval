# OpenRxn - lmms-eval Judge/Generate 分离适配说明

## 1. 架构变更

OpenRxn 已完成与 `lmms-eval` **Generation → Judging → Aggregation** 分离架构的适配。

- **旧行为**：`process_results` 内部直接调用硬编码的 vLLM/OpenAI 端点进行 LLM judge，generation 和 judging 强耦合。
- **新行为**：`process_results` 只做轻量结果包装；真正的化学领域 LLM judge 由 `lmms-eval judge` 独立流水线负责。

## 2. 核心文件说明

| 文件 | 职责 |
|------|------|
| `OpenRxn.yaml` | 任务配置，指定 `generate_until`、metric (`llm_judge_score`) 及 utils 函数注册 |
| `utils.py` | `doc_to_*` 转换、`process_results`、化学领域 judge prompt (`get_judge_prompt`)、聚合函数 |

## 3. 使用流程

### 步骤 1：纯生成（不再内嵌 LLM judge）

```bash
lmms-eval eval \
  --model <your_model> \
  --tasks OpenRxn \
  --log_samples \
  --output_path ./results/
```

此阶段仅运行模型生成，不会调用任何外部 judge API。

### 步骤 2：独立评判（使用 `lmms-eval judge`）

```bash
lmms-eval judge \
  -i results/*_samples_OpenRxn.jsonl \
  -t OpenRxn \
  \
  --judge-model gpt-4o-mini \
  --parallel 8 \
  -d judged_results/
```

- `auto` 模式下，`JudgeRunner` 会：
  1. 先调用 `process_results`（返回占位分 `llm_judge_score: 0` 和 `needs_llm_judge: True`）
  2. 检测到 `needs_llm_judge` 后触发 **LLM fallback**
  3. 自动调用 `get_judge_prompt()` 获取化学专业 prompt（支持 E-SMILES、数值容差等）
  4. 将 fallback 结果直接更新到 `llm_judge_score`

### 步骤 3：聚合

```bash
lmms-eval aggregate \
  -i judged_results/*_samples_OpenRxn.jsonl \
  -t OpenRxn
```

聚合为简单平均，最终输出 `llm_judge_score`。

## 4. 环境变量配置（标准方式）

不再需要任务内部硬编码的 `VLLM_BASE_URL`、`MODEL_NAME` 等变量。评判阶段的模型和端点完全由 `lmms-eval judge` 的标准环境变量控制：

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `JUDGE_MODEL` | Judge 模型名称 | `gpt-4o-mini` |
| `JUDGE_BASE_URL` | Judge API 端点 | `https://api.openai.com/v1` |
| `JUDGE_API_KEY` | API Key | 无 |
| `JUDGE_MAX_CONCURRENT` | 并发数 | `1` |
| `JUDGE_MODE` | 评判模式 (`rule`/`auto`/`llm`) | `auto` |

本地 vLLM/SGLang 示例：

```bash
export JUDGE_BASE_URL="http://localhost:8000/v1"
export JUDGE_MODEL="Qwen3-235B-A22B-Instruct-2507"
export JUDGE_API_KEY="dummy-key-for-local-vllm"
lmms-eval judge -i results.jsonl -t OpenRxn --parallel 8
```

## 5. `get_judge_prompt` 设计

`utils.py` 中提供了 `get_judge_prompt(doc, prediction, target)`，作为 **task-specific custom prompt hook** 被 `JudgeRunner` 动态加载。

该 prompt 包含化学领域专用的评估标准：

1. **化学公式 / E-SMILES**：结构相同即视为正确
2. **数值答案**：允许小数位差异
3. **文本答案**：语义一致即可
4. **Yes/No 问题**：答案方向一致即可

Judge 输出格式为 `correct` / `incorrect`，已由框架 `ResponseParser` 正确解析为 `1` / `0`。

## 6. 兼容历史 JSONL

如果已有旧版生成的 JSONL（其中 `process_results` 内嵌 judge 已把 `llm_judge_score` 写在 top-level），`lmms-eval judge` 会通过 `_extract_existing_metrics` 直接复用该值，不会重复调用 API。

## 7. 注意事项

- `process_results` 目前返回 `llm_judge_score: 0` 和 `needs_llm_judge: True`，这意味着 **纯 `rule` 模式不可用**（会全部判为 0）。请始终使用 `auto` 或 `llm` 模式运行 `lmms-eval judge`。
- 若 doc 在保存 JSONL 时被 tracker 丢弃，`JudgeRunner` 会通过 `__sample_context__` 重建最小 doc 以支持 `get_judge_prompt`。
