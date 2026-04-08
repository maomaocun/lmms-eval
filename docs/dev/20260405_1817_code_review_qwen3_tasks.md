# Code Review 报告：Qwen3-VL Task 适配

**审查时间**：2026-04-05 18:17  
**审查范围**：下午新增/修改的 Qwen3-VL 官方评估 Task 适配代码及相关支撑改动  
**作者**：Kimi Code CLI

---

## 一、审查范围

本次 review 覆盖以下新增/修改内容：

| 模块/文件 | 说明 |
|-----------|------|
| `lmms_eval/tasks/mathvision/utils_qwen3.py` + YAML | MathVision Qwen3 适配（标准版 + reasoning 版） |
| `lmms_eval/tasks/mmmu/utils_qwen3_official.py` + YAML | MMMU 官方评估逻辑复现 |
| `lmms_eval/tasks/mmmu_pro/utils_qwen3_official.py` + YAML | MMMU-Pro 官方评估逻辑复现 |
| `lmms_eval/tasks/wemath/utils_qwen3.py` + YAML | WeMath Qwen3 适配 |
| `lmms_eval/models/chat/vllm_backend.py` | 新增 vLLM Backend 模型 |
| `lmms_eval/models/chat/openai.py` | 同步 OpenAI 包装器增强 |
| `lmms_eval/models/simple/openai.py` | Simple OpenAI 包装器增强 |
| `lmms_eval/protocol.py` | Video 处理健壮性增强 |
| `lmms_eval/llm_judge/standalone.py` / `aggregator.py` | Judge pipeline 兼容改造 |
| `.gitignore` / `__main__.py` / CLI 日志 | 日志颜色控制等配套改动 |

---

## 二、总体评价

**优点**：
1. **官方逻辑复现准确**：MMMU/MMMU-Pro 的 rule-based 提取 + GPT Judge fallback 两阶段流程与 Qwen3-VL 官方 repo 保持一致。
2. **生成与判卷解耦设计合理**：`process_results` 在 generation 阶段仅保存原始输出，不依赖 API Key；真正的 judge 在独立的 `judge` 阶段完成，便于复跑和换模型。
3. **Judge pipeline 集成到位**：`standalone.py` 中增加的 `__sample_context__` 透传、`aggregator.py` 中新增的 `mmmu_val_qwen3_official` 配置，保证了新 task 能无缝接入现有的 LLM Judge 框架。
4. **模型侧增强实用**：`vllm_backend.py` 原生 HTTP 调用支持 `top_k`、`repetition_penalty` 等 vLLM 专属参数；同步/异步 OpenAI 包装器均增加了多 backend 负载均衡、并发控制、采样参数透传等能力。
5. **日志体验优化**：`NO_COLOR` / `LOGURU_NO_COLOR` 感知避免了重定向到文件时出现 ANSI 转义字符。

---

## 三、发现的问题及已执行的修复

### 1. 遗留备份文件 ❌ → ✅ 已修复
- **问题**：`lmms_eval/tasks/mathvision/utils_qwen3.py.bak` 是备份文件，不应提交到仓库。
- **修复**：已删除该文件。

### 2. `.gitignore` 路径不匹配 ❌ → ✅ 已修复
- **问题**：`.gitignore` 中写的是 `.judged_results/`（带点），但 `git status` 中实际未跟踪的目录是 `judged_results/`（不带点），导致无法生效。
- **修复**：已将 `.judged_results/` 改为 `judged_results/`。

### 3. `vllm_backend.py` 模块级副作用 ❌ → ✅ 已修复
- **问题**：模块导入时就全局修改环境变量和日志器：
  ```python
  os.environ['LOGURU_NO_COLOR'] = '1'
  eval_logger.remove()
  eval_logger.add(sys.stderr, colorize=False)
  tqdm.disable_color = True
  ```
  这会影响同一进程内其他模型/模块的日志和进度条颜色设置。
- **修复**：将颜色禁用逻辑移至 `VLLMBackend.__init__` 内部，使用 `os.environ.setdefault` 避免覆盖已有配置，且仅在实例化该模型时生效。

### 4. `can_infer_option` 中继承自官方的 Bug ❌ → ✅ 已修复
- **问题**（`mmmu` 与 `mmmu_pro` 两个文件均存在）：
  原代码中 `"A"` 的冠词保护逻辑写在了 `for ch in choices:` 循环的最前面：
  ```python
  if 'A' in splits and len(splits) > 3:
      return False
  if ch in splits:
      return ch
  ```
  这意味着只要回答里出现了 "A" 且词数 >3，**即使正确答案明明是 B/C/D**，也会被直接返回 `False`，永远不会匹配到后面的选项。
- **修复**：将保护逻辑后移至 `if ch in splits:` 内部，仅当 `ch == 'A'` 时才生效：
  ```python
  if ch in splits:
      if ch == 'A' and len(splits) > 3:
          return False
      return ch
  ```

### 5. 同步 OpenAI 包装器缺少 `top_k` 透传 ❌ → ✅ 已修复
- **问题**：`chat/openai.py` 和 `simple/openai.py` 从 `gen_kwargs` 里取了 `top_k`，但没有写入 `payload`，而 `async_openai.py` 已通过 `extra_body["top_k"]` 正确透传。
- **修复**：在两个同步包装器中都增加了：
  ```python
  if top_k is not None:
      payload.setdefault("extra_body", {})["top_k"] = top_k
  ```
  同时更新了注释，避免与 vLLM 支持能力相矛盾。

### 6. 模块级 YAML 加载风险 ❌ → ✅ 已修复
- **问题**：
  - `mathvision/utils_qwen3.py` 在模块导入时读取 YAML，但读出来的 `config` 变量**从未被使用**。
  - `wemath/utils_qwen3.py` 同样在导入时读 YAML，仅作为 `doc_to_messages` 函数内的一个 fallback。
  这种写法会导致：YAML 文件一旦移动或格式错误，整个模块 import 直接失败。
- **修复**：
  - `mathvision/utils_qwen3.py`：彻底移除 YAML 读取代码块及未使用的 `yaml` import。
  - `wemath/utils_qwen3.py`：移除模块级 YAML 读取，函数内的 fallback 改为 `lmms_eval_specific_kwargs = {}`，因为后续所有参数都有 `.get(..., default)` 兜底，效果完全一致。

---

## 四、无需修改的加分项

以下改动本身已经合理，review 中未做修改：

1. **`protocol.py` 的 video 处理增强**
   - 增加了对 `fetch_video` 返回 `tuple` 的兼容、更详细的异常信息，健壮性更好。
2. **Judge pipeline 的嵌套字段提取**
   - `standalone.py` 中新增对 `hit`/`score`/`correct`/`accuracy` 等嵌套字典字段的提取，适配了 MMMU 官方 task 的聚合需求。
3. **日志颜色控制**
   - `__main__.py`、`judge_cmd.py`、`aggregate_cmd.py` 统一支持 `NO_COLOR` 环境变量，避免了日志文件被 ANSI 转义码污染。

---

## 五、后续建议

1. **测试覆盖**：建议为 `can_infer_option` / `can_infer_text` 补几个单测（尤其是包含 "A" 的多选项场景），防止后续重构再引入回归。
2. **文档同步**：`lmms_eval/tasks/mmmu/README_qwen3_official.md` 已写得非常详细，建议在 README 里加一句说明 `can_infer_option` 的 "A" 保护逻辑已经做了本地修复（相对于官方 repo）。
3. **vllm_backend 的异常处理**：目前主要依赖 `requests` 的原生异常，后续可考虑把常见的 vLLM 报错（如 `prompt too long`、`model not found`）做更友好的封装和日志输出。

---

## 六、结论

本次新增的 Qwen3-VL Task 适配代码**功能正确、设计合理**，在修复上述 6 项问题后，已达到合入标准。所有修复已就地应用，可直接进行后续测试与验证。
