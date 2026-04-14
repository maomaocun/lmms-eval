# lmms-eval Qwen3-VL 启动脚本

本目录包含了一套用于 **本地单机 / DLC 集群** 运行 `lmms-eval` + `vLLM` 后端的脚本。

---

## 文件结构

```
scripts/
├── eval_common.sh          # 共享 bash 函数库（配置解析、启动 vLLM、跑 eval、清理）
├── qwen3_vl_worker.sh      # Worker 入口：只负责启动 vLLM 后端并执行 lmms-eval
├── qwen3_vl_submit.sh      # Submitter 入口：读取 DLC 配置 + eval 配置，并向 DLC 集群提交任务
└── README.md               # 本文档
```

项目 `scripts/` 目录下配套的配置示例：

```
scripts/
├── config_eval.json        # evaluation 相关配置（env / log / distributed / model / eval）
└── config_dlc.json         # DLC 集群提交相关配置（dlc 字段）
```

---

## 核心设计思想

### 1. 职责分离

原来的 `vllm_qwen3_vl_async_multi_gpu_yyc.sh` 把 **"DLC 提交"** 和 **"worker 执行"** 混在一起，通过 `dlc.submit=true/false` 在同一份脚本里做模式切换，导致：

- 控制流晦涩难懂（递归自调用）
- DLC `--command` 是一团嵌套转义的长字符串，极易写错引号
- 本地调试和集群运行行为不一致

现在的设计把两者彻底拆成两个独立脚本，并且把配置也拆成两份：

- **`qwen3_vl_submit.sh <dlc_config> <eval_config>`**：仅在本地运行，做一件事——生成 runtime config 并 `dlc submit`。
- **`qwen3_vl_worker.sh <eval_config> [model_path]`**：在每台机器（本地机器或 DLC worker 容器）上运行，做一件事——启动 vLLM 后端、等待就绪、运行 lmms-eval、退出清理。

### 2. 共享库 `eval_common.sh`

所有可复用的逻辑（解析 JSON、启动 backend、health check、cleanup trap、跑 eval）都抽取到 `eval_common.sh`。未来如果有其他模型需要类似流程，直接 `source` 即可，不需要复制代码。

---

## 配置拆分说明

### `config_eval.json`（evaluation 配置）

包含本地调试和 DLC worker 运行所需的所有参数：

- `env`：HF_HOME、HF_TOKEN、venv_path、数据集缓存路径、offline 开关等
- `log`：日志根目录
- `distributed`：master_addr、master_port、world_size、rank
- `model`：模型路径、TP 大小、max_model_len、gpu_memory_utilization 等
- `eval`：任务列表、output_path、并发数、gen_kwargs、limit、debug 等

### `config_dlc.json`（DLC 集群配置）

仅包含集群调度参数：

- `dlc.binary`：dlc CLI 路径
- `dlc.job_name`：作业名称
- `dlc.workers` / `dlc.worker_gpu` / `dlc.worker_cpu` / `dlc.worker_memory` 等
- `dlc.worker_image`、`dlc.data_source_uris`、`dlc.resource_id` 等 VPC/安全组参数

**好处**：
- 本地调试时完全不需要关心 DLC 参数；
- 切换集群资源（如换 quota、换镜像）时，只需要改 `config_dlc.json`，不动 `config_eval.json`；
- 同一份 `config_eval.json` 可以搭配不同的 `config_dlc.json`（开发环境 / 生产环境 / 不同 region）。

---

## 使用方式

### 本地单机调试

```bash
cd /mnt/cpfs/yangyicun/lmms-eval
bash scripts/qwen3_vl_worker.sh scripts/config_eval.json
```

也可以覆盖模型路径：

```bash
bash scripts/qwen3_vl_worker.sh scripts/config_eval.json /path/to/another/model
```

**注意**：本地运行时**不会**自动 staging 数据集缓存（避免不必要的 124G 拷贝）。如果你确实想在本地也做 staging，可以手动设置环境变量：

```bash
LMMS_EVAL_STAGE_DATASETS=1 bash scripts/qwen3_vl_worker.sh scripts/config_eval.json
```

### 提交到 DLC 集群

```bash
cd /mnt/cpfs/yangyicun/lmms-eval
bash scripts/qwen3_vl_submit.sh scripts/config_dlc.json scripts/config_eval.json
```

Submitter 会：

1. 读取 `config_dlc.json` 中的集群参数；
2. 读取 `config_eval.json` 中的模型、任务、环境等参数；
3. 生成 `runtime_config.json`（强制 `dlc.submit=false`、`eval.debug=false`）；
4. 调用 `dlc submit pytorchjob`，把 `qwen3_vl_worker.sh` 作为 worker 的入口命令，同时开启 `LMMS_EVAL_STAGE_DATASETS=1`。

### 向后兼容的包装器（可选）

如果你之前一直用旧脚本，可以继续保持习惯：

```bash
# 本地调试
bash vllm_qwen3_vl_async_multi_gpu_yyc.sh scripts/config_eval.json

# 提交到集群（推荐新用法）
bash vllm_qwen3_vl_async_multi_gpu_yyc.sh scripts/config_dlc.json scripts/config_eval.json
```

这个包装器会自动判断参数：

- 传入 **1 个 config** → 转发给 `qwen3_vl_worker.sh`（本地运行）
- 传入 **2 个 config（且第二个以 `.json` 结尾）** → 转发给 `qwen3_vl_submit.sh`（提交到集群）
- 传入 **1 个旧版 config（内部含 `dlc.submit=true`）** → 降级为单配置提交模式，并打印 deprecation 警告

---

## 数据流与执行时序

### 本地 / Worker 执行时序 (`qwen3_vl_worker.sh`)

```
source eval_common.sh
      │
      ▼
load_config()        ← 读取 eval config、导出 HF_HOME/HF_TOKEN/offline flags
      │
      ▼
compute_resources()  ← 探测本地 GPU 数、计算 MACHINE_RANK、NUM_BACKENDS
      │
      ▼
setup_logging()      ← 确定 LOG_DIR（如果 LMMS_EVAL_LOG_DIR 已设置则用它）
      │
      ▼
ensure_venv()        ← 激活 .venv
      │
      ▼
setup_cleanup_trap() ← 注册 EXIT/INT/TERM 时的 vLLM 清理函数
      │
      ▼
launch_vllm_backends() ──► 启动 NUM_BACKENDS 个 vLLM 进程
      │
      ├────────────────────────────────────┐
      ▼                                    ▼
wait_for_backends()                stage_datasets() (仅在 LMMS_EVAL_STAGE_DATASETS=1 时后台 cp -r)
轮询 backend health check          并行拷贝数据集缓存
      │                                    │
      ▼                                    ▼
(wait 数据集拷贝完成) ◄──────────────────┘
      │
      ▼
run_lmms_eval()      ← accelerate launch -m lmms_eval ...
      │
      ▼
cleanup_vllm()       ← 杀死所有 backend 进程（debug=true 时跳过）
```

### DLC 提交时序 (`qwen3_vl_submit.sh`)

```
读取 dlc_config.json + eval_config.json
      │
      ▼
生成 runtime_config.json
   (基于 eval_config，强制 dlc.submit=false, eval.debug=false)
      │
      ▼
构建 DLC command:
   export LMMS_EVAL_LOG_DIR=<FIXED_LOG_DIR>;
   export LMMS_EVAL_STAGE_DATASETS=1;
   bash qwen3_vl_worker.sh <runtime_config.json>
      │
      ▼
dlc submit pytorchjob --command="..."
```

DLC 调度器在每台 worker 上执行这个 command，于是 worker 容器里就走上一节的 **Worker 执行时序**。

---

## 注意事项

1. **不要直接执行 `eval_common.sh`**  
   它是被 `source` 的函数库，直接运行会报错并退出。

2. **Worker 脚本里不再包含任何 DLC 逻辑**  
   如果你需要改集群提交的参数（如 worker 数、镜像、资源 ID），只改 `config_dlc.json` 或 `qwen3_vl_submit.sh` 即可，不会影响本地调试逻辑。

3. **日志目录**  
   - 本地跑：默认在 `/mnt/cpfs/yangyicun/vllm_logs/2026-04-12_19-44-46/`（时间戳目录）
   - DLC 提交：强制统一到 `/mnt/cpfs/yangyicun/vllm_logs/<JOB_NAME>/`（与 runtime config 同目录）

4. **数据集缓存**  
   `stage_datasets()` 仅在 `LMMS_EVAL_STAGE_DATASETS=1` 时执行。`qwen3_vl_submit.sh` 会自动为 DLC worker 设置该变量，而本地运行默认不设置，避免误拷贝 124G 数据。

5. **向后兼容**  
   旧版单配置文件（同时包含 `dlc` 和 `eval`）仍可通过包装器运行，但会打印 deprecation 警告。建议尽快拆分为 `config_dlc.json` + `config_eval.json`。
