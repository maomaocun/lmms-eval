# TritonBench (lmms-eval port)

Adaptation of [thunlp/TritonBench](https://github.com/thunlp/TritonBench) —
*Benchmarking Large Language Model Capabilities for Generating Triton Operators*
([arXiv:2502.14752](https://arxiv.org/abs/2502.14752)) — to the `lmms-eval`
framework.

## Tasks

| Task name              | Track          | Instruction style |
| ---------------------- | -------------- | ----------------- |
| `tritonbench_g`        | TritonBench-G  | Simplified        |
| `tritonbench_g_comp`   | TritonBench-G  | Complex (full)    |
| `tritonbench_t`        | TritonBench-T  | Simplified        |
| `tritonbench_t_comp`   | TritonBench-T  | Complex (full)    |

TritonBench-G is the 184-problem set crawled from real GitHub repositories.
TritonBench-T is sourced from PyTorch operator interfaces.

## Metrics (both `higher_is_better`)

* **`call_acc`** — fraction of generated kernels that compile and execute
  without error when grafted onto the upstream gold test harness. Mirrors
  `EVAL/eval_*/0_call_acc.py` upstream.
* **`exec_acc`** — fraction whose stdout matches the gold kernel's stdout
  byte-for-byte. Mirrors `EVAL/eval_*/1_exe_acc.py` upstream. Implies `call_acc`.

The `speedup` metric from the upstream paper is **not** reported by this port;
the spec for this adaptation only requires the two correctness metrics.

## Data flow

1. Per-record metadata (instructions, gold output, file name, repo) loads
   directly from the upstream raw URLs declared in each YAML.
2. The gold test harness (a separate `.py` file per problem) is fetched lazily
   on first use by `data.py` and cached on disk. The first call downloads the
   upstream tarball once and extracts every harness for both tracks; subsequent
   calls hit the cache.
3. `executor.py` extracts code from the model's response, grafts it in front of
   the gold harness, runs the resulting script in a subprocess, and compares
   stdout to the gold's.

## Cache & runtime knobs

| Env var                              | Default                                                  | Effect |
| ------------------------------------ | -------------------------------------------------------- | ------ |
| `LMMS_TRITONBENCH_CACHE`             | `~/.cache/lmms_eval/tritonbench`                         | Where to cache upstream data files. |
| `LMMS_TRITONBENCH_TIMEOUT`           | `120` (seconds)                                          | Per-script subprocess timeout. |
| `LMMS_TRITONBENCH_DRY_RUN`           | unset                                                    | Truthy → skip subprocess execution, report `0` for both metrics. Pipeline smoke-test mode. |
| `LMMS_TRITONBENCH_SANDBOX`           | `none`                                                   | `none` = bare subprocess (Colab default). `docker` = run each script in a one-shot container with `--network none`, read-only fs, memory/cpu caps. |
| `LMMS_TRITONBENCH_DOCKER_IMAGE`      | `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`          | Image used when `SANDBOX=docker`. Must have `python`, `torch`, `triton`. |
| `LMMS_TRITONBENCH_DOCKER_MEM`        | `8g`                                                     | Container memory cap. |
| `LMMS_TRITONBENCH_DOCKER_CPUS`       | `2`                                                      | Container cpu cap. |
| `LMMS_TRITONBENCH_DOCKER_GPUS`       | `all`                                                    | `--gpus` value, or `none` to disable. |
| `LMMS_TRITONBENCH_DOCKER_EXTRA_ARGS` | unset                                                    | Extra `docker run` args (whitespace-split). |

## Execution requirements

`process_results` runs model-generated Python. The script imports `triton` +
`torch` and allocates CUDA tensors, so the runtime needs a CUDA GPU and those
packages installed.

**Do not run this task on a machine you don't fully control** without
`SANDBOX=docker` — model output is executed as Python code.

* On **Colab** (the project's intended runtime): leave `SANDBOX=none`. Colab is
  itself an ephemeral VM, so the bare-subprocess mode is acceptable.
* On a **workstation or shared host**: set `LMMS_TRITONBENCH_SANDBOX=docker`.
  Each generated script runs in a one-shot container with no network, read-only
  rootfs (writable `/tmp` tmpfs for triton's autotune cache), memory/cpu caps,
  and the configured GPU set forwarded.
