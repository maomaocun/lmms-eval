# KernelBench (lmms-eval port)

Adaptation of [ScalingIntelligence/KernelBench](https://github.com/ScalingIntelligence/KernelBench) — *KernelBench: Can LLMs Write Efficient GPU Kernels?* ([paper](https://scalingintelligence.stanford.edu/pubs/kernelbench.pdf)) — to the `lmms-eval` framework.

## Tasks

| Task name              | Split   | Problems | Description                                |
| ---------------------- | ------- | -------- | ------------------------------------------ |
| `kernelbench_level1`   | level_1 | 100      | Single-kernel ops (matmul, conv, layernorm…) |
| `kernelbench_level2`   | level_2 | 100      | Simple fusion patterns (Conv+Bias+ReLU, …)   |
| `kernelbench_level3`   | level_3 | 50       | Full model architectures                     |
| `kernelbench_level4`   | level_4 | 20       | HuggingFace model optimization               |

Dataset: `ScalingIntelligence/KernelBench` (default config) on HuggingFace.

## Metrics (all `higher_is_better`)

| Metric        | Definition                                                   |
| ------------- | ------------------------------------------------------------ |
| `compiled`    | Generated `ModelNew` imports without error                   |
| `correctness` | All `n_correct_trials` outputs match the reference within tolerance |
| `fast_1`      | Correct AND faster than the PyTorch reference (paper's `fast_1`) |
| `fast_2`      | Correct AND ≥ 2× the PyTorch reference (paper's `fast_2`)    |

`speedup = ref_runtime / runtime`, computed via `torch.cuda.Event` timing.

## Prompt template

Mirrors upstream's `cuda` backend, `zero_shot` option (from `src/kernelbench/prompts/prompts.toml`):

```
You write custom CUDA operators to replace the pytorch operators in the given
architecture to get speedups.

You are given the following architecture:

<reference Model source>

Note: The kernels should be optimized for FP32 (32-bit floating point) precision.

Optimize the architecture named Model with custom CUDA operators! Name your
optimized output architecture ModelNew. Output the new code in codeblocks.
```

`one_shot` and `few_shot` options are not yet ported.

## Execution model

`process_results` extracts the model's code, drops it into a temp Python file
that calls upstream's `kernelbench.eval.eval_kernel_against_ref`, and runs it
in a subprocess. Each problem evaluates in isolation so a crash, hang, or GPU
OOM in one generation can't take down the whole eval.

The runtime must have `kernelbench` installed:

```bash
pip install "git+https://github.com/ScalingIntelligence/KernelBench.git"
```

## Runtime knobs

| Env var                              | Default                                          | Effect |
| ------------------------------------ | ------------------------------------------------ | ------ |
| `LMMS_KERNELBENCH_DRY_RUN`           | unset                                            | Truthy → skip subprocess, report `0` for every metric. Pipeline smoke-test mode. |
| `LMMS_KERNELBENCH_TIMEOUT`           | `300` (seconds)                                  | Per-problem subprocess wall-clock cap. |
| `LMMS_KERNELBENCH_NUM_CORRECT`       | `5`                                              | `num_correct_trials` passed to upstream. |
| `LMMS_KERNELBENCH_NUM_PERF`          | `100`                                            | `num_perf_trials` passed to upstream. |
| `LMMS_KERNELBENCH_BACKEND`           | `cuda`                                           | One of `cuda`, `triton`, `tilelang`, `cute`. |
| `LMMS_KERNELBENCH_SANDBOX`           | `none`                                           | `none` (bare subprocess, Colab default) or `docker` (one-shot container). |
| `LMMS_KERNELBENCH_DOCKER_IMAGE`      | `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`  | Image used when `SANDBOX=docker`. Must include `python`, `torch`, plus `kernelbench` (or install it in the entrypoint). |
| `LMMS_KERNELBENCH_DOCKER_MEM`        | `16g`                                            | Container memory cap. |
| `LMMS_KERNELBENCH_DOCKER_CPUS`       | `4`                                              | Container cpu cap. |
| `LMMS_KERNELBENCH_DOCKER_GPUS`       | `all`                                            | `--gpus` value, or `none`. |
| `LMMS_KERNELBENCH_DOCKER_EXTRA_ARGS` | unset                                            | Extra `docker run` args (whitespace-split). |

## Execution requirements

The subprocess imports `torch` + `kernelbench` and compiles / runs CUDA kernels,
so the runtime needs a CUDA GPU.

**Do not run this task on a machine you don't fully control** without
`SANDBOX=docker` — the model output is executed as Python code with the eval
runner's privileges.

* On **Colab** (project's intended runtime): leave `SANDBOX=none`. Colab is
  itself ephemeral.
* On a **workstation or shared host**: use `LMMS_KERNELBENCH_SANDBOX=docker`.
