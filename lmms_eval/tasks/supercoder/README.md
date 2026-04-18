# SuperCoder (lmms-eval port)

Adaptation of [Anjiang-Wei/SuperCoder](https://github.com/Anjiang-Wei/SuperCoder) — *SuperCoder: Assembly Program Superoptimization with Large Language Models* ([arXiv:2505.11480](https://arxiv.org/abs/2505.11480)) — to the `lmms-eval` framework.

The benchmark asks the model to take a C program plus its `gcc -O3` x86-64 baseline assembly and produce a faster, semantically equivalent x86-64 assembly.

## Tasks

| Task name        | Split | Problems |
| ---------------- | ----- | -------- |
| `supercoder_val` | val   | 200      |

Dataset: [`LLM4Code/llm_superoptimizer_ds`](https://huggingface.co/datasets/LLM4Code/llm_superoptimizer_ds).

## Metrics (all `higher_is_better`)

| Metric        | Per-problem definition                                                                | Aggregation       |
| ------------- | ------------------------------------------------------------------------------------- | ----------------- |
| `correctness` | fraction of test cases whose stdout exactly matches the expected output (cap 10/case) | arithmetic mean   |
| `speedup`     | mean of per-test-case speedups (`baseline_mean / model_mean`, clamped ≥ 1.0)          | **geometric mean** |
| `fast_1`      | 1.0 iff all test cases pass AND mean speedup > 1.0                                    | arithmetic mean   |

Both correctness and speedup follow the upstream `src/test_benchmark.py` conventions: speedup is clamped to `1.0` from below for failed tests or hyperfine errors, so the geometric mean is well-defined across the eval set.

## Prompt

The HF dataset already contains a fully-formatted prompt under `prompt[0]['content']` (a single user message with the C source and the gcc -O3 assembly). `doc_to_text` passes it through unchanged. There's also a `c_only_question` field in `extra_info` for an experimental variant — not yet exposed as a separate task.

## Execution model

`process_results` extracts the model's assembly, drops it into a temp Python file that:

1. Compiles model and baseline assembly with `gcc <file>.s -o <bin> -no-pie`
2. Runs the model binary with each test input piped to stdin (cap 10 cases / problem)
3. Compares stdout to expected (exact equality)
4. For each passing case, runs `hyperfine --warmup 3 --runs 10 --time-unit millisecond` on both binaries and computes `baseline_mean / model_mean`

…and runs that script in a subprocess (optionally inside a one-shot Docker container).

### Runtime requirements

Host needs:

- `gcc` (x86-64 Linux toolchain)
- `hyperfine` (for speedup measurement; if missing, all per-case speedups silently fall back to 1.0)
- `python3` (only when `SANDBOX=docker`)

The Colab notebook (intended runtime) installs both via `apt-get install -y gcc hyperfine`.

## Runtime knobs

| Env var                              | Default     | Effect |
| ------------------------------------ | ----------- | ------ |
| `LMMS_SUPERCODER_DRY_RUN`            | unset       | Truthy → skip subprocess; `correctness=0`, `speedup=1.0`, `fast_1=0`. |
| `LMMS_SUPERCODER_TIMEOUT`            | `600s`      | Per-problem total wall-clock cap. |
| `LMMS_SUPERCODER_MAX_CASES`          | `10`        | Test cases evaluated per problem. |
| `LMMS_SUPERCODER_SANDBOX`            | `none`      | `none` (Colab default) or `docker`. |
| `LMMS_SUPERCODER_DOCKER_IMAGE`       | `gcc:13`    | Image must include `gcc`, `hyperfine`, `python3`. |
| `LMMS_SUPERCODER_DOCKER_MEM`         | `4g`        | Container memory cap. |
| `LMMS_SUPERCODER_DOCKER_CPUS`        | `2`         | Container cpu cap. |
| `LMMS_SUPERCODER_DOCKER_EXTRA_ARGS`  | unset       | Extra `docker run` args (whitespace-split). |

## Execution requirements & safety

The subprocess compiles and runs arbitrary x86-64 assembly produced by the
model. **Do not run on a machine you don't fully control** without
`SANDBOX=docker`.

- On **Colab**: `SANDBOX=none` is fine — Colab is itself ephemeral. Install `gcc`, `hyperfine` in the notebook setup cell.
- On a **workstation or shared host**: `SANDBOX=docker`. The default `gcc:13` image needs hyperfine added; either bake your own image or use `LMMS_SUPERCODER_DOCKER_EXTRA_ARGS="--entrypoint sh"` and supply an init.
