# Industrial Benchmark Report - Qwen3-VL-8B-Instruct

**Run dates:** 2026-04-29 to 2026-04-30
**Model:** Qwen3-VL-8B-Instruct
**Backend:** vLLM, tensor parallel 2, greedy decoding
**Hardware:** 8 x H800, one GPU pair per benchmark job

This report summarizes the four industrial/code-generation benchmarks added to this lmms-eval branch: CAD-Coder, KernelBench, SuperCoder, and TritonBench. The first full 4-way run completed KernelBench, SuperCoder, and TritonBench. CAD-Coder was later completed as a 4-shard run because the monolithic CAD job was terminated after the other benchmarks finished.

## Summary

| Benchmark | Task(s) run | Samples | Headline metric | Score |
| --- | --- | ---: | --- | ---: |
| CAD-Coder | `cad_coder_test` | 7,355 | `iou` | 0.0467 |
| KernelBench | `kernelbench_level1`-`kernelbench_level4` | 270 | mean `correctness` | 0.148 |
| SuperCoder | `supercoder_val` | 200 | `correctness` | 0.1695 |
| TritonBench | `tritonbench_g`, `tritonbench_t` | 267 | best `exec_acc` | 0.0482 |

All benchmark scores come from real subprocess execution of generated code. The run scripts check that dry-run and skip-scoring environment variables are unset before launching production runs.

## CAD-Coder

CAD-Coder evaluates image-plus-text generation of CadQuery programs. The full `cad_coder_test` split has 7,355 samples. It was run with `run_cadcoder_shards.sh` as four independent shards, then merged with `merge_cadcoder_shards.py`.

| Metric | Score |
| --- | ---: |
| `valid_syntax` | 0.2091 |
| `valid_step` | 0.1410 |
| `iou` | 0.0467 |

Reproduction:

```bash
bash run_cadcoder_shards.sh
python3 merge_cadcoder_shards.py logs/cadcoder_shards_YYYYMMDD_HHMM
```

## KernelBench

KernelBench evaluates custom CUDA kernel generation across four levels.

| Level | Samples | `compiled` | `correctness` | `fast_1` | `fast_2` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `kernelbench_level1` | 100 | 0.69 | 0.16 | 0.04 | 0.01 |
| `kernelbench_level2` | 100 | 0.87 | 0.23 | 0.05 | 0.00 |
| `kernelbench_level3` | 50 | 0.56 | 0.20 | 0.16 | 0.04 |
| `kernelbench_level4` | 20 | 0.00 | 0.00 | 0.00 | 0.00 |

## SuperCoder

SuperCoder evaluates x86-64 assembly superoptimization against a `gcc -O3` baseline.

| Task | Samples | `correctness` | `speedup` | `fast_1` |
| --- | ---: | ---: | ---: | ---: |
| `supercoder_val` | 200 | 0.1695 | 1.0064 | 0.1550 |

## TritonBench

TritonBench evaluates generated Triton GPU kernels.

| Task | `call_acc` | `exec_acc` |
| --- | ---: | ---: |
| `tritonbench_g` | 0.0000 | 0.0000 |
| `tritonbench_t` | 0.0482 | 0.0482 |

## Run Scripts

The PR includes the scripts used for production and verification runs:

- `run_4benches_full.sh`: starts the four benchmark families in parallel.
- `run_4benches_test1.sh`: runs a one-sample-per-family pipeline check.
- `run_cadcoder_shards.sh`: runs full CAD-Coder as four shards.
- `merge_cadcoder_shards.py`: merges CAD-Coder shard sample files into one aggregate result.

Raw run artifacts are intentionally left under `logs/`, which is ignored by git. The committed report records the metrics needed to review the benchmark accomplishment without adding large generated outputs.
