# CAD-Coder (lmms-eval port)

Adaptation of [anniedoris/CAD-Coder](https://github.com/anniedoris/CAD-Coder) — *CAD-Coder: An Open-Source Vision-Language Model for Computer-Aided Design Code* ([paper](https://decode.mit.edu/assets/papers/IDETC_CadCode_decodeweb.pdf)) — to the `lmms-eval` framework.

The model is given a 448×448 CAD rendering plus a fixed text prompt and must return CadQuery Python code that, when executed, produces the same 3D solid.

This is the only **truly multimodal** task in the domain-bench set: `doc_to_visual` returns the image, `doc_to_text` returns the prompt.

## Tasks

| Task name             | Test set                                | Problems |
| --------------------- | --------------------------------------- | -------- |
| `cad_coder_test100`   | `test` filtered to `hundred_subset==True` (matches the paper's 100-sample IoU benchmark) | 100      |
| `cad_coder_test`      | full `test` split                       | 7,355    |

Dataset: [`CADCoder/GenCAD-Code`](https://huggingface.co/datasets/CADCoder/GenCAD-Code) (147k train + 8.2k validation + 7.36k test).

## Metrics (all `higher_is_better`)

| Metric         | Per-problem definition                                                                          |
| -------------- | ----------------------------------------------------------------------------------------------- |
| `valid_syntax` | 1.0 iff the generated CadQuery script runs to completion in a subprocess (no Python exception)  |
| `valid_step`   | 1.0 iff a STEP file is successfully exported from the executed solid                            |
| `iou`          | Volumetric IoU between model and gold solids, aligned via center-of-mass + principal axes (see `cq_align_shapes` upstream). Samples without a valid STEP get IoU = 0. |

`iou` aggregation is the **arithmetic mean** across the eval set, matching upstream's `compute_iou.py`. The paper reports IoU = 0.675 for CAD-Coder on the 100-sample test subset.

## Eval pipeline (per problem)

1. Strip ` ```python ` fences from the model output.
2. **Phase A:** Run the script in a subprocess (no STEP export). Sets `valid_syntax`.
3. **Phase B:** If syntax-valid, append `cq.exporters.export(solid, "<path>.step")` and re-run. Sets `valid_step`.
4. **Phase C:** If `valid_step`, ensure the gold STEP exists in the on-disk cache (regenerate from `cadquery` field if missing), then compute IoU via `cq_align_shapes` (4-candidate reflection search).

All four phases happen inside one runner subprocess so a misbehaving model script can't take down the whole eval.

## Cache & runtime knobs

| Env var                               | Default                                             | Effect |
| ------------------------------------- | --------------------------------------------------- | ------ |
| `LMMS_CADCODER_CACHE`                 | `~/.cache/lmms_eval/cad_coder`                      | Cache root for gold STEP files. |
| `LMMS_CADCODER_DRY_RUN`               | unset                                               | Truthy → skip subprocess; `valid_syntax=valid_step=iou=0`. |
| `LMMS_CADCODER_EXEC_TIMEOUT`          | `60s`                                               | Per-phase Python subprocess timeout. |
| `LMMS_CADCODER_TIMEOUT`               | `300s`                                              | Total per-problem wall-clock cap. |
| `LMMS_CADCODER_SKIP_IOU`              | unset                                               | Truthy → skip IoU phase, only report syntax/STEP rates. Faster for sweeps. |
| `LMMS_CADCODER_SANDBOX`               | `none`                                              | `none` (Colab default) or `docker`. |
| `LMMS_CADCODER_DOCKER_IMAGE`          | `cadquery/cadquery:latest`                          | Image must include `python3`, `cadquery`, `numpy`. |
| `LMMS_CADCODER_DOCKER_MEM`            | `4g`                                                | Container memory cap. |
| `LMMS_CADCODER_DOCKER_CPUS`           | `2`                                                 | Container cpu cap. |
| `LMMS_CADCODER_DOCKER_EXTRA_ARGS`     | unset                                               | Extra `docker run` args (whitespace-split). |

In Docker mode, the gold-step cache directory is mounted read-write at `/work/gold` (host path = `<cache>/gold_steps/`).

## Execution requirements

The runtime needs the `cadquery` Python package (which brings in OpenCASCADE):

```bash
pip install cadquery
```

**Do not run on a machine you don't fully control** without `SANDBOX=docker`
— the model output is executed as Python code with the eval runner's
privileges.

* On **Colab**: `SANDBOX=none` is fine — Colab is itself ephemeral. Add `pip install cadquery` to the setup cell.
* On a **workstation or shared host**: `SANDBOX=docker` with the official `cadquery/cadquery:latest` image (or your own that has `cadquery` + `numpy`).
