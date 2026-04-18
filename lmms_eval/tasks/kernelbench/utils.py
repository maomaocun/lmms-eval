"""
lmms-eval glue for the KernelBench tasks.

Heavy lifting (subprocess, sandbox, upstream eval invocation) lives in
`executor.py`. This module is the thin lmms-eval-side shim.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import datasets
from loguru import logger as eval_logger

# See tritonbench/utils.py for why we need this dual-mode import.
try:
    from . import executor as _executor_mod  # type: ignore[no-redef]
except ImportError:
    _HERE = os.path.dirname(os.path.abspath(__file__))

    def _load_sibling(name: str):
        unique = f"_kernelbench_{name}"
        spec = importlib.util.spec_from_file_location(unique, os.path.join(_HERE, f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[unique] = mod
        spec.loader.exec_module(mod)
        return mod

    _executor_mod = _load_sibling("executor")

executor = _executor_mod


# ---- prompt template -------------------------------------------------------

# Mirrors upstream's "cuda / zero_shot" template (src/kernelbench/prompts/prompts.toml).
# Components: problem_statement + arch_block + precision_note + instruction.
_PROMPT_TEMPLATE = (
    "You write custom CUDA operators to replace the pytorch operators in the "
    "given architecture to get speedups.\n\n"
    "You are given the following architecture:\n\n"
    "{ref_arch_src}\n\n"
    "Note: The kernels should be optimized for FP32 (32-bit floating point) "
    "precision.\n\n"
    "Optimize the architecture named Model with custom CUDA operators! "
    "Name your optimized output architecture ModelNew. Output the new code in "
    "codeblocks."
)


# ---- process_docs ----------------------------------------------------------


def _normalize(example: dict) -> dict:
    return {
        "id": f"level{example['level']}/{example['name']}",
        "level": int(example["level"]),
        "problem_id": int(example["problem_id"]),
        "name": example["name"],
        "ref_arch_src": example["code"],
    }


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.map(_normalize, remove_columns=dataset.column_names)


# ---- doc_to_text / target --------------------------------------------------


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    pre = (lmms_eval_specific_kwargs or {}).get("pre_prompt", "")
    post = (lmms_eval_specific_kwargs or {}).get("post_prompt", "")
    body = _PROMPT_TEMPLATE.format(ref_arch_src=doc["ref_arch_src"])
    return f"{pre}{body}{post}"


def doc_to_target(doc):
    # KernelBench has no "gold optimized" output — the reference IS the target;
    # scoring measures speedup over it. We expose the reference as the nominal
    # target so the framework's bookkeeping has something to hold.
    return doc["ref_arch_src"]


# ---- env knobs -------------------------------------------------------------


def _dry_run() -> bool:
    return os.environ.get("LMMS_KERNELBENCH_DRY_RUN", "").lower() in ("1", "true", "yes")


def _exec_timeout() -> float:
    try:
        return float(os.environ.get("LMMS_KERNELBENCH_TIMEOUT", "300"))
    except ValueError:
        return 300.0


def _num_correct() -> int:
    try:
        return int(os.environ.get("LMMS_KERNELBENCH_NUM_CORRECT", "5"))
    except ValueError:
        return 5


def _num_perf() -> int:
    try:
        return int(os.environ.get("LMMS_KERNELBENCH_NUM_PERF", "100"))
    except ValueError:
        return 100


def _backend() -> str:
    return os.environ.get("LMMS_KERNELBENCH_BACKEND", "cuda")


# ---- process_results -------------------------------------------------------

_METRICS = ("compiled", "correctness", "fast_1", "fast_2")


def _zeroed(rid: str, level: int, *, error: str | None = None) -> dict:
    base = {"id": rid, "level": level, "value": 0.0}
    if error is not None:
        base["error"] = error
    return {m: dict(base) for m in _METRICS}


def process_results(doc, results):
    pred = results[0] if results else ""
    rid, level = doc["id"], doc["level"]

    if _dry_run():
        out = _zeroed(rid, level)
        for v in out.values():
            v["skipped"] = True
        return out

    out = executor.score_one(
        reference_src=doc["ref_arch_src"],
        model_raw=pred,
        num_correct=_num_correct(),
        num_perf=_num_perf(),
        backend=_backend(),
        timeout=_exec_timeout(),
    )

    if "error" in out:
        eval_logger.warning(f"kernelbench {rid}: {out['error']}")

    return {m: {"id": rid, "level": level, "value": out.get(m, 0.0), **({"error": out["error"]} if "error" in out else {}), **({"speedup": out["speedup"]} if "speedup" in out else {})} for m in _METRICS}


# ---- aggregations ----------------------------------------------------------


def _mean(items):
    if not items:
        return 0.0
    return sum(it["value"] for it in items) / len(items)


def aggregate_compiled(results):
    return _mean(results)


def aggregate_correctness(results):
    return _mean(results)


def aggregate_fast_1(results):
    return _mean(results)


def aggregate_fast_2(results):
    return _mean(results)
