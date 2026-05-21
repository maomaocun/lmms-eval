"""
lmms-eval glue for the SuperCoder task.

Heavy lifting (subprocess, gcc, hyperfine) lives in `executor.py`.
"""

from __future__ import annotations

import ast
import importlib.util
import math
import os
import sys

import datasets
from loguru import logger as eval_logger

# Same dual-mode import dance as the other ports — see tritonbench/utils.py.
try:
    from . import executor as _executor_mod  # type: ignore[no-redef]
except ImportError:
    _HERE = os.path.dirname(os.path.abspath(__file__))

    def _load_sibling(name: str):
        unique = f"_supercoder_{name}"
        spec = importlib.util.spec_from_file_location(unique, os.path.join(_HERE, f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[unique] = mod
        spec.loader.exec_module(mod)
        return mod

    _executor_mod = _load_sibling("executor")

executor = _executor_mod


# ---- process_docs ----------------------------------------------------------


def _coerce_str_list(v) -> list[str]:
    """The HF dataset stringifies inputs/outputs as a Python list literal."""
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, str):
        try:
            parsed = ast.literal_eval(v)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except (SyntaxError, ValueError):
            pass
    return []


def _normalize(example: dict) -> dict:
    extra = example.get("extra_info") or {}
    prompt_msgs = example.get("prompt") or []
    prompt_text = ""
    if prompt_msgs:
        first = prompt_msgs[0]
        if isinstance(first, dict):
            prompt_text = first.get("content", "") or ""
        elif isinstance(first, str):
            prompt_text = first

    return {
        "id": str(extra.get("index", "")),
        "prompt": prompt_text,
        "c_source": extra.get("c_code", "") or "",
        "baseline_asm": extra.get("unoptimized_assembly", "") or "",
        "gold_asm": extra.get("answer", "") or "",
        "inputs": _coerce_str_list(extra.get("inputs", [])),
        "outputs": _coerce_str_list(extra.get("outputs", [])),
    }


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.map(_normalize, remove_columns=dataset.column_names)


# ---- doc_to_text / target --------------------------------------------------


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    pre = (lmms_eval_specific_kwargs or {}).get("pre_prompt", "")
    post = (lmms_eval_specific_kwargs or {}).get("post_prompt", "")
    return f"{pre}{doc['prompt']}{post}"


def doc_to_target(doc):
    # The "target" for this task is informational only — scoring is execution-
    # based, not reference-matching. Surface the gold optimized assembly.
    return doc["gold_asm"]


# ---- env knobs -------------------------------------------------------------


def _dry_run() -> bool:
    return os.environ.get("LMMS_SUPERCODER_DRY_RUN", "").lower() in ("1", "true", "yes")


def _max_cases() -> int:
    try:
        return int(os.environ.get("LMMS_SUPERCODER_MAX_CASES", "10"))
    except ValueError:
        return 10


def _total_timeout() -> float:
    try:
        return float(os.environ.get("LMMS_SUPERCODER_TIMEOUT", "600"))
    except ValueError:
        return 600.0


# ---- process_results -------------------------------------------------------

_METRICS = ("correctness", "speedup", "fast_1")


def _zeroed(rid: str, *, error: str | None = None) -> dict:
    base = {"id": rid}
    if error is not None:
        base["error"] = error
    out = {}
    for m in _METRICS:
        v = dict(base)
        # speedup defaults to 1.0 (baseline ratio); other metrics to 0.
        v["value"] = 1.0 if m == "speedup" else 0.0
        out[m] = v
    return out


def process_results(doc, results):
    pred = results[0] if results else ""
    rid = doc["id"]

    if _dry_run():
        out = _zeroed(rid)
        for v in out.values():
            v["skipped"] = True
        return out

    out = executor.score_one(
        model_raw=pred,
        baseline_asm=doc["baseline_asm"],
        inputs=doc["inputs"],
        outputs=doc["outputs"],
        max_cases=_max_cases(),
        total_timeout=_total_timeout(),
    )

    if "error" in out:
        eval_logger.warning(f"supercoder {rid}: {out['error']}")

    return {
        m: {"id": rid, "value": float(out.get(m, 0.0 if m != "speedup" else 1.0)), **({"error": out["error"]} if "error" in out else {}), **({"n_cases": out["n_cases"], "n_passed": out["n_passed"]} if "n_cases" in out else {})}
        for m in _METRICS
    }


# ---- aggregations ----------------------------------------------------------


def _mean(items):
    if not items:
        return 0.0
    return sum(it["value"] for it in items) / len(items)


def _geomean(items):
    """Geometric mean of (positive) values; clamps each value to >= 1e-12."""
    if not items:
        return 0.0
    ln_sum = 0.0
    for it in items:
        v = max(float(it["value"]), 1e-12)
        ln_sum += math.log(v)
    return math.exp(ln_sum / len(items))


def aggregate_correctness(results):
    return _mean(results)


def aggregate_speedup_geomean(results):
    return _geomean(results)


def aggregate_fast_1(results):
    return _mean(results)
