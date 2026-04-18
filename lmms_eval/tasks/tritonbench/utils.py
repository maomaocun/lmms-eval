"""
lmms-eval glue for the TritonBench tasks.

Heavy lifting:
  * `data.py`     — fetches gold reference test harnesses from upstream (lazy + cached)
  * `executor.py` — extracts code, runs subprocess, computes call/exec accuracy

The dataset itself (instruction + gold output + meta) is loaded by HF datasets
directly from the upstream raw URLs declared in the YAML configs. `process_docs`
normalizes that to the columns we use throughout this module.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import datasets
from loguru import logger as eval_logger

# lmms-eval loads `utils.py` via importlib.util.spec_from_file_location, which
# means we're not part of a proper package and `from . import executor` would
# fail. When imported normally (e.g. from a Colab notebook via
# `lmms_eval.tasks.tritonbench.utils`), we *are* in a package and the relative
# import works. Try the package import first, then fall back to file loading.
try:
    from . import data as _data_mod  # type: ignore[no-redef]
    from . import executor as _executor_mod  # type: ignore[no-redef]
except ImportError:
    _HERE = os.path.dirname(os.path.abspath(__file__))

    def _load_sibling(name: str):
        unique = f"_tritonbench_{name}"
        spec = importlib.util.spec_from_file_location(unique, os.path.join(_HERE, f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[unique] = mod
        spec.loader.exec_module(mod)
        return mod

    _executor_mod = _load_sibling("executor")
    _data_mod = _load_sibling("data")

executor = _executor_mod
_data = _data_mod

# ---- process_docs ------------------------------------------------------------


def _normalize(example: dict, track: str) -> dict:
    """Project an upstream meta record into the canonical schema we use."""
    file_name = example.get("file") or ""
    return {
        "id": file_name.removesuffix(".py") if file_name else "",
        "track": track,
        "file_name": file_name,
        "instruction_simp": example.get("simp_instru", "") or "",
        "instruction_comp": example.get("comp_instru", "") or "",
        "gold_output": example.get("output", "") or "",
        "difficulty": str(example.get("difficulty", "")),
        "repo": example.get("repo", "") or "",
        "star": int(example.get("star") or 0),
    }


def process_docs_g(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.map(lambda ex: _normalize(ex, "G"), remove_columns=dataset.column_names)


def process_docs_t(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.map(lambda ex: _normalize(ex, "T"), remove_columns=dataset.column_names)


# ---- doc_to_text -------------------------------------------------------------


def _doc_to_text(doc: dict, lmms_eval_specific_kwargs, key: str) -> str:
    pre = (lmms_eval_specific_kwargs or {}).get("pre_prompt", "")
    post = (lmms_eval_specific_kwargs or {}).get("post_prompt", "")
    return f"{pre}{doc[key]}{post}"


def doc_to_text_simp(doc, lmms_eval_specific_kwargs=None):
    return _doc_to_text(doc, lmms_eval_specific_kwargs, "instruction_simp")


def doc_to_text_comp(doc, lmms_eval_specific_kwargs=None):
    return _doc_to_text(doc, lmms_eval_specific_kwargs, "instruction_comp")


def doc_to_target(doc):
    return doc["gold_output"]


# ---- process_results ---------------------------------------------------------


def _dry_run() -> bool:
    return os.environ.get("LMMS_TRITONBENCH_DRY_RUN", "").lower() in ("1", "true", "yes")


def _exec_timeout() -> float:
    try:
        return float(os.environ.get("LMMS_TRITONBENCH_TIMEOUT", "120"))
    except ValueError:
        return 120.0


def process_results(doc, results):
    """Run the model's prediction through the call/exec accuracy pipeline.
    `results` is a one-element list with the raw model response."""
    pred = results[0] if results else ""
    rid, track = doc["id"], doc["track"]

    if _dry_run():
        return {
            "call_acc": {"id": rid, "track": track, "value": 0.0, "skipped": True},
            "exec_acc": {"id": rid, "track": track, "value": 0.0, "skipped": True},
        }

    try:
        gold_src = _data.gold_test_src(track, doc["file_name"])
    except Exception as e:  # network / cache failure
        eval_logger.warning(f"tritonbench: ref fetch failed for {rid}: {e}")
        return {
            "call_acc": {"id": rid, "track": track, "value": 0.0, "error": str(e)},
            "exec_acc": {"id": rid, "track": track, "value": 0.0, "error": str(e)},
        }

    out = executor.score_one(
        model_raw=pred,
        gold_test_src=gold_src,
        timeout=_exec_timeout(),
    )
    return {
        "call_acc": {"id": rid, "track": track, "value": out["call_acc"]},
        "exec_acc": {"id": rid, "track": track, "value": out["exec_acc"]},
    }


# ---- aggregations ------------------------------------------------------------


def _mean(items):
    if not items:
        return 0.0
    return sum(it["value"] for it in items) / len(items)


def aggregate_call_acc(results):
    return _mean(results)


def aggregate_exec_acc(results):
    return _mean(results)
