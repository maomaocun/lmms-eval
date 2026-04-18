"""
lmms-eval glue for the CAD-Coder task.

The only truly multimodal port in this set: the model receives a 448x448 CAD
rendering plus a fixed text prompt and returns CadQuery Python source.
Scoring delegates to executor.py (subprocess running cadquery).
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import datasets
from loguru import logger as eval_logger

# Same dual-mode import as the other ports.
try:
    from . import executor as _executor_mod  # type: ignore[no-redef]
except ImportError:
    _HERE = os.path.dirname(os.path.abspath(__file__))

    def _load_sibling(name: str):
        unique = f"_cadcoder_{name}"
        spec = importlib.util.spec_from_file_location(unique, os.path.join(_HERE, f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[unique] = mod
        spec.loader.exec_module(mod)
        return mod

    _executor_mod = _load_sibling("executor")

executor = _executor_mod


# ---- cache ----------------------------------------------------------------

def _cache_root() -> Path:
    override = os.environ.get("LMMS_CADCODER_CACHE")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".cache" / "lmms_eval" / "cad_coder"


def _gold_step_path(deepcad_id: str) -> str:
    # deepcad_id looks like "0000/00006371" — preserve the subdirectory split.
    return str(_cache_root() / "gold_steps" / f"{deepcad_id}.step")


# ---- process_docs ----------------------------------------------------------

def _is_truthy(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return bool(v)


def _normalize(example: dict) -> dict:
    return {
        "id": example.get("deepcad_id", "") or "",
        "image": example["image"],
        "prompt": example.get("prompt", "") or "",
        "gold_cadquery": example.get("cadquery", "") or "",
        "hundred_subset": _is_truthy(example.get("hundred_subset", False)),
    }


def process_docs_full(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.map(_normalize, remove_columns=[c for c in dataset.column_names if c != "image"])


def process_docs_hundred_subset(dataset: datasets.Dataset) -> datasets.Dataset:
    """The 100-sample test subset that the CAD-Coder paper reports IoU on."""
    filtered = dataset.filter(lambda ex: _is_truthy(ex.get("hundred_subset", False)))
    return filtered.map(_normalize, remove_columns=[c for c in filtered.column_names if c != "image"])


# ---- doc_to_visual / text / target ----------------------------------------

def doc_to_visual(doc):
    img = doc["image"]
    if hasattr(img, "convert"):
        img = img.convert("RGB")
    return [img]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    pre = (lmms_eval_specific_kwargs or {}).get("pre_prompt", "")
    post = (lmms_eval_specific_kwargs or {}).get("post_prompt", "")
    return f"{pre}{doc['prompt']}{post}"


def doc_to_target(doc):
    return doc["gold_cadquery"]


# ---- env knobs ------------------------------------------------------------

def _dry_run() -> bool:
    return os.environ.get("LMMS_CADCODER_DRY_RUN", "").lower() in ("1", "true", "yes")


def _exec_timeout() -> float:
    try:
        return float(os.environ.get("LMMS_CADCODER_EXEC_TIMEOUT", "60"))
    except ValueError:
        return 60.0


def _total_timeout() -> float:
    try:
        return float(os.environ.get("LMMS_CADCODER_TIMEOUT", "300"))
    except ValueError:
        return 300.0


def _skip_iou() -> bool:
    return os.environ.get("LMMS_CADCODER_SKIP_IOU", "").lower() in ("1", "true", "yes")


# ---- process_results ------------------------------------------------------

_METRICS = ("valid_syntax", "valid_step", "iou")


def _zeroed(rid: str, *, error: str | None = None) -> dict:
    base = {"id": rid, "value": 0.0}
    if error is not None:
        base["error"] = error
    return {m: dict(base) for m in _METRICS}


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
        gold_code=doc["gold_cadquery"],
        gold_step_path=_gold_step_path(rid),
        exec_timeout=_exec_timeout(),
        total_timeout=_total_timeout(),
        skip_iou=_skip_iou(),
    )

    if "error" in out:
        eval_logger.warning(f"cad_coder {rid}: {out['error']}")

    return {
        m: {"id": rid, "value": float(out.get(m, 0.0)),
            **({"error": out["error"]} if "error" in out else {})}
        for m in _METRICS
    }


# ---- aggregations ---------------------------------------------------------

def aggregate_mean(results):
    if not results:
        return 0.0
    return sum(r["value"] for r in results) / len(results)


def aggregate_iou_mean(results):
    """Mean IoU. Per upstream, samples that fail to produce a valid STEP get
    IoU=0 (already set by executor)."""
    return aggregate_mean(results)
