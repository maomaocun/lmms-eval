"""Offline scorer: recompute task metrics from saved --log_samples JSONL files."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger as eval_logger

from lmms_eval.evaluator_utils import (
    TaskOutput,
    consolidate_group_results,
    consolidate_results,
    get_subtask_list,
    prepare_print_tasks,
)
from lmms_eval.tasks import TaskManager
from lmms_eval.utils import make_table


# Keys that are framework-generated and should not be part of reconstructed doc.
_FRAMEWORK_KEYS = {
    "doc_id",
    "target",
    "filtered_resps",
    "token_counts",
    "doc_hash",
    "input",
    "resps",
    "input_media",
    "arguments",
    "doc",
}


def _reconstruct_doc(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Reconstruct a pseudo-doc from a JSONL sample by excluding framework keys."""
    return {k: v for k, v in sample.items() if k not in _FRAMEWORK_KEYS}


def _compute_doc_hash(doc: Dict[str, Any]) -> str:
    """Compute the same doc_hash that evaluate() uses."""
    from lmms_eval.utils import handle_non_serializable
    serialized = json.dumps(doc, indent=2, default=handle_non_serializable, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _load_all_eval_docs(task) -> List[Dict[str, Any]]:
    """Load the full evaluation docs list for a task.
    
    During evaluate(), docs are often sourced from eval_docs_no_media,
    so we prefer that for hash matching.
    """
    if hasattr(task, "eval_docs_no_media"):
        return list(task.eval_docs_no_media)
    if task.has_test_docs():
        return list(task.eval_docs)
    else:
        return list(task.validation_docs)


def _resolve_doc_for_sample(sample: Dict[str, Any], all_docs: List[Dict[str, Any]], verbose: bool = False) -> Optional[Dict[str, Any]]:
    """Find the original doc matching a sample, first by doc_id then by doc_hash."""
    doc_id = sample.get("doc_id")
    doc_hash = sample.get("doc_hash")

    # Fast path: doc_id is the index in the sliced iterator.
    # For most single-node runs with offset=0, it's simply the index in all_docs.
    if doc_id is not None and 0 <= doc_id < len(all_docs):
        candidate = all_docs[doc_id]
        # Best-effort hash verification (skip if mismatch due to serialization quirks)
        if _compute_doc_hash(candidate) == doc_hash:
            if verbose:
                eval_logger.debug(f"Sample {doc_id}: matched doc by doc_id + hash.")
            return candidate
        if verbose:
            eval_logger.debug(f"Sample {doc_id}: using doc_id match (hash mismatch likely due to serialization differences).")
        return candidate

    # Fallback: scan by hash
    if doc_hash:
        for doc in all_docs:
            if _compute_doc_hash(doc) == doc_hash:
                return doc

    return None


def score_file(
    input_file: Path,
    task_name: str,
    output_path: Optional[Path] = None,
    verbose: bool = False,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Recompute per-sample and aggregate metrics from a samples JSONL.

    Args:
        input_file: Path to *_samples_<task>.jsonl.
        task_name: Name of the task to load.
        output_path: Optional directory to write results.json into.
        verbose: Whether to log at DEBUG level.

    Returns:
        (results_dict, samples_dict) compatible with the framework's output,
        or (None, None) on fatal error.
    """
    eval_logger.info(f"[score] Loading samples from {input_file}")
    samples: List[Dict[str, Any]] = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not samples:
        eval_logger.error("No samples found in input file.")
        return None, None

    eval_logger.info(f"[score] Loaded {len(samples)} samples for task '{task_name}'")

    # Load task
    task_manager = TaskManager(verbosity="DEBUG" if verbose else "INFO")
    try:
        task_dict = task_manager.load_task_or_group([task_name])
    except Exception as e:
        eval_logger.error(f"Failed to load task '{task_name}': {e}")
        return None, None

    task = task_dict.get(task_name)
    if task is None:
        eval_logger.error(f"Task '{task_name}' not found in registry.")
        return None, None

    # Warnings for known-hard cases
    if getattr(task.config, "process_results_use_image", False):
        eval_logger.warning(
            f"Task '{task_name}' has process_results_use_image=True; "
            "images will be reloaded from the dataset if possible."
        )
    if getattr(task.config, "full_docs", False):
        eval_logger.warning(
            f"Task '{task_name}' has full_docs=True; "
            "scoring may be incomplete because full dataset context is not available in JSONL."
        )

    # Pre-load evaluation docs for hash matching
    eval_logger.info(f"[score] Loading evaluation docs for '{task_name}' ...")
    try:
        all_docs = _load_all_eval_docs(task)
    except Exception as e:
        eval_logger.warning(f"Could not preload eval docs: {e}. Will rely on pseudo-doc reconstruction.")
        all_docs = []
    else:
        eval_logger.info(f"[score] Preloaded {len(all_docs)} eval docs.")

    # Build TaskOutput
    task_output = TaskOutput.from_taskdict(task_name, task)

    for idx, sample in enumerate(samples):
        doc = _resolve_doc_for_sample(sample, all_docs, verbose=verbose)
        if doc is None:
            doc = _reconstruct_doc(sample)
            if verbose:
                eval_logger.debug(f"Sample {idx}: using pseudo-doc reconstruction.")
        else:
            if verbose:
                eval_logger.debug(f"Sample {idx}: matched original doc by hash/doc_id.")

        filtered_resps = sample.get("filtered_resps", "")
        # Ensure list form expected by process_results for generate_until
        if isinstance(filtered_resps, str):
            results_arg = [filtered_resps]
        elif isinstance(filtered_resps, list):
            results_arg = filtered_resps
        else:
            results_arg = [str(filtered_resps)]

        try:
            if getattr(task.config, "full_docs", False) and all_docs:
                metrics = task.process_results(doc, results_arg, full_docs=all_docs)
            else:
                metrics = task.process_results(doc, results_arg)
        except Exception as e:
            if verbose:
                eval_logger.debug(f"process_results failed for sample {idx}: {e}")
            # Fallback: use any metrics already present in the JSONL
            metrics = {k: v for k, v in sample.items() if k not in _FRAMEWORK_KEYS and isinstance(v, (int, float, bool))}
            if not metrics:
                eval_logger.warning(f"Sample {idx}: process_results failed and no fallback metrics found. Skipping.")
                continue

        # Rebuild logged sample (same schema as evaluate())
        logged_sample = {
            "doc_id": sample.get("doc_id", idx),
            "target": sample.get("target", ""),
            "filtered_resps": filtered_resps,
            "token_counts": sample.get("token_counts"),
            "doc_hash": sample.get("doc_hash", ""),
            "input": sample.get("input", ""),
        }
        logged_sample.update(metrics)
        task_output.logged_samples.append(logged_sample)

        for metric_name, value in metrics.items():
            # We only track metrics for the default "none" filter in MVP.
            task_output.sample_metrics[(metric_name, "none")].append(value)

    # Aggregation
    try:
        task_output.calculate_aggregate_metric(bootstrap_iters=100000)
    except Exception as e:
        eval_logger.error(f"calculate_aggregate_metric failed: {e}")
        return None, None

    try:
        task_output.calculate_clt_aggregate_metric()
    except Exception as e:
        eval_logger.debug(f"calculate_clt_aggregate_metric failed: {e}")

    try:
        task_output.calculate_stability_metrics()
    except Exception as e:
        eval_logger.debug(f"calculate_stability_metrics failed: {e}")

    eval_tasks = [task_output]
    results, samples_out, configs, versions, num_fewshot, higher_is_better = consolidate_results(eval_tasks)

    if bool(results):
        results, versions, show_group_table, *_ = consolidate_group_results(
            results, versions, task_dict
        )

    results_agg, group_agg = prepare_print_tasks(task_dict, results)
    subtask_list = get_subtask_list(task_dict)

    _higher_is_better = {}
    for group, task_list in subtask_list.items():
        if len(task_list) != 0:
            for t in task_list:
                for m, h in higher_is_better[t].items():
                    if m not in _higher_is_better:
                        _higher_is_better[m] = h
                    if _higher_is_better[m] is not None and _higher_is_better[m] != h:
                        _higher_is_better[m] = None
                higher_is_better[group] = _higher_is_better

    results_dict = {
        "results": dict(results_agg.items()),
        **({"groups": dict(group_agg.items())} if (bool(group_agg) & show_group_table) else {}),
        "group_subtasks": dict(reversed(subtask_list.items())),
        "configs": dict(sorted(configs.items())),
        "versions": dict(sorted(versions.items())),
        "n-shot": dict(sorted(num_fewshot.items())),
        "higher_is_better": dict(sorted(higher_is_better.items())),
        "n-samples": {
            task_output.task_name: {
                "original": len(all_docs) if all_docs else (len(task.eval_docs) if hasattr(task, "eval_docs") else len(samples)),
                "effective": len(samples),
            }
            for task_output in eval_tasks
        },
    }
    results_dict["samples"] = dict(samples_out)

    # Print table
    print(f"\n{'='*60}")
    print("OFFLINE SCORE RESULTS")
    print(f"{'='*60}")
    print(make_table(results_dict))
    if "groups" in results_dict:
        print(make_table(results_dict, "groups"))
    print(f"{'='*60}\n")

    # Save results.json
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
        out_file = output_path / f"{input_file.stem}_results.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False, default=str)
        eval_logger.info(f"[score] Saved results to {out_file}")

    return results_dict, samples_out
