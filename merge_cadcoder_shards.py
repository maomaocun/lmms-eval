"""Merge per-shard CAD-Coder samples into a single full-dataset metric.

Usage:
    python3 merge_cadcoder_shards.py logs/cadcoder_shards_YYYYMMDD_HHMM
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from loguru import logger


def _find_samples(shard_dir: Path) -> Path | None:
    matches = sorted(shard_dir.glob("**/*samples*.jsonl"))
    return matches[-1] if matches else None


def _aggregate(samples: list[dict]) -> dict:
    metrics = ("valid_syntax", "valid_step", "iou")
    agg: dict[str, float] = {}
    for m in metrics:
        vals: list[float] = []
        for s in samples:
            entry = s.get(m)
            if isinstance(entry, dict) and "value" in entry:
                vals.append(float(entry["value"]))
            elif isinstance(entry, (int, float)):
                vals.append(float(entry))
        agg[m] = (sum(vals) / len(vals)) if vals else 0.0
        agg[f"{m}_n"] = len(vals)
    agg["total_samples"] = len(samples)
    return agg


def _sample_id(sample: dict, shard_name: str, line_number: int) -> str:
    if sample.get("doc_id") is not None:
        return f"doc_id:{sample['doc_id']}"

    metric_id = sample.get("valid_syntax", {}).get("id")
    if metric_id:
        return f"metric_id:{metric_id}"

    doc_hash = sample.get("doc_hash")
    if doc_hash:
        return f"doc_hash:{doc_hash}"

    return f"{shard_name}:line:{line_number}"


def main() -> int:
    if len(sys.argv) < 2:
        logger.error("Usage: python3 merge_cadcoder_shards.py <shards_root_dir>")
        return 1

    root = Path(sys.argv[1]).resolve()
    if not root.is_dir():
        logger.error(f"Not a directory: {root}")
        return 1

    shard_dirs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("shard_"))
    if not shard_dirs:
        logger.error(f"No shard_* subdirectories under {root}")
        return 1

    logger.info(f"Merging {len(shard_dirs)} shards under {root}")

    all_samples: list[dict] = []
    seen_ids: set[str] = set()

    for sd in shard_dirs:
        samples_file = _find_samples(sd)
        if samples_file is None:
            logger.warning(f"  {sd.name}: no samples_*.jsonl yet (still running?)")
            continue
        n_added = 0
        n_dup = 0
        try:
            with samples_file.open() as fh:
                for line_number, line in enumerate(fh, start=1):
                    if not line.strip():
                        continue
                    s = json.loads(line)
                    sid = _sample_id(s, sd.name, line_number)
                    if sid in seen_ids:
                        n_dup += 1
                        continue
                    seen_ids.add(sid)
                    all_samples.append(s)
                    n_added += 1
            logger.info(f"  {sd.name}: +{n_added} samples ({n_dup} duplicates)  <- {samples_file.name}")
        except Exception as e:
            logger.opt(exception=e).error(f"  {sd.name}: failed to read {samples_file}")

    if not all_samples:
        logger.error("No samples merged. Aborting.")
        return 1

    agg = _aggregate(all_samples)
    logger.info("=" * 60)
    logger.info(f"CAD-Coder merged across {len(shard_dirs)} shards")
    logger.info(f"  total samples : {agg['total_samples']}")
    logger.info(f"  valid_syntax  : {agg['valid_syntax']:.4f}  (n={agg['valid_syntax_n']})")
    logger.info(f"  valid_step    : {agg['valid_step']:.4f}  (n={agg['valid_step_n']})")
    logger.info(f"  iou           : {agg['iou']:.4f}  (n={agg['iou_n']})")
    logger.info("=" * 60)

    out_path = root / "merged_results.json"
    out_path.write_text(json.dumps({"results": {"cad_coder_test": agg}}, indent=2))
    logger.info(f"Wrote merged result -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
