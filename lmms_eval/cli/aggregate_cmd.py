"""lmms-eval aggregate subcommand: aggregate judged results with task-specific logic.

This module provides a CLI interface for aggregating per-sample judged results
into final task metrics, handling complex aggregation logic like WeMath's multi-step
analysis.

Usage:
    lmms-eval aggregate --input judged_results.jsonl --task wemath_testmini_reasoning
    lmms-eval aggregate --input judged_results.jsonl --task wemath_testmini_reasoning --metric wemath_strict
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger as eval_logger

from lmms_eval.utils import get_eval_banner, make_table


def add_aggregate_parser(subparsers):
    """Add aggregate subcommand to CLI."""
    parser = subparsers.add_parser(
        "aggregate",
        help="Aggregate judged results with task-specific logic",
        description="""
Aggregate per-sample judged results into final task metrics.

This command handles complex aggregation logic that cannot be done per-sample,
such as WeMath's multi-step analysis (CompleteMastery, RoteMemorization, etc.).

Examples:
    # Aggregate WeMath results (default: both loose and strict)
    lmms-eval aggregate --input judged_wemath.jsonl --task wemath_testmini_reasoning

    # Aggregate specific metric only
    lmms-eval aggregate --input judged_wemath.jsonl --task wemath_testmini_reasoning --metric wemath_loose

    # Output to specific file
    lmms-eval aggregate --input judged_wemath.jsonl --task wemath_testmini_reasoning --output results.json
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to judged JSONL file from 'lmms-eval judge'",
    )
    parser.add_argument(
        "--task", "-t",
        required=True,
        help="Task name for loading aggregation function",
    )
    parser.add_argument(
        "--metric", "-m",
        default=None,
        help="Specific metric to aggregate (e.g., wemath_loose, wemath_strict). If not specified, aggregates all available metrics.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file path for aggregated results (JSON format)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.set_defaults(func=run_aggregate)


def run_aggregate(args: argparse.Namespace) -> None:
    """Execute aggregate command."""
    
    def _setup_logger():
        """Configure logging."""
        eval_logger.remove()
        # Check if colors should be disabled (for clean log files)
        use_color = os.environ.get('NO_COLOR', '') == '' and os.environ.get('LOGURU_NO_COLOR', '') == ''
        if use_color:
            log_format = (
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                "<level>{message}</level>"
            )
        else:
            log_format = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
        log_level = "DEBUG" if args.verbose else "INFO"
        eval_logger.add(sys.stdout, colorize=use_color, level=log_level, format=log_format)

    _setup_logger()

    # Import here to avoid heavy imports during CLI parsing
    try:
        from lmms_eval.llm_judge.aggregator import Aggregator
    except ImportError as e:
        eval_logger.error(f"Failed to import Aggregator: {e}")
        eval_logger.error("Please ensure lmms-eval is installed: pip install -e .")
        sys.exit(1)

    # Re-configure after heavy imports
    _setup_logger()

    input_path = Path(args.input)
    if not input_path.exists():
        eval_logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    eval_logger.info(f"Aggregating results from: {input_path}")
    eval_logger.info(f"Task: {args.task}")
    if args.metric:
        eval_logger.info(f"Metric: {args.metric}")

    # Initialize aggregator
    aggregator = Aggregator()

    try:
        # Load samples
        samples = _load_jsonl(input_path)
        eval_logger.info(f"Loaded {len(samples)} samples")

        if len(samples) == 0:
            eval_logger.error("No samples found in input file")
            sys.exit(1)

        # Run aggregation
        results = aggregator.aggregate(samples, args.task, metric_name=args.metric)

        if not results:
            eval_logger.warning("No aggregation results produced")
            sys.exit(0)

        # Display results
        _display_results(args.task, results)

        # Save results if output path specified
        if args.output:
            _save_results(results, Path(args.output))
            eval_logger.info(f"Results saved to: {args.output}")

    except Exception as e:
        eval_logger.error(f"Aggregation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                samples.append(sample)
            except json.JSONDecodeError as e:
                eval_logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
    return samples


def _display_results(task_name: str, results: Dict[str, Any]) -> None:
    """Display aggregation results in table format."""
    # Build results dict compatible with make_table
    table_data = {
        "results": {
            task_name: {}
        },
        "n-shot": {task_name: " "},
        "higher_is_better": {task_name: {}},
    }
    
    # Flatten results into the table format
    for metric_name, metric_value in results.items():
        if isinstance(metric_value, (int, float, str)):
            table_data["results"][task_name][metric_name] = metric_value
    
    print("\n" + "=" * 60)
    print("AGGREGATION RESULTS")
    print("=" * 60)
    print(make_table(table_data))
    
    # Also print detailed breakdown if available
    print("\nDetailed Results:")
    for key, value in results.items():
        print(f"  {key}: {value}")
    print("=" * 60 + "\n")


def _save_results(results: Dict[str, Any], output_path: Path) -> None:
    """Save results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
