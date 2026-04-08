"""lmms-eval judge subcommand: standalone judging from JSONL files.

This module provides a CLI interface for judging model outputs from JSONL files
without regeneration. It separates the generation and judging phases completely.

Usage:
    lmms-eval judge --input results.jsonl --task mathvision_reason_testmini
    lmms-eval judge -i "*.jsonl" --judge-mode llm --judge-model gpt-4o
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger as eval_logger

from lmms_eval.utils import get_eval_banner, make_table


def add_judge_parser(subparsers):
    """Add judge subcommand to CLI."""
    parser = subparsers.add_parser(
        "judge",
        help="Judge model outputs from JSONL files without regeneration",
        description="""
Standalone judge command for evaluating model outputs from JSONL files.

This command separates the generation and judging phases, allowing you to:
1. Re-judge existing results with different criteria
2. Use LLM-as-judge for tasks that normally use rule-based judging
3. Batch process multiple result files

Examples:
    # Basic usage with auto-detected task from a single file
    lmms-eval judge --input_result results/model_samples_task.jsonl

    # Specify single task explicitly
    lmms-eval judge --input_result results.jsonl -t mathvision_reason_testmini

    # Judge multiple tasks from a directory
    lmms-eval judge -i /path/to/results/ -t mathvision_test,wemath_testmini_reasoning

    # Use LLM judge
    lmms-eval judge -i results.jsonl --judge-mode llm --judge-model gpt-4o

    # Batch process with output directory
    lmms-eval judge --input_result "results/*.jsonl" -d judged/ --parallel 8
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input_result", "-i",
        required=True,
        help="Path to JSONL result file(s). Supports wildcards (*.jsonl)",
    )
    parser.add_argument(
        "--task", "-t",
        default="auto-detect",
        help="Task name(s) for loading process_results. Use comma-separated list for multiple tasks (e.g., 'task1,task2'). Use 'auto-detect' to infer from filename(s). When multiple tasks are given, --input_result should be a directory.",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output JSONL file path (single file mode)",
    )
    parser.add_argument(
        "--output-dir", "-d",
        help="Output directory (batch mode)",
    )
    parser.add_argument(
        "--judge-mode",
        choices=["rule", "llm", "auto"],
        default=os.getenv("JUDGE_MODE", "auto"),
        help="Judging mode: rule=rule-based only, llm=LLM judge, auto=rule first then LLM fallback (default: from JUDGE_MODE env or auto)",
    )
    parser.add_argument(
        "--judge-model",
        default=os.getenv("JUDGE_MODEL", "gpt-4o-mini"),
        help="Judge model name (default: from JUDGE_MODEL env var or gpt-4o-mini)",
    )
    parser.add_argument(
        "--judge-api-key",
        default=os.getenv("JUDGE_API_KEY"),
        help="API key for judge model (default: from JUDGE_API_KEY env var)",
    )
    parser.add_argument(
        "--judge-base-url",
        default=os.getenv("JUDGE_BASE_URL", "https://api.openai.com/v1"),
        help=(
            "Base URL for judge API. "
            "For local vLLM/SGLang: http://localhost:8000/v1 "
            "(default: from JUDGE_BASE_URL env or OpenAI default)"
        ),
    )
    parser.add_argument(
        "--parallel", "-p",
        type=int,
        default=int(os.getenv("JUDGE_MAX_CONCURRENT", "1")),
        help="Number of parallel judge workers (default: from JUDGE_MAX_CONCURRENT env or 1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run without saving results",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.set_defaults(func=run_judge)


def _detect_task_from_filename(filename: str) -> str:
    """Extract task name from samples filename.
    
    Example patterns:
        - '20240328_samples_mathvision_reason_testmini.jsonl' -> 'mathvision_reason_testmini'
        - 'model_Qwen_samples_mmmu_val.jsonl' -> 'mmmu_val'
        - 'samples_wemath_testmini.jsonl' -> 'wemath_testmini'
    """
    # Remove .jsonl extension
    name = filename.replace(".jsonl", "")
    
    # Try to find _samples_ pattern
    if "_samples_" in name:
        parts = name.split("_samples_")
        if len(parts) >= 2:
            return parts[1]
    
    # Try to find just samples_ pattern
    if "samples_" in name:
        parts = name.split("samples_")
        if len(parts) >= 2:
            return parts[1]
    
    raise ValueError(
        f"Cannot auto-detect task from filename: {filename}. "
        f"Expected pattern: *_samples_{{task}}.jsonl"
    )


def _resolve_input_files(input_result: str, task_list: List[str]) -> List[Tuple[str, Path]]:
    """Resolve input files for given tasks.
    
    Mimics the evaluation framework's multi-task selection by allowing
    comma-separated task names. When multiple tasks are provided,
    --input_result is treated as a directory and matching files are
    auto-discovered using the *samples_<task>.jsonl pattern.
    
    Returns:
        List of (task_name, input_file_path) tuples.
        task_name may be "auto-detect" for wildcard/directory modes.
    """
    input_path = Path(input_result)

    # Case 1: Wildcard pattern
    if "*" in input_result:
        files = sorted(Path(".").glob(input_result))
        if not files:
            raise ValueError(f"No files found matching pattern: {input_result}")
        # Always auto-detect task from filename when wildcards are used
        return [("auto-detect", f) for f in files]

    # Case 2: Single file
    if input_path.is_file():
        if len(task_list) > 1:
            raise ValueError(
                f"Multiple tasks specified but --input_result is a single file. "
                f"Please provide a directory or use a single task."
            )
        return [(task_list[0], input_path)]

    # Case 3: Directory
    if input_path.is_dir():
        result = []
        if task_list == ["auto-detect"]:
            files = sorted(input_path.glob("*samples_*.jsonl"))
            if not files:
                raise ValueError(f"No *samples_*.jsonl files found in directory: {input_path}")
            for f in files:
                try:
                    task = _detect_task_from_filename(f.name)
                    result.append((task, f))
                except ValueError:
                    eval_logger.warning(f"Skipping file with unrecognized pattern: {f.name}")
            return result
        else:
            for task in task_list:
                pattern = f"*samples_{task}.jsonl"
                files = sorted(input_path.glob(pattern))
                if not files:
                    raise ValueError(
                        f"No file found for task: {task} (pattern: {pattern}) in directory: {input_path}"
                    )
                # Pick the latest file by mtime (same logic as shell script)
                latest = max(files, key=lambda p: p.stat().st_mtime)
                result.append((task, latest))
            return result

    raise ValueError(f"Input path not found: {input_path}")


def _get_output_path(input_file: Path, output: Optional[str], output_dir: Optional[str]) -> Path:
    """Determine output file path."""
    if output:
        return Path(output)
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / input_file.name
    # Default: add _judged suffix
    return input_file.parent / f"{input_file.stem}_judged.jsonl"


def _build_results_dict(task_name: str, summary: dict) -> dict:
    """Build a results dict compatible with make_table."""
    return {
        "results": {
            task_name: {
                f"{k}": v for k, v in summary.items()
            }
        },
        "n-shot": {task_name: " "},
        "higher_is_better": {task_name: {}},
    }


def run_judge(args: argparse.Namespace) -> None:
    """Execute judge command."""

    def _setup_logger():
        """Configure logging to match the framework style."""
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
        from lmms_eval.llm_judge.standalone import JudgeRunner
    except ImportError as e:
        eval_logger.error(f"Failed to import JudgeRunner: {e}")
        eval_logger.error("Please ensure lmms-eval is installed: pip install -e .")
        sys.exit(1)

    # Some sub-modules (e.g. lmms_eval.models) reset the global loguru logger
    # during their first import. Re-configure after heavy imports are done.
    _setup_logger()

    # Parse task list (mimics evaluation framework's --tasks comma separation)
    if args.task == "auto-detect":
        task_list = ["auto-detect"]
    else:
        task_list = [t.strip() for t in args.task.split(",") if t.strip()]
    if not task_list:
        eval_logger.error("No tasks specified.")
        sys.exit(1)

    # Expand group tasks into their subtasks so that judge can find sample files
    if task_list != ["auto-detect"]:
        try:
            from lmms_eval.tasks import get_task_dict
            from lmms_eval.evaluator_utils import get_subtask_list

            def _collect_leaf_tasks(subtasks):
                """Recursively collect leaf task names from get_subtask_list result."""
                leaves = []
                for name, children in subtasks.items():
                    if not children:
                        leaves.append(name)
                    else:
                        leaves.extend(children)
                return leaves

            expanded_task_list = []
            for task_name in task_list:
                try:
                    task_dict = get_task_dict(task_name)
                    subtasks = get_subtask_list(task_dict)
                    leaves = _collect_leaf_tasks(subtasks)
                    if leaves:
                        expanded_task_list.extend(leaves)
                    else:
                        expanded_task_list.append(task_name)
                except Exception:
                    # If resolution fails, keep the original name
                    expanded_task_list.append(task_name)
            task_list = expanded_task_list
        except Exception as e:
            eval_logger.debug(f"Failed to expand group tasks: {e}")

    # Resolve input files for the requested tasks
    try:
        judge_items = _resolve_input_files(args.input_result, task_list)
    except ValueError as e:
        eval_logger.error(str(e))
        sys.exit(1)

    if not judge_items:
        eval_logger.error("No files to judge.")
        sys.exit(1)

    eval_logger.info(f"Found {len(judge_items)} file(s) to judge")
    for task_name, input_file in judge_items:
        eval_logger.info(f"  [{task_name}] -> {input_file}")

    # Initialize runner
    runner = JudgeRunner(
        judge_mode=args.judge_mode,
        judge_model=args.judge_model,
        judge_api_key=args.judge_api_key,
        judge_base_url=args.judge_base_url,
        parallel=args.parallel,
    )

    # Print judge config at the start (same style as normal evaluation)
    eval_logger.info(
        f"judge ({args.input_result}), judge_mode: ({args.judge_mode}), "
        f"judge_model: ({args.judge_model}), parallel: {args.parallel}"
    )

    # Process each file
    success_count = 0
    error_count = 0
    all_summaries = []

    for task_name, input_file in judge_items:
        # Auto-detect task from filename if needed
        if task_name == "auto-detect":
            try:
                task_name = _detect_task_from_filename(input_file.name)
            except ValueError as e:
                eval_logger.error(f"{e}. Use --task to specify explicitly.")
                error_count += 1
                continue

        try:
            # Run judging
            results = runner.judge_file(input_file, task_name)

            # Compute summary
            summary = runner.compute_summary(results)
            if summary:
                all_summaries.append((task_name, summary))

            # Save results
            if not args.dry_run:
                output_path = _get_output_path(input_file, args.output, args.output_dir)
                runner.save_results(results, output_path)

            success_count += 1

        except Exception as e:
            eval_logger.error(f"Error processing {input_file}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            error_count += 1

    # Inject group-level summaries for hierarchical display
    def _load_group_map():
        import yaml
        tasks_dir = Path(__file__).parent.parent / "tasks"
        group_map = {}
        for yaml_file in tasks_dir.rglob("*.yaml"):
            try:
                with open(yaml_file, "r") as f:
                    data = yaml.safe_load(f)
                if data and isinstance(data, dict) and "group" in data and "task" in data:
                    members = [str(t) for t in data["task"] if isinstance(t, str)]
                    if members:
                        group_map[str(data["group"])] = members
            except Exception:
                continue
        return group_map

    def _inject_group_rows(summaries):
        group_map = _load_group_map()
        task_index = {name: (idx, summary) for idx, (name, summary) in enumerate(summaries)}
        grouped_tasks = set()
        new_rows = []

        for group_name, members in group_map.items():
            member_present = []
            for m in members:
                if m in task_index:
                    member_present.append(m)
            if not member_present:
                continue

            # Aggregate numeric metrics from member summaries.
            # We look for the first usable float per summary (prefer exact "score",
            # then keys ending with ".score", then any numeric value).
            scores = []
            for m in member_present:
                s = task_index[m][1]
                val = None
                if "score" in s and isinstance(s["score"], (int, float)):
                    val = float(s["score"])
                else:
                    for k, v in s.items():
                        if k.endswith(".score") and isinstance(v, (int, float)):
                            val = float(v)
                            break
                        elif isinstance(v, (int, float)):
                            val = float(v)
                            break
                if val is not None:
                    scores.append(val)
            if scores:
                group_summary = {"score": round(sum(scores) / len(scores), 4)}
            else:
                group_summary = {}

            # Insert group header + indented members + group total
            new_rows.append((group_name, group_summary))
            for m in member_present:
                grouped_tasks.add(m)
                orig_summary = task_index[m][1]
                new_rows.append((f"  {m}", orig_summary))

        # Append any tasks that are not part of a group
        for name, summary in summaries:
            if name not in grouped_tasks:
                new_rows.append((name, summary))

        return new_rows

    all_summaries = _inject_group_rows(all_summaries)

    # Log results in the same style as normal evaluation
    if all_summaries:
        combined_results = {}
        combined_nshot = {}
        combined_hib = {}
        for task_name, summary in all_summaries:
            combined_results[task_name] = {f"{k}": v for k, v in summary.items()}
            combined_nshot[task_name] = " "
            combined_hib[task_name] = {}
        combined_dict = {
            "results": combined_results,
            "n-shot": combined_nshot,
            "higher_is_better": combined_hib,
        }
        eval_logger.info("\n" + make_table(combined_dict))

    if error_count > 0:
        sys.exit(1)
