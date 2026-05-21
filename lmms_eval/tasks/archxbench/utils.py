import json
import os
import re
import shutil
import subprocess
import tempfile

from datasets import Dataset, DatasetDict
from loguru import logger as eval_logger

from lmms_eval.api.task import ConfigurableMessagesTask

# ── Benchmark layout ──────────────────────────────────────────────────────────

_LEVELS = ["level-0", "level-1a", "level-1b", "level-1c", "level-2", "level-3", "level-4", "level-5", "level-6"]


def _get_benchmark_path() -> str:
    path = os.environ.get("ARCHXBENCH_PATH", "").strip()
    if not path:
        raise EnvironmentError(
            "ARCHXBENCH_PATH environment variable is not set.\n"
            "  export ARCHXBENCH_PATH=/path/to/ArchXBench"
        )
    return path


def _find_iverilog() -> str:
    """Return path to iverilog, preferring the archxbench conda env."""
    candidates = [
        "/root/miniconda3/envs/archxbench/bin/iverilog",
        shutil.which("iverilog") or "",
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    raise FileNotFoundError(
        "iverilog not found. Install it with:\n"
        "  conda install -n archxbench -c conda-forge iverilog"
    )


def _find_vvp() -> str:
    candidates = [
        "/root/miniconda3/envs/archxbench/bin/vvp",
        shutil.which("vvp") or "",
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    raise FileNotFoundError("vvp not found alongside iverilog.")


# ── Dataset loading ───────────────────────────────────────────────────────────

def _load_archxbench_data(benchmark_path: str) -> list[dict]:
    records = []
    for level in _LEVELS:
        level_dir = os.path.join(benchmark_path, level)
        if not os.path.isdir(level_dir):
            continue
        for task_name in sorted(os.listdir(level_dir)):
            task_dir = os.path.join(level_dir, task_name)
            if not os.path.isdir(task_dir):
                continue

            prob_path = os.path.join(task_dir, "problem-description.txt")
            spec_path = os.path.join(task_dir, "design-specs.txt")
            tb_path = os.path.join(task_dir, "tb.v")

            # Some level-4+ tasks use a different tb filename
            if not os.path.exists(tb_path):
                candidates = [f for f in os.listdir(task_dir) if f.startswith("tb") and f.endswith(".v")]
                tb_path = os.path.join(task_dir, candidates[0]) if candidates else None

            if not (os.path.exists(prob_path) and os.path.exists(spec_path) and tb_path):
                eval_logger.warning(f"[ArchXBench] Skipping {level}/{task_name}: missing required files")
                continue

            problem = open(prob_path).read().strip()
            specs = open(spec_path).read().strip()
            tb = open(tb_path).read().strip()

            # Extract module name from design-specs
            module_name = _extract_module_name(specs)

            # Numeric I/O (optional)
            stimuli_path = os.path.join(task_dir, "inputs", "stimuli.json")
            golden_path = os.path.join(task_dir, "outputs", "golden_output.json")
            has_numeric_io = os.path.exists(stimuli_path) and os.path.exists(golden_path)

            records.append({
                "task_id": f"{level}/{task_name}",
                "level": level,
                "task_name": task_name,
                "module_name": module_name,
                "problem_description": problem,
                "design_specs": specs,
                "testbench": tb,
                "testbench_filename": os.path.basename(tb_path),
                "task_dir": task_dir,
                "has_numeric_io": has_numeric_io,
            })

    eval_logger.info(f"[ArchXBench] Loaded {len(records)} tasks from {benchmark_path}")
    return records


def _extract_module_name(specs: str) -> str:
    """Extract the module name from design-specs.txt."""
    for line in specs.splitlines():
        line = line.strip()
        if line.startswith("Module Name:"):
            name = line.split(":", 1)[1].strip().lstrip("-").strip()
            if name:
                return name
        # Also try 'module <name> (' pattern in the Design Signature
        m = re.search(r"\bmodule\s+(\w+)\s*[#(]", line)
        if m:
            return m.group(1)
    return ""


# ── Code extraction ───────────────────────────────────────────────────────────

def _extract_verilog(text: str) -> str:
    """Extract Verilog from a markdown code block, or return text as-is."""
    match = re.search(
        r"```(?:verilog|systemverilog|sv|v)?\s*(.*?)```",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    return match.group(1).strip() if match else text.strip()


# ── Verilog simulation ────────────────────────────────────────────────────────

def _run_simulation(task_dir: str, tb_filename: str, dut_code: str, module_name: str, timeout: int = 60) -> dict:
    """
    Compile and simulate the DUT against the testbench.
    Returns a dict with keys: syntax_pass, function_pass, error_msg, tb_output.
    """
    iverilog = _find_iverilog()
    vvp = _find_vvp()

    with tempfile.TemporaryDirectory() as tmp:
        # Copy task_dir contents (stimuli, golden outputs, scripts, etc.)
        for item in os.listdir(task_dir):
            src = os.path.join(task_dir, item)
            dst = os.path.join(tmp, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

        # Write DUT code
        dut_path = os.path.join(tmp, f"{module_name}.v")
        with open(dut_path, "w") as f:
            f.write(dut_code)

        tb_path = os.path.join(tmp, tb_filename)
        sim_out = os.path.join(tmp, "sim.vvp")

        # Compile
        compile_result = subprocess.run(
            [iverilog, "-o", sim_out, tb_path, dut_path],
            capture_output=True, text=True, timeout=30,
        )
        if compile_result.returncode != 0:
            return {
                "syntax_pass": 0,
                "function_pass": 0,
                "error_msg": compile_result.stderr.strip(),
                "tb_output": "",
            }

        # Simulate
        try:
            sim_result = subprocess.run(
                [vvp, sim_out],
                capture_output=True, text=True, timeout=timeout,
                cwd=tmp,
            )
        except subprocess.TimeoutExpired:
            return {
                "syntax_pass": 1,
                "function_pass": 0,
                "error_msg": "simulation timeout",
                "tb_output": "",
            }

        tb_output = sim_result.stdout + sim_result.stderr

        # Parse pass/fail from testbench JSON output line
        # Testbenches print: {"module": "...", "passed": N, "failed": N}
        function_pass = _parse_tb_result(tb_output, sim_result.returncode)

        # For numeric-IO tasks, also run compare_outputs.py if dut_output.json was produced
        dut_output_path = os.path.join(tmp, "outputs", "dut_output.json")
        if os.path.exists(dut_output_path):
            function_pass = _compare_numeric_outputs(
                os.path.join(tmp, "outputs", "golden_output.json"),
                dut_output_path,
            )

        return {
            "syntax_pass": 1,
            "function_pass": function_pass,
            "error_msg": "",
            "tb_output": tb_output,
        }


def _parse_tb_result(tb_output: str, returncode: int) -> int:
    """Parse testbench stdout for pass/fail JSON line."""
    # Look for JSON summary line: {"module": "...", "passed": N, "failed": N}
    for line in reversed(tb_output.splitlines()):
        line = line.strip()
        if line.startswith("{") and "passed" in line and "failed" in line:
            try:
                data = json.loads(line)
                failed = int(data.get("failed", 1))
                return 1 if failed == 0 else 0
            except (json.JSONDecodeError, ValueError):
                pass
    # Fallback: check for [PASS] / [FAIL] markers or exit code
    if "[FAIL]" in tb_output:
        return 0
    if "[PASS]" in tb_output:
        return 1
    return 1 if returncode == 0 else 0


def _compare_numeric_outputs(golden_path: str, dut_path: str, tol: int = 1) -> int:
    """Compare numeric outputs with tolerance. Returns 1 if all match."""
    try:
        with open(golden_path) as f:
            ref = json.load(f)
        with open(dut_path) as f:
            dut = json.load(f)
        # Flatten if nested
        def flatten(x):
            if isinstance(x, list):
                for item in x:
                    yield from flatten(item)
            else:
                yield x
        ref = list(flatten(ref))
        dut = list(flatten(dut))
        n = min(len(ref), len(dut))
        mismatches = sum(1 for r, d in zip(ref[:n], dut[:n]) if abs(r - d) > tol)
        return 1 if mismatches == 0 else 0
    except Exception:
        return 0


# ── Prompt construction ───────────────────────────────────────────────────────

_DEFAULT_PRE_PROMPT = (
    "You are an expert digital hardware designer. "
    "Implement the following Verilog module exactly as specified. "
    "Output only the complete, synthesizable Verilog code wrapped in a ```verilog ... ``` code block. "
    "Do not include any explanation.\n\n"
)

_DEFAULT_POST_PROMPT = ""


def archxbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre = kwargs.get("pre_prompt", _DEFAULT_PRE_PROMPT)
    post = kwargs.get("post_prompt", _DEFAULT_POST_PROMPT)

    prompt = (
        f"## Problem Description\n\n{doc['problem_description']}\n\n"
        f"## Design Specification\n\n{doc['design_specs']}"
    )
    return pre + prompt + post


def archxbench_doc_to_visual(doc):
    return []


def archxbench_doc_to_target(doc):
    return ""


# ── Metrics ───────────────────────────────────────────────────────────────────

def archxbench_process_results(doc, results):
    code = _extract_verilog(results[0])
    sim = _run_simulation(
        task_dir=doc["task_dir"],
        tb_filename=doc["testbench_filename"],
        dut_code=code,
        module_name=doc["module_name"],
    )
    return {
        "syntax_pass_rate": sim["syntax_pass"],
        "function_pass_rate": sim["function_pass"],
    }


# ── Task class ────────────────────────────────────────────────────────────────

_METRIC_LIST = [
    {"metric": "syntax_pass_rate", "aggregation": "mean", "higher_is_better": True},
    {"metric": "function_pass_rate", "aggregation": "mean", "higher_is_better": True},
]

_GEN_KWARGS = {
    "temperature": 0,
    "do_sample": False,
}


class ArchXBenchTask(ConfigurableMessagesTask):
    """ArchXBench: LLM-driven RTL synthesis benchmark (61 tasks, level-0 to level-6)."""

    def __init__(self, config=None):
        task_config = {
            "task": "archxbench",
            "output_type": "generate_until",
            "test_split": "test",
            "doc_to_text": archxbench_doc_to_text,
            "doc_to_visual": archxbench_doc_to_visual,
            "doc_to_target": archxbench_doc_to_target,
            "process_results": archxbench_process_results,
            "generation_kwargs": dict(_GEN_KWARGS),
            "metric_list": list(_METRIC_LIST),
            "metadata": {"version": 0},
        }
        if config:
            task_config.update({k: v for k, v in config.items() if k != "class"})
        super().__init__(config=task_config)

    def download(self, dataset_kwargs=None):
        benchmark_path = _get_benchmark_path()
        data = _load_archxbench_data(benchmark_path)
        self.dataset = DatasetDict({"test": Dataset.from_list(data)})
        self.dataset_no_image = self.dataset
