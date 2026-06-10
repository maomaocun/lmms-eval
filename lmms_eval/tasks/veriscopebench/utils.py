import os
import re
import shutil
import subprocess
import tempfile

import yaml
from datasets import Dataset, DatasetDict
from loguru import logger as eval_logger

from lmms_eval.api.task import ConfigurableMessagesTask

_LEVELS = ["L1_basic", "L2_sequential", "L3_module", "L4_system", "L5_ultimate"]


def _get_benchmark_path() -> str:
    path = os.environ.get("VERISCOPEBENCH_PATH", "").strip()
    if not path:
        raise EnvironmentError(
            "VERISCOPEBENCH_PATH environment variable is not set.\n"
            "  export VERISCOPEBENCH_PATH=/mnt/data/data_zoo/VeriScope"
        )
    return path


def _find_tool(name: str) -> str:
    candidates = [
        f"/root/miniconda3/envs/veriscopebench/bin/{name}",
        f"/root/miniconda3/envs/archxbench/bin/{name}",
        shutil.which(name) or "",
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    raise FileNotFoundError(f"{name} not found. Install with: conda install -c conda-forge iverilog")


def _load_data(benchmark_path: str) -> list[dict]:
    records = []
    for level in _LEVELS:
        level_dir = os.path.join(benchmark_path, "bundles", level)
        if not os.path.isdir(level_dir):
            continue
        for task_name in sorted(os.listdir(level_dir)):
            task_dir = os.path.join(level_dir, task_name)
            if not os.path.isdir(task_dir):
                continue

            problem_path = os.path.join(task_dir, "problem.md")
            config_path = os.path.join(task_dir, "config.yaml")
            tb_path = os.path.join(task_dir, "testbench.v")

            if not all(os.path.exists(p) for p in [problem_path, config_path, tb_path]):
                eval_logger.warning(f"[VeriScope] Skipping {level}/{task_name}: missing required files")
                continue

            problem = open(problem_path).read().strip()
            tb = open(tb_path).read().strip()
            cfg = yaml.safe_load(open(config_path).read())

            module_name = cfg.get("module_name", "")
            timeout_compile = cfg.get("timeout_compile", 10)
            timeout_simulate = cfg.get("timeout_simulate", 30)

            records.append({
                "task_id": f"{level}/{task_name}",
                "level": level,
                "task_name": task_name,
                "module_name": module_name,
                "problem": problem,
                "testbench": tb,
                "task_dir": task_dir,
                "timeout_compile": timeout_compile,
                "timeout_simulate": timeout_simulate,
            })

    eval_logger.info(f"[VeriScope] Loaded {len(records)} tasks from {benchmark_path}")
    return records


def _extract_verilog(text: str) -> str:
    match = re.search(r"```(?:verilog|systemverilog|sv|v)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()


def _run_simulation(doc: dict, dut_code: str) -> dict:
    iverilog = _find_tool("iverilog")
    vvp = _find_tool("vvp")

    with tempfile.TemporaryDirectory() as tmp:
        dut_path = os.path.join(tmp, f"{doc['module_name']}.v")
        tb_path = os.path.join(tmp, "testbench.v")
        sim_out = os.path.join(tmp, "sim.vvp")

        with open(dut_path, "w") as f:
            f.write(dut_code)
        with open(tb_path, "w") as f:
            f.write(doc["testbench"])

        compile_result = subprocess.run(
            [iverilog, "-o", sim_out, tb_path, dut_path],
            capture_output=True, text=True, timeout=doc["timeout_compile"],
        )
        if compile_result.returncode != 0:
            return {"syntax_pass": 0, "function_pass": 0}

        try:
            sim_result = subprocess.run(
                [vvp, sim_out],
                capture_output=True, text=True, timeout=doc["timeout_simulate"],
                cwd=tmp,
            )
        except subprocess.TimeoutExpired:
            return {"syntax_pass": 1, "function_pass": 0}

        tb_output = sim_result.stdout + sim_result.stderr
        function_pass = _parse_tb_result(tb_output, sim_result.returncode)
        return {"syntax_pass": 1, "function_pass": function_pass}


def _parse_tb_result(tb_output: str, returncode: int) -> int:
    if "TEST PASSED" in tb_output:
        return 1
    if "TEST FAILED" in tb_output:
        return 0
    if "[FAIL]" in tb_output:
        return 0
    if "[PASS]" in tb_output:
        return 1
    return 1 if returncode == 0 else 0


_DEFAULT_PRE_PROMPT = (
    "You are an expert digital hardware designer. "
    "Implement the following Verilog module exactly as specified. "
    "Output only the complete, synthesizable Verilog code wrapped in a ```verilog ... ``` code block. "
    "Do not include any explanation.\n\n"
)


def veriscopebench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre = kwargs.get("pre_prompt", _DEFAULT_PRE_PROMPT)
    return pre + doc["problem"]


def veriscopebench_doc_to_visual(doc):
    return []


def veriscopebench_doc_to_target(doc):
    return ""


def veriscopebench_process_results(doc, results):
    code = _extract_verilog(results[0])
    sim = _run_simulation(doc, code)
    return {
        "syntax_pass_rate": sim["syntax_pass"],
        "function_pass_rate": sim["function_pass"],
    }


_METRIC_LIST = [
    {"metric": "syntax_pass_rate", "aggregation": "mean", "higher_is_better": True},
    {"metric": "function_pass_rate", "aggregation": "mean", "higher_is_better": True},
]

_GEN_KWARGS = {
    "temperature": 0,
    "do_sample": False,
}


class VeriScopeBenchTask(ConfigurableMessagesTask):
    """VeriScope: 568-problem Verilog design benchmark (L1-L5)."""

    def __init__(self, config=None):
        task_config = {
            "task": "veriscopebench",
            "output_type": "generate_until",
            "test_split": "test",
            "doc_to_text": veriscopebench_doc_to_text,
            "doc_to_visual": veriscopebench_doc_to_visual,
            "doc_to_target": veriscopebench_doc_to_target,
            "process_results": veriscopebench_process_results,
            "generation_kwargs": dict(_GEN_KWARGS),
            "metric_list": list(_METRIC_LIST),
            "metadata": {"version": 0},
        }
        if config:
            task_config.update({k: v for k, v in config.items() if k != "class"})
        super().__init__(config=task_config)

    def download(self, dataset_kwargs=None):
        benchmark_path = _get_benchmark_path()
        data = _load_data(benchmark_path)
        self.dataset = DatasetDict({"test": Dataset.from_list(data)})
        self.dataset_no_image = self.dataset
