import json
import os
import re
import subprocess
import tempfile

from datasets import Dataset, DatasetDict
from loguru import logger as eval_logger

from lmms_eval.api.task import ConfigurableMessagesTask, ConfigurableTask, TaskConfig

# ── benchmark_info (mirrors RealBench/benchmark_info.py) ──────────────────────
_BENCHMARK_INFO = {
    "sdc": [
        "sd_bd", "sd_clock_divider", "sd_crc_16", "sd_crc_7",
        "sd_controller_wb", "sd_data_master", "sd_cmd_master",
        "sd_rx_fifo", "sd_tx_fifo", "sd_fifo_rx_filler",
        "sd_fifo_tx_filler", "sd_data_serial_host", "sd_cmd_serial_host",
        "sdc_controller",
    ],
    "aes": [
        "aes_sbox", "aes_rcon", "aes_inv_sbox",
        "aes_key_expand_128", "aes_cipher_top", "aes_inv_cipher_top",
    ],
    "e203_hbirdv2": [
        "e203_biu", "e203_clk_ctrl", "e203_clkgate", "e203_core",
        "e203_cpu", "e203_cpu_top", "e203_dtcm_ctrl", "e203_dtcm_ram",
        "e203_extend_csr", "e203_exu", "e203_exu_alu", "e203_exu_alu_bjp",
        "e203_exu_alu_csrctrl", "e203_exu_alu_dpath", "e203_exu_alu_lsuagu",
        "e203_exu_alu_muldiv", "e203_exu_alu_rglr", "e203_exu_branchslv",
        "e203_exu_commit", "e203_exu_csr", "e203_exu_decode",
        "e203_exu_disp", "e203_exu_excp", "e203_exu_longpwbck",
        "e203_exu_nice", "e203_exu_oitf", "e203_exu_regfile",
        "e203_exu_wbck", "e203_ifu", "e203_ifu_ifetch", "e203_ifu_ift2icb",
        "e203_ifu_litebpu", "e203_ifu_minidec", "e203_irq_sync",
        "e203_itcm_ctrl", "e203_itcm_ram", "e203_lsu", "e203_lsu_ctrl",
        "e203_reset_ctrl", "e203_srams",
    ],
}

_MODULE_TO_SYSTEM = {
    module: system
    for system, modules in _BENCHMARK_INFO.items()
    for module in modules
}


def _get_benchmark_path() -> str:
    path = os.environ.get("REALBENCH_PATH", "").strip()
    if not path:
        raise EnvironmentError(
            "REALBENCH_PATH environment variable is not set. "
            "Set it to the absolute path of the RealBench directory, e.g.:\n"
            "  export REALBENCH_PATH=/path/to/RealBench"
        )
    return path


def _get_tmp_base() -> str | None:
    try:
        candidate = f"/run/user/{os.getuid()}"
        if os.path.isdir(candidate) and os.access(candidate, os.W_OK):
            return candidate
    except Exception:
        pass
    return None


def _extract_code(text: str) -> str:
    """Extract Verilog code from a markdown code block, or return text as-is."""
    match = re.search(
        r"```(?:verilog|systemverilog|sv|v)?\s*(.*?)```",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    return match.group(1).strip() if match else text.strip()


# ── verification helpers (adapted from RealBench/run_verify.py) ───────────────

def _run_testbench(template_dir: str, top_name: str, code: str):
    """
    Copy verification files to a temp dir, replace the top module with
    generated code, run `make all`, and return (syntax, function, err_msgs).
    """
    with tempfile.TemporaryDirectory(dir=_get_tmp_base()) as tmp:
        ret = subprocess.run(
            f"cp {template_dir}/* {tmp}/",
            shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE,
        )

        top_path = os.path.join(tmp, f"{top_name}_top.sv")
        if not os.path.exists(top_path):
            return -2, -2, "top file missing in verification dir", ""
        os.remove(top_path)
        with open(top_path, "w") as f:
            f.write(code)

        try:
            result = subprocess.run(
                f"cd {tmp} && make all",
                shell=True, timeout=300,
                stderr=subprocess.PIPE, stdout=subprocess.PIPE,
            )
        except subprocess.TimeoutExpired:
            return -2, -2, "verification timeout", ""

        if result.stderr:
            err = result.stderr.decode(errors="replace")
            syntax_msg = "\n".join(
                l for l in err.splitlines()
                if l.startswith("%Error") or l.startswith("%Warning")
            )
            return 0, 0, syntax_msg, ""

        tb_out = result.stdout.decode(errors="replace")
        func_msg = "\n".join(
            l[6:] for l in tb_out.splitlines()
            if "Hint: Output" in l and "mismatches" in l and "no mismatches" not in l
        )
        return 1, (1 if not func_msg else 0), "", func_msg


def testbench_verification_module(code: str, system_name: str, module_name: str, benchmark_path: str):
    template_dir = os.path.join(benchmark_path, system_name, module_name, "verification")
    return _run_testbench(template_dir, module_name, code)


def testbench_verification_system(code: str, system_name: str, benchmark_path: str):
    template_dir = os.path.join(benchmark_path, "system", system_name)
    return _run_testbench(template_dir, system_name, code)


# ── doc helpers ───────────────────────────────────────────────────────────────

def realbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre = kwargs.get(
        "pre_prompt",
        "Please implement the following Verilog/SystemVerilog module. "
        "Output only the complete module code wrapped in a ```verilog ... ``` code block.\n\n",
    )
    post = kwargs.get("post_prompt", "")
    return pre + doc["problem"] + post


def realbench_doc_to_visual(doc):
    return []


def realbench_doc_to_target(doc):
    return ""


# ── process_results ───────────────────────────────────────────────────────────

def realbench_module_process_results(doc, results):
    code = _extract_code(results[0])
    module_name = doc["task"]
    system_name = doc.get("system") or _MODULE_TO_SYSTEM.get(module_name, "")
    benchmark_path = _get_benchmark_path()

    syntax, function, _, _ = testbench_verification_module(
        code, system_name, module_name, benchmark_path
    )
    return {
        "syntax_pass_rate": int(syntax == 1),
        "function_pass_rate": int(function == 1),
    }


def realbench_system_process_results(doc, results):
    code = _extract_code(results[0])
    system_name = doc["task"]
    benchmark_path = _get_benchmark_path()

    syntax, function, _, _ = testbench_verification_system(
        code, system_name, benchmark_path
    )
    return {
        "syntax_pass_rate": int(syntax == 1),
        "function_pass_rate": int(function == 1),
    }


# ── task classes ──────────────────────────────────────────────────────────────

_METRIC_LIST = [
    {"metric": "syntax_pass_rate", "aggregation": "mean", "higher_is_better": True},
    {"metric": "function_pass_rate", "aggregation": "mean", "higher_is_better": True},
]

_BASE_GEN_KWARGS = {
    "max_new_tokens": 32768,
    "temperature": 0,
    "do_sample": False,
}


class RealBenchModuleTask(ConfigurableMessagesTask):
    """Module-level Verilog code generation benchmark (AES / SDC / E203)."""

    def __init__(self, config=None):
        task_config = {
            "task": "realbench_module",
            "output_type": "generate_until",
            "test_split": "test",
            "doc_to_text": realbench_doc_to_text,
            "doc_to_visual": realbench_doc_to_visual,
            "doc_to_target": realbench_doc_to_target,
            "process_results": realbench_module_process_results,
            "generation_kwargs": dict(_BASE_GEN_KWARGS),
            "metric_list": list(_METRIC_LIST),
            "metadata": {"version": 0},
        }
        if config:
            task_config.update({k: v for k, v in config.items() if k != "class"})
        super().__init__(config=task_config)

    def download(self, dataset_kwargs=None):
        benchmark_path = _get_benchmark_path()
        data = []
        for system in _BENCHMARK_INFO:
            jsonl_path = os.path.join(benchmark_path, "problems", system, "problems.jsonl")
            if not os.path.exists(jsonl_path):
                raise FileNotFoundError(
                    f"Problems file not found: {jsonl_path}\n"
                    f"Run the following in the RealBench directory first:\n"
                    f"  make decrypt && make clean_encrypt\n"
                    f"  python generate_problem.py --task_level module"
                )
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    record["system"] = system
                    data.append(record)

        self.dataset = DatasetDict({"test": Dataset.from_list(data)})
        self.dataset_no_image = self.dataset
        eval_logger.info(f"[RealBench] Loaded {len(data)} module-level problems.")


class RealBenchSystemTask(ConfigurableMessagesTask):
    """System-level Verilog code generation benchmark."""

    def __init__(self, config=None):
        task_config = {
            "task": "realbench_system",
            "output_type": "generate_until",
            "test_split": "test",
            "doc_to_text": realbench_doc_to_text,
            "doc_to_visual": realbench_doc_to_visual,
            "doc_to_target": realbench_doc_to_target,
            "process_results": realbench_system_process_results,
            "generation_kwargs": {**_BASE_GEN_KWARGS, "max_new_tokens": 8192},
            "metric_list": list(_METRIC_LIST),
            "metadata": {"version": 0},
        }
        if config:
            task_config.update({k: v for k, v in config.items() if k != "class"})
        super().__init__(config=task_config)

    def download(self, dataset_kwargs=None):
        benchmark_path = _get_benchmark_path()
        jsonl_path = os.path.join(benchmark_path, "problems", "system", "problems.jsonl")
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(
                f"Problems file not found: {jsonl_path}\n"
                f"Run the following in the RealBench directory first:\n"
                f"  make decrypt && make clean_encrypt\n"
                f"  python generate_problem.py --task_level system"
            )
        data = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))

        self.dataset = DatasetDict({"test": Dataset.from_list(data)})
        self.dataset_no_image = self.dataset
        eval_logger.info(f"[RealBench] Loaded {len(data)} system-level problems.")
