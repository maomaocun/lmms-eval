"""
Pure scoring module for TritonBench. No network. No state. No automatic exec on
import. Designed to be called from `process_results` (utils.py) at evaluation
time, which on this project happens inside a Colab runtime with CUDA + Triton.

Mirrors the upstream eval logic from
    https://github.com/thunlp/TritonBench/blob/main/EVAL/eval_G/0_call_acc.py
    https://github.com/thunlp/TritonBench/blob/main/EVAL/eval_G/1_exe_acc.py
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass

# Separator the upstream gold reference files use between (kernel + wrapper)
# and (test harness). We split on this to re-graft the model's code in front of
# the original test harness.
_SEP_RE = re.compile(r"^#{40,}\s*$", re.MULTILINE)

_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
_SPECIAL_TOKENS = ("<|im_end|>", "<|EOT|>", "<|endoftext|>")


def extract_code(raw: str) -> str:
    """Pull Python source out of a model response.

    Tries, in order: fenced ```python``` block, fenced ``` block, raw string.
    Then strips known chat / EOS tokens.
    """
    if not raw:
        return ""
    m = _FENCE_RE.search(raw)
    code = m.group(1) if m else raw
    for tok in _SPECIAL_TOKENS:
        code = code.replace(tok, "")
    return code.strip()


def normalize_code(code: str) -> str:
    """Keep only top-level imports and function/class definitions.

    Matches the cleanup the upstream `process_code` performs before grafting
    onto the test harness — drops stray prints, top-level test calls, etc.
    Falls back to the raw input if AST parsing fails.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign)):
            keep.append(node)
    if not keep:
        return code
    module = ast.Module(body=keep, type_ignores=[])
    try:
        return ast.unparse(module)
    except Exception:
        return code


def split_gold_reference(gold_src: str) -> tuple[str, str]:
    """Return (kernel_and_wrapper, test_harness) from a gold reference file.

    Splits at the first long `###...` separator. If no separator is present,
    returns ("", gold_src) so the whole file is treated as harness — call
    accuracy will then degrade to "does the model's code run alongside the
    full gold file" but we won't crash.
    """
    parts = _SEP_RE.split(gold_src, maxsplit=1)
    if len(parts) == 2:
        return parts[0].rstrip() + "\n", parts[1].lstrip()
    return "", gold_src


def build_eval_script(model_code: str, test_harness: str) -> str:
    """Compose the script that will be executed for call/exec accuracy."""
    return "# === model-generated code ===\n" + normalize_code(model_code).rstrip() + "\n\n# === gold test harness ===\n" + test_harness.rstrip() + "\n"


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool


def run_script(script: str, *, timeout: float = 120.0, python_bin: str | None = None, cuda_visible_devices: str | None = None) -> RunResult:
    """Run `script` as a Python subprocess and capture stdout/stderr.

    The harness is dropped into a temp file. CUDA_VISIBLE_DEVICES and the
    interpreter path are configurable for the Colab caller.
    """
    python_bin = python_bin or sys.executable
    env = os.environ.copy()
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as fh:
        fh.write(script)
        path = fh.name
    try:
        proc = subprocess.run(
            [python_bin, path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
        return RunResult(proc.returncode, proc.stdout, proc.stderr, False)
    except subprocess.TimeoutExpired as e:
        return RunResult(-1, e.stdout or "", e.stderr or "", True)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def score_one(model_raw: str, gold_test_src: str, *, gold_stdout: str | None = None, timeout: float = 120.0, python_bin: str | None = None, cuda_visible_devices: str | None = None) -> dict:
    """Compute call_acc and exec_acc for a single (prediction, gold) pair.

    If `gold_stdout` is supplied, the gold script is not re-run. Otherwise
    the gold reference is executed as-is to capture its stdout.
    """
    code = extract_code(model_raw)
    _, harness = split_gold_reference(gold_test_src)
    eval_script = build_eval_script(code, harness)

    pred_run = run_script(
        eval_script,
        timeout=timeout,
        python_bin=python_bin,
        cuda_visible_devices=cuda_visible_devices,
    )
    call_pass = (pred_run.returncode == 0) and not pred_run.timed_out

    if not call_pass:
        return {
            "call_acc": 0.0,
            "exec_acc": 0.0,
            "pred_returncode": pred_run.returncode,
            "pred_timed_out": pred_run.timed_out,
            "pred_stderr_tail": pred_run.stderr[-500:],
        }

    if gold_stdout is None:
        gold_run = run_script(
            gold_test_src,
            timeout=timeout,
            python_bin=python_bin,
            cuda_visible_devices=cuda_visible_devices,
        )
        if gold_run.returncode != 0 or gold_run.timed_out:
            # Gold itself didn't run cleanly in this environment — can't
            # meaningfully compare. Report call pass but no exec verdict.
            return {
                "call_acc": 1.0,
                "exec_acc": 0.0,
                "exec_skipped": True,
                "gold_returncode": gold_run.returncode,
            }
        gold_stdout = gold_run.stdout

    exec_pass = pred_run.stdout == gold_stdout
    return {
        "call_acc": 1.0,
        "exec_acc": 1.0 if exec_pass else 0.0,
    }
