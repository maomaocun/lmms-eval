"""
Per-problem scoring for the KernelBench tasks.

Defers actual GPU execution to upstream's `kernelbench.eval.eval_kernel_against_ref`,
invoked in a subprocess (optionally inside a one-shot Docker sandbox).

Why subprocess: each evaluation compiles and runs CUDA / Triton kernels that
can crash, hang, or leak GPU memory; isolating each one keeps a single bad
generation from taking down the whole eval run.

The Colab notebook (project's intended runtime) is expected to have
`kernelbench` installed:

    pip install "git+https://github.com/ScalingIntelligence/KernelBench.git"

If the upstream package is missing, the runner script writes a structured error
to its JSON output, which `score_one` surfaces as a 0 across all metrics with
an `error` field.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass

_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
_SPECIAL_TOKENS = ("<|im_end|>", "<|EOT|>", "<|endoftext|>")


def extract_code(raw: str) -> str:
    """Pull Python source out of a model response. Same logic as tritonbench."""
    if not raw:
        return ""
    m = _FENCE_RE.search(raw)
    code = m.group(1) if m else raw
    for tok in _SPECIAL_TOKENS:
        code = code.replace(tok, "")
    return code.strip()


# ---- sandbox / subprocess --------------------------------------------------


def _sandbox_mode() -> str:
    return (os.environ.get("LMMS_KERNELBENCH_SANDBOX") or "none").lower()


def _docker_image() -> str:
    return os.environ.get(
        "LMMS_KERNELBENCH_DOCKER_IMAGE",
        "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime",
    )


def _docker_extra_args() -> list[str]:
    extra = os.environ.get("LMMS_KERNELBENCH_DOCKER_EXTRA_ARGS", "").strip()
    return extra.split() if extra else []


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool


def _run_bare(python_bin: str, script_path: str, env: dict, timeout: float) -> RunResult:
    try:
        proc = subprocess.run(
            [python_bin, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
        return RunResult(proc.returncode, proc.stdout, proc.stderr, False)
    except subprocess.TimeoutExpired as e:
        return RunResult(-1, e.stdout or "", e.stderr or "", True)


def _run_docker(script_path: str, timeout: float, cuda_visible_devices: str | None) -> RunResult:
    """One-shot container: no network, read-only rootfs, /tmp tmpfs, gpu forwarded."""
    image = _docker_image()
    mem = os.environ.get("LMMS_KERNELBENCH_DOCKER_MEM", "16g")
    cpus = os.environ.get("LMMS_KERNELBENCH_DOCKER_CPUS", "4")
    gpus = os.environ.get("LMMS_KERNELBENCH_DOCKER_GPUS", "all")

    cmd = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,size=2g,exec",
        "--memory",
        mem,
        "--cpus",
        cpus,
        "-e",
        "HOME=/tmp",
        "-e",
        "TRITON_CACHE_DIR=/tmp/.triton",
        "-e",
        "TORCH_INDUCTOR_CACHE_DIR=/tmp/.inductor",
        "-v",
        f"{script_path}:/work/eval.py:ro",
        "-w",
        "/work",
    ]
    if gpus and gpus.lower() != "none":
        cmd += ["--gpus", gpus]
    if cuda_visible_devices is not None:
        cmd += ["-e", f"CUDA_VISIBLE_DEVICES={cuda_visible_devices}"]
    cmd += _docker_extra_args()
    cmd += [image, "python", "/work/eval.py"]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return RunResult(proc.returncode, proc.stdout, proc.stderr, False)
    except subprocess.TimeoutExpired as e:
        return RunResult(-1, e.stdout or "", e.stderr or "", True)
    except FileNotFoundError as e:
        return RunResult(127, "", f"docker not available: {e}", False)


def _run_script(script: str, *, timeout: float, python_bin: str | None, cuda_visible_devices: str | None) -> RunResult:
    python_bin = python_bin or sys.executable
    mode = _sandbox_mode()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as fh:
        fh.write(script)
        path = fh.name
    try:
        if mode == "docker":
            return _run_docker(path, timeout=timeout, cuda_visible_devices=cuda_visible_devices)
        env = os.environ.copy()
        if cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        return _run_bare(python_bin, path, env, timeout)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# ---- runner script ---------------------------------------------------------

# This Python source is what gets dropped into the temp file and executed in
# the sandbox / subprocess. It loads upstream's eval helper, runs it, and
# prints a single JSON line on stdout that the parent process parses.
_RUNNER_TEMPLATE = textwrap.dedent(
    """
    import json, os, sys, traceback

    OUT_TAG = {OUT_TAG!r}

    def _emit(payload):
        print(OUT_TAG + json.dumps(payload))

    REF_SRC = {ref_src!r}
    GEN_SRC = {gen_src!r}
    NUM_CORRECT = {num_correct}
    NUM_PERF = {num_perf}
    BACKEND = {backend!r}

    try:
        from kernelbench.eval import eval_kernel_against_ref
    except Exception as e:
        _emit({{
            "ok": False,
            "stage": "import",
            "error": f"{{type(e).__name__}}: {{e}}",
            "hint": "pip install git+https://github.com/ScalingIntelligence/KernelBench.git",
        }})
        sys.exit(0)

    try:
        result = eval_kernel_against_ref(
            original_model_src=REF_SRC,
            custom_model_src=GEN_SRC,
            num_correct_trials=NUM_CORRECT,
            num_perf_trials=NUM_PERF,
            measure_performance=True,
            backend=BACKEND,
            verbose=False,
        )
    except Exception as e:
        _emit({{
            "ok": False,
            "stage": "eval",
            "error": f"{{type(e).__name__}}: {{e}}",
            "trace": traceback.format_exc()[-800:],
        }})
        sys.exit(0)

    payload = {{
        "ok": True,
        "compiled": bool(getattr(result, "compiled", False)),
        "correctness": bool(getattr(result, "correctness", False)),
        "runtime": float(getattr(result, "runtime", -1.0)),
        "ref_runtime": float(getattr(result, "ref_runtime", -1.0)),
        "metadata": dict(getattr(result, "metadata", {{}})),
    }}
    _emit(payload)
"""
).lstrip()


_OUT_TAG = "<<<KB_RESULT>>>"


def _build_runner(ref_src: str, gen_src: str, *, num_correct: int, num_perf: int, backend: str) -> str:
    return _RUNNER_TEMPLATE.format(
        OUT_TAG=_OUT_TAG,
        ref_src=ref_src,
        gen_src=gen_src,
        num_correct=num_correct,
        num_perf=num_perf,
        backend=backend,
    )


def _parse_runner_output(stdout: str) -> dict | None:
    """Find the tagged JSON line emitted by the runner; tolerate prologue logs."""
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith(_OUT_TAG):
            try:
                return json.loads(line[len(_OUT_TAG) :])
            except json.JSONDecodeError:
                return None
    return None


# ---- public entry ----------------------------------------------------------


def score_one(reference_src: str, model_raw: str, *, num_correct: int = 5, num_perf: int = 100, backend: str = "cuda", timeout: float = 300.0, python_bin: str | None = None, cuda_visible_devices: str | None = None) -> dict:
    """Score one (reference, model_response) pair.

    Returns a dict with float metric values for `compiled`, `correctness`,
    `fast_1`, `fast_2`, plus diagnostic fields. All values are 0/1.
    """
    gen_src = extract_code(model_raw)
    if not gen_src:
        return {"compiled": 0.0, "correctness": 0.0, "fast_1": 0.0, "fast_2": 0.0, "error": "empty generated code"}

    script = _build_runner(reference_src, gen_src, num_correct=num_correct, num_perf=num_perf, backend=backend)
    res = _run_script(script, timeout=timeout, python_bin=python_bin, cuda_visible_devices=cuda_visible_devices)

    if res.timed_out:
        return {"compiled": 0.0, "correctness": 0.0, "fast_1": 0.0, "fast_2": 0.0, "error": "timeout", "stderr_tail": res.stderr[-400:]}

    payload = _parse_runner_output(res.stdout)
    if payload is None:
        return {"compiled": 0.0, "correctness": 0.0, "fast_1": 0.0, "fast_2": 0.0, "error": "no result", "returncode": res.returncode, "stderr_tail": res.stderr[-400:]}

    if not payload.get("ok"):
        return {"compiled": 0.0, "correctness": 0.0, "fast_1": 0.0, "fast_2": 0.0, "error": payload.get("error", "unknown"), "stage": payload.get("stage"), "hint": payload.get("hint")}

    compiled = 1.0 if payload.get("compiled") else 0.0
    correct = 1.0 if payload.get("correctness") else 0.0
    runtime = payload.get("runtime") or -1.0
    ref_rt = payload.get("ref_runtime") or -1.0
    speedup = (ref_rt / runtime) if (correct and runtime > 0 and ref_rt > 0) else 0.0
    fast_1 = 1.0 if (correct and speedup > 1.0) else 0.0
    fast_2 = 1.0 if (correct and speedup >= 2.0) else 0.0

    return {
        "compiled": compiled,
        "correctness": correct,
        "fast_1": fast_1,
        "fast_2": fast_2,
        "runtime": runtime,
        "ref_runtime": ref_rt,
        "speedup": speedup,
    }
