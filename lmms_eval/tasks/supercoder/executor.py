"""
Per-problem scoring for SuperCoder (assembly superoptimization).

Mirrors upstream `src/test_benchmark.py`:
  1. Strip ```assembly fences from model output.
  2. Compile model assembly to a binary with `gcc`.
  3. For each (input, expected_output) test case:
       a. Run binary, piping input to stdin.
       b. Compare stdout to expected (exact equality).
  4. If correct, run `hyperfine` on (model_bin, baseline_bin) with the same
     input as stdin; speedup = baseline_mean / model_mean (clamped >= 1.0).

Per the upstream convention, both `gcc` and `hyperfine` must be installed on
the host. The Colab notebook (intended runtime) will install them.

Sandbox-safe: scoring runs entirely inside a single subprocess script, so the
Docker mode wraps the whole compile+run+benchmark pipeline.
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

_FENCE_RE = re.compile(r"```(?:assembly|asm|x86|x86-64|nasm|gas)?\s*\n(.*?)```",
                       re.DOTALL | re.IGNORECASE)


def extract_assembly(raw: str) -> str:
    """Pull assembly source from a model response."""
    if not raw:
        return ""
    m = _FENCE_RE.search(raw)
    body = m.group(1) if m else raw
    body = body.replace("```assembly\n", "").replace("```", "")
    return body.strip()


# ---- sandbox / subprocess --------------------------------------------------

def _sandbox_mode() -> str:
    return (os.environ.get("LMMS_SUPERCODER_SANDBOX") or "none").lower()


def _docker_image() -> str:
    return os.environ.get(
        "LMMS_SUPERCODER_DOCKER_IMAGE",
        # Default targets x86_64 Linux with gcc + hyperfine pre-installable.
        "gcc:13",
    )


def _docker_extra_args() -> list[str]:
    extra = os.environ.get("LMMS_SUPERCODER_DOCKER_EXTRA_ARGS", "").strip()
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
            capture_output=True, text=True, timeout=timeout, env=env, check=False,
        )
        return RunResult(proc.returncode, proc.stdout, proc.stderr, False)
    except subprocess.TimeoutExpired as e:
        return RunResult(-1, e.stdout or "", e.stderr or "", True)


def _run_docker(script_path: str, payload_path: str, timeout: float) -> RunResult:
    """One-shot container with gcc + hyperfine."""
    image = _docker_image()
    mem = os.environ.get("LMMS_SUPERCODER_DOCKER_MEM", "4g")
    cpus = os.environ.get("LMMS_SUPERCODER_DOCKER_CPUS", "2")

    cmd = [
        "docker", "run", "--rm",
        "--network", "none",
        "--read-only",
        "--tmpfs", "/tmp:rw,size=512m,exec",
        "--memory", mem,
        "--cpus", cpus,
        "-e", "HOME=/tmp",
        "-v", f"{script_path}:/work/run.py:ro",
        "-v", f"{payload_path}:/work/payload.json:ro",
        "-w", "/work",
    ]
    cmd += _docker_extra_args()
    cmd += [image, "python3", "/work/run.py", "/work/payload.json"]

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False,
        )
        return RunResult(proc.returncode, proc.stdout, proc.stderr, False)
    except subprocess.TimeoutExpired as e:
        return RunResult(-1, e.stdout or "", e.stderr or "", True)
    except FileNotFoundError as e:
        return RunResult(127, "", f"docker not available: {e}", False)


def _run(script: str, payload: dict, *, timeout: float,
         python_bin: str | None) -> RunResult:
    python_bin = python_bin or sys.executable
    mode = _sandbox_mode()

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(script)
        script_path = fh.name
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as fh:
        json.dump(payload, fh)
        payload_path = fh.name

    try:
        if mode == "docker":
            return _run_docker(script_path, payload_path, timeout=timeout)
        env = os.environ.copy()
        env["LMMS_SUPERCODER_PAYLOAD"] = payload_path
        return _run_bare(python_bin, script_path, env, timeout)
    finally:
        for p in (script_path, payload_path):
            try:
                os.unlink(p)
            except OSError:
                pass


# ---- runner script ---------------------------------------------------------

# Self-contained: compiles model_asm + baseline_asm with gcc, runs each test
# case via subprocess piping input to stdin, then hyperfines passing cases.
# Emits a single JSON line on stdout prefixed with the tag below; parent finds
# it amid any compiler chatter.
_OUT_TAG = "<<<SC_RESULT>>>"

_RUNNER_SOURCE = textwrap.dedent('''
    import json, os, sys, subprocess, tempfile, shutil

    OUT_TAG = "<<<SC_RESULT>>>"

    def _emit(payload):
        print(OUT_TAG + json.dumps(payload))

    def _read_payload():
        if len(sys.argv) >= 2:
            return json.load(open(sys.argv[1], "r", encoding="utf-8"))
        path = os.environ.get("LMMS_SUPERCODER_PAYLOAD")
        if not path:
            raise SystemExit("payload path missing")
        return json.load(open(path, "r", encoding="utf-8"))

    def _have(cmd):
        return shutil.which(cmd) is not None

    def _compile(asm_src, out_bin):
        asm_path = out_bin + ".s"
        open(asm_path, "w", encoding="utf-8").write(asm_src)
        proc = subprocess.run(
            ["gcc", asm_path, "-o", out_bin, "-no-pie"],
            capture_output=True, text=True, timeout=60, check=False,
        )
        return proc.returncode == 0, proc.stderr

    def _run_with_stdin(bin_path, stdin_text, timeout=30):
        try:
            proc = subprocess.run(
                [bin_path], input=stdin_text, capture_output=True,
                text=True, timeout=timeout, check=False,
            )
            return proc.returncode, proc.stdout, proc.stderr, False
        except subprocess.TimeoutExpired as e:
            return -1, e.stdout or "", e.stderr or "", True

    def _hyperfine(bin_path, stdin_text, timeout=60):
        if not _have("hyperfine"):
            return None
        with tempfile.TemporaryDirectory() as td:
            in_path = os.path.join(td, "in.txt")
            out_path = os.path.join(td, "out.json")
            open(in_path, "w", encoding="utf-8").write(stdin_text)
            cmd = [
                "hyperfine", "--warmup", "3", "--runs", "10",
                "--input", in_path, "--export-json", out_path,
                "--time-unit", "millisecond", bin_path,
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True,
                                      timeout=timeout, check=False)
            except subprocess.TimeoutExpired:
                return None
            if proc.returncode != 0 or not os.path.exists(out_path):
                return None
            data = json.load(open(out_path))
            for r in data.get("results", []):
                return {"mean": r.get("mean"), "median": r.get("median")}
        return None

    def main():
        p = _read_payload()
        model_asm = p["model_asm"]
        baseline_asm = p["baseline_asm"]
        inputs = p["inputs"]
        outputs = p["outputs"]
        max_cases = p.get("max_cases", 10)
        run_timeout = p.get("run_timeout", 30)
        bench_timeout = p.get("bench_timeout", 60)

        if not model_asm.strip():
            _emit({"ok": False, "stage": "extract", "error": "empty assembly"})
            return

        if not _have("gcc"):
            _emit({"ok": False, "stage": "tooling",
                   "error": "gcc not on PATH",
                   "hint": "install gcc (and hyperfine for speedup) in the runtime"})
            return

        with tempfile.TemporaryDirectory() as td:
            model_bin = os.path.join(td, "model")
            base_bin = os.path.join(td, "base")

            ok, stderr = _compile(model_asm, model_bin)
            if not ok:
                _emit({"ok": True, "compiled": False, "correctness": 0.0,
                       "speedup": 1.0, "fast_1": 0.0, "n_cases": 0,
                       "n_passed": 0, "stderr_tail": stderr[-500:]})
                return

            base_ok, base_stderr = _compile(baseline_asm, base_bin)
            if not base_ok:
                _emit({"ok": False, "stage": "baseline_compile",
                       "error": "baseline gcc failed",
                       "stderr_tail": base_stderr[-500:]})
                return

            cases = list(zip(inputs[:max_cases], outputs[:max_cases]))
            n_cases = len(cases)
            n_passed = 0
            speedups = []
            for stdin_text, expected in cases:
                rc, out, err, timed_out = _run_with_stdin(model_bin, stdin_text,
                                                           timeout=run_timeout)
                if timed_out or rc != 0 or out != expected:
                    continue
                n_passed += 1
                m = _hyperfine(model_bin, stdin_text, timeout=bench_timeout)
                b = _hyperfine(base_bin, stdin_text, timeout=bench_timeout)
                if m and b and (m["mean"] or 0) > 0:
                    sp = b["mean"] / m["mean"]
                    speedups.append(max(sp, 1.0))
                else:
                    speedups.append(1.0)

            correctness = (n_passed / n_cases) if n_cases else 0.0
            mean_sp = (sum(speedups) / len(speedups)) if speedups else 1.0
            fast_1 = 1.0 if (correctness >= 1.0 - 1e-9 and mean_sp > 1.0) else 0.0

            _emit({
                "ok": True,
                "compiled": True,
                "correctness": correctness,
                "speedup": mean_sp,
                "fast_1": fast_1,
                "n_cases": n_cases,
                "n_passed": n_passed,
            })

    try:
        main()
    except Exception as e:
        import traceback
        _emit({"ok": False, "stage": "runner",
               "error": f"{type(e).__name__}: {e}",
               "trace": traceback.format_exc()[-800:]})
''').lstrip()


def _parse_runner_output(stdout: str) -> dict | None:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith(_OUT_TAG):
            try:
                return json.loads(line[len(_OUT_TAG):])
            except json.JSONDecodeError:
                return None
    return None


# ---- public entry ----------------------------------------------------------

def score_one(model_raw: str, baseline_asm: str,
              inputs: list[str], outputs: list[str], *,
              max_cases: int = 10,
              run_timeout: float = 30.0,
              bench_timeout: float = 60.0,
              total_timeout: float = 600.0,
              python_bin: str | None = None) -> dict:
    """Score one (model_response, baseline) pair against test cases."""
    model_asm = extract_assembly(model_raw)
    if not model_asm:
        return {"compiled": 0.0, "correctness": 0.0, "speedup": 1.0,
                "fast_1": 0.0, "error": "empty generated assembly"}

    payload = {
        "model_asm": model_asm,
        "baseline_asm": _strip_fences(baseline_asm),
        "inputs": list(inputs),
        "outputs": list(outputs),
        "max_cases": max_cases,
        "run_timeout": run_timeout,
        "bench_timeout": bench_timeout,
    }

    res = _run(_RUNNER_SOURCE, payload, timeout=total_timeout,
               python_bin=python_bin)

    if res.timed_out:
        return {"compiled": 0.0, "correctness": 0.0, "speedup": 1.0,
                "fast_1": 0.0, "error": "timeout",
                "stderr_tail": res.stderr[-400:]}

    payload_out = _parse_runner_output(res.stdout)
    if payload_out is None:
        return {"compiled": 0.0, "correctness": 0.0, "speedup": 1.0,
                "fast_1": 0.0, "error": "no result",
                "returncode": res.returncode,
                "stderr_tail": res.stderr[-400:]}

    if not payload_out.get("ok"):
        return {"compiled": 0.0, "correctness": 0.0, "speedup": 1.0,
                "fast_1": 0.0, "error": payload_out.get("error", "unknown"),
                "stage": payload_out.get("stage"),
                "hint": payload_out.get("hint")}

    return {
        "compiled": 1.0 if payload_out.get("compiled") else 0.0,
        "correctness": float(payload_out.get("correctness", 0.0)),
        "speedup": float(payload_out.get("speedup", 1.0)),
        "fast_1": float(payload_out.get("fast_1", 0.0)),
        "n_cases": payload_out.get("n_cases", 0),
        "n_passed": payload_out.get("n_passed", 0),
    }


def _strip_fences(text: str) -> str:
    """Remove ```assembly fences if present (baseline asm is also fenced)."""
    if not text:
        return ""
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.replace("```assembly\n", "").replace("```", "").strip()
