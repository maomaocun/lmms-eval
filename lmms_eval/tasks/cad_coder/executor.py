"""
Per-problem scoring for CAD-Coder.

Mirrors upstream's two-stage eval (`scripts/generate_model_cad.py` + `scripts/compute_iou.py`):
  1. Strip ```python fences from model output.
  2. Compile-by-execution: run the script in a subprocess. If it errors,
     valid_syntax=0.
  3. Append CadQuery STEP export, run again. If STEP file appears,
     valid_step=1.
  4. Run gold cadquery → gold STEP (cached on disk by deepcad_id).
  5. Compute IoU via `cq_align_shapes` (center-of-mass + principal axes
     alignment, intersect.Volume / union.Volume across 4 reflection
     candidates).

The gold STEP cache lives under `LMMS_CADCODER_CACHE` (default
`~/.cache/lmms_eval/cad_coder/gold_steps/<deepcad_id>.step`) so a single
run only pays gold-CAD generation cost once.

The runtime needs `cadquery` installed (heavy: brings in OpenCASCADE).
The Colab notebook is the intended runtime; pip install in the setup cell.
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

_FENCE_RE = re.compile(r"```(?:python|py|cadquery)?\s*\n(.*?)```", re.DOTALL)
_SPECIAL_TOKENS = ("<|im_end|>", "<|EOT|>", "<|endoftext|>")


def extract_code(raw: str) -> str:
    if not raw:
        return ""
    m = _FENCE_RE.search(raw)
    code = m.group(1) if m else raw
    for tok in _SPECIAL_TOKENS:
        code = code.replace(tok, "")
    return code.strip()


# ---- sandbox / subprocess --------------------------------------------------

def _sandbox_mode() -> str:
    return (os.environ.get("LMMS_CADCODER_SANDBOX") or "none").lower()


def _docker_image() -> str:
    return os.environ.get(
        "LMMS_CADCODER_DOCKER_IMAGE",
        "cadquery/cadquery:latest",
    )


def _docker_extra_args() -> list[str]:
    extra = os.environ.get("LMMS_CADCODER_DOCKER_EXTRA_ARGS", "").strip()
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


def _run_docker(script_path: str, payload_path: str, gold_dir: str,
                timeout: float) -> RunResult:
    image = _docker_image()
    mem = os.environ.get("LMMS_CADCODER_DOCKER_MEM", "4g")
    cpus = os.environ.get("LMMS_CADCODER_DOCKER_CPUS", "2")

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
        "-v", f"{gold_dir}:/work/gold:rw",
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


def _run(script: str, payload: dict, gold_dir: str, *,
         timeout: float, python_bin: str | None) -> RunResult:
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
            return _run_docker(script_path, payload_path, gold_dir, timeout=timeout)
        env = os.environ.copy()
        env["LMMS_CADCODER_PAYLOAD"] = payload_path
        return _run_bare(python_bin, script_path, env, timeout)
    finally:
        for p in (script_path, payload_path):
            try:
                os.unlink(p)
            except OSError:
                pass


# ---- runner script ---------------------------------------------------------

# Self-contained: takes (model_code, gold_code, gold_step_path), executes
# both, computes IoU between the resulting solids using upstream's alignment
# logic. Emits one tagged JSON line on stdout.
_OUT_TAG = "<<<CC_RESULT>>>"

_RUNNER_SOURCE = textwrap.dedent('''
    import json, os, sys, subprocess, tempfile, traceback

    OUT_TAG = "<<<CC_RESULT>>>"

    def _emit(payload):
        print(OUT_TAG + json.dumps(payload))

    def _read_payload():
        if len(sys.argv) >= 2:
            return json.load(open(sys.argv[1], "r", encoding="utf-8"))
        path = os.environ.get("LMMS_CADCODER_PAYLOAD")
        if not path:
            raise SystemExit("payload missing")
        return json.load(open(path, "r", encoding="utf-8"))

    def _run_python(code, timeout):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                         delete=False, encoding="utf-8") as fh:
            fh.write(code)
            path = fh.name
        try:
            proc = subprocess.run(
                [sys.executable, path], capture_output=True, text=True,
                timeout=timeout, check=False,
            )
            return proc.returncode == 0, proc.stderr
        except subprocess.TimeoutExpired as e:
            return False, "timeout"
        finally:
            try: os.unlink(path)
            except OSError: pass

    def _ensure_gold_step(gold_code, gold_step_path, timeout):
        if os.path.exists(gold_step_path) and os.path.getsize(gold_step_path) > 0:
            return True, ""
        os.makedirs(os.path.dirname(gold_step_path), exist_ok=True)
        appended = (
            gold_code.rstrip()
            + "\\nimport cadquery as cq\\n"
            + f"cq.exporters.export(solid, {gold_step_path!r})\\n"
        )
        return _run_python(appended, timeout=timeout)

    def _gen_model_step(code, step_path, timeout):
        # Phase A: bare execution = valid_syntax check
        valid_syntax, syntax_err = _run_python(code, timeout=timeout)
        if not valid_syntax:
            return False, False, syntax_err
        # Phase B: append STEP export and re-run
        appended = (
            code.rstrip()
            + "\\nimport cadquery as cq\\n"
            + f"cq.exporters.export(solid, {step_path!r})\\n"
        )
        ok, err = _run_python(appended, timeout=timeout)
        valid_step = ok and os.path.exists(step_path) and os.path.getsize(step_path) > 0
        return True, valid_step, err

    def _compute_iou(model_step, gold_step):
        try:
            import numpy as np
            import cadquery as cq
        except ImportError as e:
            return None, f"missing dep: {e}"

        try:
            ms = cq.importers.importStep(model_step)
            gs = cq.importers.importStep(gold_step)
        except Exception as e:
            return 0.0, f"step import: {e}"

        try:
            c_s = cq.Shape.centerOfMass(ms.val())
            c_t = cq.Shape.centerOfMass(gs.val())
            I_s = np.array(cq.Shape.matrixOfInertia(ms.val()))
            I_t = np.array(cq.Shape.matrixOfInertia(gs.val()))
            v_s = cq.Shape.computeMass(ms.val())
            v_t = cq.Shape.computeMass(gs.val())
            _, V_s = np.linalg.eigh(I_s)
            _, V_t = np.linalg.eigh(I_t)
            I_p_s = np.linalg.eigvalsh(I_s)
            I_p_t = np.linalg.eigvalsh(I_t)
            s_s = float(np.sqrt(np.abs(I_p_s).sum() / v_s))
            s_t = float(np.sqrt(np.abs(I_p_t).sum() / v_t))
            n_s = ms.translate(-c_s).val().scale(1 / s_s)
            n_t = gs.translate(-c_t).val().scale(1 / s_t)

            Rs = np.zeros((4, 3, 3))
            Rs[0] = V_t @ V_s.T
            for i in range(3):
                alignment = 1 - 2 * np.array([i > 0, (i + 1) % 2, i % 3 <= 1])
                Rs[i + 1] = V_t @ (alignment[None, :] * V_s).T

            best_iou = 0.0
            for i in range(4):
                T = np.zeros((4, 4)); T[:3, :3] = Rs[i]; T[-1, -1] = 1
                aligned = n_s.transformGeometry(cq.Matrix(T.tolist()))
                try:
                    inter = aligned.intersect(n_t)
                    union = aligned.fuse(n_t)
                    iou = inter.Volume() / union.Volume()
                    if iou > best_iou:
                        best_iou = iou
                except Exception:
                    pass
            return float(best_iou), ""
        except Exception as e:
            return 0.0, f"align/iou: {e}"

    def main():
        p = _read_payload()
        model_code = p["model_code"]
        gold_code = p["gold_code"]
        gold_step = p["gold_step_path"]
        exec_timeout = float(p.get("exec_timeout", 60))
        skip_iou = bool(p.get("skip_iou", False))

        if not model_code.strip():
            _emit({"ok": True, "valid_syntax": 0.0, "valid_step": 0.0,
                   "iou": 0.0, "error": "empty code"})
            return

        with tempfile.TemporaryDirectory() as td:
            model_step = os.path.join(td, "model.step")
            valid_syntax, valid_step, exec_err = _gen_model_step(
                model_code, model_step, timeout=exec_timeout,
            )
            payload = {
                "ok": True,
                "valid_syntax": 1.0 if valid_syntax else 0.0,
                "valid_step": 1.0 if valid_step else 0.0,
                "iou": 0.0,
            }
            if exec_err:
                payload["exec_err_tail"] = exec_err[-400:]

            if not (valid_step and not skip_iou):
                _emit(payload)
                return

            gold_ok, gold_err = _ensure_gold_step(gold_code, gold_step,
                                                   timeout=exec_timeout)
            if not gold_ok:
                payload["error"] = "gold step gen failed"
                payload["gold_err_tail"] = (gold_err or "")[-400:]
                _emit(payload)
                return

            iou, iou_err = _compute_iou(model_step, gold_step)
            if iou is None:
                payload["error"] = iou_err
                _emit(payload)
                return
            payload["iou"] = float(iou)
            if iou_err:
                payload["iou_err_tail"] = iou_err[-200:]
            _emit(payload)

    try:
        main()
    except Exception as e:
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

def score_one(model_raw: str, gold_code: str, gold_step_path: str, *,
              exec_timeout: float = 60.0,
              total_timeout: float = 300.0,
              skip_iou: bool = False,
              python_bin: str | None = None) -> dict:
    code = extract_code(model_raw)
    payload = {
        "model_code": code,
        "gold_code": gold_code,
        "gold_step_path": gold_step_path,
        "exec_timeout": exec_timeout,
        "skip_iou": skip_iou,
    }
    gold_dir = os.path.dirname(gold_step_path) or "."
    os.makedirs(gold_dir, exist_ok=True)

    res = _run(_RUNNER_SOURCE, payload, gold_dir,
               timeout=total_timeout, python_bin=python_bin)

    if res.timed_out:
        return {"valid_syntax": 0.0, "valid_step": 0.0, "iou": 0.0,
                "error": "timeout", "stderr_tail": res.stderr[-400:]}

    out = _parse_runner_output(res.stdout)
    if out is None:
        return {"valid_syntax": 0.0, "valid_step": 0.0, "iou": 0.0,
                "error": "no result", "returncode": res.returncode,
                "stderr_tail": res.stderr[-400:]}

    if not out.get("ok"):
        return {"valid_syntax": 0.0, "valid_step": 0.0, "iou": 0.0,
                "error": out.get("error", "unknown"),
                "stage": out.get("stage")}

    return {
        "valid_syntax": float(out.get("valid_syntax", 0.0)),
        "valid_step": float(out.get("valid_step", 0.0)),
        "iou": float(out.get("iou", 0.0)),
        **({"error": out["error"]} if "error" in out else {}),
    }
