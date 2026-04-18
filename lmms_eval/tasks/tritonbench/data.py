"""
Lazy data helpers for the TritonBench tasks.

The benchmark records (instruction + gold output) are loaded by HuggingFace
datasets directly from the upstream raw URLs (see the YAML configs). Each
problem also has a separate gold reference Python file that contains the test
harness — we fetch those on demand from upstream and cache them on disk so
`process_results` doesn't pay HTTP cost per item per run.

Cache layout (override with env var `LMMS_TRITONBENCH_CACHE`):

    <cache>/refs/G/<filename>.py
    <cache>/refs/T/<filename>.py
"""
from __future__ import annotations

import io
import os
import sys
import tarfile
import threading
import urllib.request
from pathlib import Path

UPSTREAM_REPO = "thunlp/TritonBench"
UPSTREAM_BRANCH = "main"
UPSTREAM_TARBALL = (
    f"https://github.com/{UPSTREAM_REPO}/archive/refs/heads/{UPSTREAM_BRANCH}.tar.gz"
)
UPSTREAM_RAW = (
    f"https://raw.githubusercontent.com/{UPSTREAM_REPO}/{UPSTREAM_BRANCH}"
)

_REFS_DIR_IN_TAR = {
    "G": "data/TritonBench_G_v1/",
    "T": "data/TritonBench_T_v1/",
}

_lock = threading.Lock()


def cache_root() -> Path:
    override = os.environ.get("LMMS_TRITONBENCH_CACHE")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".cache" / "lmms_eval" / "tritonbench"


def _refs_dir(track: str) -> Path:
    return cache_root() / "refs" / track


def _download_tarball() -> bytes:
    print(f"[tritonbench] fetching {UPSTREAM_TARBALL}", file=sys.stderr)
    with urllib.request.urlopen(UPSTREAM_TARBALL) as r:
        return r.read()


def _populate_refs_from_tarball() -> None:
    """One-shot: download the upstream tarball and extract every gold reference
    .py file for both tracks into the cache. Cheaper than fetching 368 files
    individually, and gives a complete cache so subsequent runs are offline."""
    raw = _download_tarball()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        members = tar.getnames()
        if not members:
            raise RuntimeError("empty tritonbench tarball")
        root_prefix = members[0].split("/", 1)[0]

        for track, ref_subdir in _REFS_DIR_IN_TAR.items():
            out_dir = _refs_dir(track)
            out_dir.mkdir(parents=True, exist_ok=True)
            prefix = f"{root_prefix}/{ref_subdir}"
            extracted = 0
            for m in tar.getmembers():
                if not m.isfile() or not m.name.startswith(prefix):
                    continue
                if not m.name.endswith(".py"):
                    continue
                fh = tar.extractfile(m)
                if fh is None:
                    continue
                target = out_dir / Path(m.name).name
                target.write_bytes(fh.read())
                extracted += 1
            print(f"[tritonbench] cached {extracted} refs for track {track} -> {out_dir}",
                  file=sys.stderr)


def ensure_refs_cached() -> None:
    """Idempotent: populate the local cache if either track's refs dir is empty."""
    with _lock:
        for track in _REFS_DIR_IN_TAR:
            d = _refs_dir(track)
            if not d.exists() or not any(d.glob("*.py")):
                _populate_refs_from_tarball()
                return


def gold_test_src(track: str, file_name: str) -> str:
    """Read the gold reference / test harness source for one problem.

    Falls back to a single-file HTTP fetch if the cache is missing this entry
    (e.g. the upstream added a problem after we cached the tarball)."""
    ensure_refs_cached()
    p = _refs_dir(track) / file_name
    if p.exists():
        return p.read_text(encoding="utf-8")
    # Last-resort single-file fetch.
    url = f"{UPSTREAM_RAW}/{_REFS_DIR_IN_TAR[track]}{file_name}"
    print(f"[tritonbench] cache miss, fetching {url}", file=sys.stderr)
    with urllib.request.urlopen(url) as r:
        body = r.read().decode("utf-8")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return body
