"""Immediate function-level coverage of arbitrary inputs.

The edge validator (validator.py) answers "did this input add an edge the
campaign doesn't have?". That is necessary but not sufficient for a DIRECTED
hunt: we asked the LLM to reach a SPECIFIC unreached function, so we must check
whether that function is now executed — right now, before injecting anything.

This module runs the OSS-Fuzz clang-coverage build (a libFuzzer binary
instrumented with llvm source-based coverage) on a set of candidate files and
returns the set of function names that executed. It is the ground truth for
"did we actually reach the target code?".

How it works:
  1. run `/out/<fuzzer> <cand1> <cand2> ...` with LLVM_PROFILE_FILE set, which
     executes each input once and writes a .profraw;
  2. inside the OSS-Fuzz base-runner (the host LLVM is usually too old for the
     profraw format) merge with `llvm-profdata` and export with `llvm-cov`;
  3. parse the export and return covered function names.

Everything is config-driven (FUZZ_COVERAGE_*), so it is target-agnostic.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .config import CONFIG


class CoverageReplayError(RuntimeError):
    pass


def _binary_path() -> Path:
    """Host path of the coverage-instrumented fuzzer binary."""
    cov_dir = CONFIG.coverage_dir
    if not cov_dir:
        raise CoverageReplayError(
            "FUZZ_COVERAGE_DIR is not set — function-level replay unavailable")
    binary = Path(cov_dir) / CONFIG.coverage_bin
    if not binary.is_file():
        raise CoverageReplayError(f"coverage binary not found: {binary}")
    return binary


def available() -> bool:
    """True when function-level replay is configured and the binary exists."""
    if not CONFIG.coverage_dir or not CONFIG.coverage_bin:
        return False
    try:
        return (Path(CONFIG.coverage_dir) / CONFIG.coverage_bin).is_file()
    except OSError:
        return False


def run_coverage(input_paths: list[str]) -> set[str]:
    """Run the coverage build on `input_paths`; return covered function names.

    Executes each input once in ONE process (so the profraw holds the union of
    all of them), then digests it. Raises CoverageReplayError on failure.
    """
    if not input_paths:
        return set()
    binary = _binary_path()
    cov_dir = Path(CONFIG.coverage_dir).resolve()
    bin_name = CONFIG.coverage_bin
    image = CONFIG.coverage_image or "gcr.io/oss-fuzz-base/base-runner"

    work = Path(tempfile.mkdtemp(prefix="covr_", dir="/tmp/opencode" if
                                 os.path.isdir("/tmp/opencode") else None))
    try:
        # stage candidates into the work dir with safe names (host)
        staged: list[str] = []
        for i, src in enumerate(input_paths):
            dst = work / f"in_{i:04d}.bin"
            try:
                shutil.copy2(src, dst)
            except OSError:
                continue
            staged.append(f"/work/{dst.name}")
        if not staged:
            return set()
        (work / "run.profraw").unlink(missing_ok=True)

        # 1) execute all inputs once, capturing a single profraw
        run_cmd = [
            "docker", "run", "--rm",
            "-e", "LLVM_PROFILE_FILE=/work/run.profraw",
            "-e", "ASAN_OPTIONS=detect_leaks=0:symbolize=0",
            "-v", f"{cov_dir}:/out",
            "-v", f"{work}:/work",
            image, "bash", "-c",
            f"cd /out && /out/{bin_name} " + " ".join(staged),
        ]
        proc = subprocess.run(run_cmd, capture_output=True, text=True,
                             timeout=CONFIG.coverage_timeout)
        # libFuzzer exits non-zero sometimes even on success; rely on profraw
        if not (work / "run.profraw").is_file():
            raise CoverageReplayError(
                "coverage run produced no profraw: "
                + (proc.stderr or proc.stdout or "")[-300:])

        # 2) digest: merge profraw + export function coverage
        inner = (
            "cd /out && "
            "llvm-profdata merge -sparse /work/run.profraw -o /work/run.profdata && "
            f"shared=$(coverage_helper shared_libs -build-dir=/out -object=/out/{bin_name} || true) && "
            f"llvm-cov export -instr-profile=/work/run.profdata -object=/out/{bin_name} "
            "$shared > /work/export.json"
        )
        proc = subprocess.run(
            ["docker", "run", "--rm",
             "-v", f"{cov_dir}:/out",
             "-v", f"{work}:/work",
             image, "bash", "-c", inner],
            capture_output=True, text=True, timeout=CONFIG.coverage_timeout,
        )
        export_path = work / "export.json"
        if not export_path.is_file() or export_path.stat().st_size == 0:
            raise CoverageReplayError(
                "coverage export failed: "
                + (proc.stderr or proc.stdout or "")[-300:])

        # 3) parse: covered function names (strip llvm-cov 'file.c:' prefix)
        data = json.loads(export_path.read_text())
        fns = (data.get("data") or [{}])[0].get("functions", [])
        covered: set[str] = set()
        for fn in fns:
            if fn.get("count", 0) > 0:
                name = fn.get("name", "")
                covered.add(name.split(":", 1)[-1] if ":" in name else name)
        return covered
    finally:
        shutil.rmtree(work, ignore_errors=True)


def reached_new_functions(input_paths: list[str],
                          before: set[str]) -> tuple[set[str], set[str]]:
    """Return (covered, newly_reached) function names for `input_paths`.

    `before` is the set of functions the campaign already covered; the result's
    second element is what these inputs added on top of it.
    """
    covered = run_coverage(input_paths)
    return covered, covered - before
