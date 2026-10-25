"""gcov-backed function/line coverage.

Complements AFL++ edge coverage with human/LLM-meaningful coverage data:
which FUNCTIONS and which SOURCE LINES an input executed.

Mechanism (same idea as afl-cov, without the wrapper):
  1. The target is built once with gcov instrumentation
     (gcc --coverage => .gcno notes + .gcda counters at runtime).
  2. Per input: reset counters -> run target -> `gcov -b -i` dumps a JSON
     intermediate (.gcov.json.gz) -> parse functions/lines/branches.
  3. Aggregate over the corpus => per-input attribution + zero-cov report
     (functions/lines NEVER executed by any input).

This module is additive: edge coverage (coverage.py) stays the cheap
acceptance filter; this module runs only where richer context is worth
the extra exec (snapshot, accepted-input reporting).
"""

from __future__ import annotations

import gzip
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .config import CONFIG


@dataclass
class GcovResult:
    """Function/line/branch coverage for ONE input execution."""
    functions: dict[str, int] = field(default_factory=dict)   # name -> exec count
    lines: set[int] = field(default_factory=set)              # covered line numbers
    branches_total: int = 0
    branches_taken: int = 0
    source_files: list[str] = field(default_factory=list)

    @property
    def function_names(self) -> list[str]:
        return sorted(self.functions.keys())


def _reset_counters() -> None:
    """Delete .gcda files (build dir + GCOV_PREFIX output) so the next run
    counts only its own execution."""
    for gcda in _find_gcda():
        try:
            gcda.unlink()
        except OSError:
            pass


def _find_gcda() -> list[Path]:
    """gcda files live either in the build dir or under GCOV_PREFIX output."""
    roots = [Path(CONFIG.gcov_build_dir)]
    prefix = Path(CONFIG.gcov_build_dir) / "gcov_out"
    if prefix.is_dir():
        roots.append(prefix)
    out: list[Path] = []
    for r in roots:
        out.extend(r.rglob("*.gcda"))
    return out


def _run_gcov_json() -> dict:
    """Run gcov -b -i on every .gcda and merge the JSON intermediates."""
    merged: dict = {"files": []}
    for gcda in _find_gcda():
        out = subprocess.run(
            ["gcov", "-b", "-i", str(gcda)],
            cwd=CONFIG.gcov_build_dir,
            capture_output=True,
            timeout=30,
        )
        for gz in Path(CONFIG.gcov_build_dir).rglob("*.gcov.json.gz"):
            try:
                data = json.load(gzip.open(gz))
                merged["files"].extend(data.get("files", []))
            except (OSError, json.JSONDecodeError):
                continue
            finally:
                try:
                    gz.unlink()
                except OSError:
                    pass
    return merged

def run_with_gcov(argv_cmd: list[str], stdin_data: bytes | None = None) -> GcovResult | None:
    """Execute the gcov-built target on one input and extract coverage.

    argv_cmd: full command line for the gcov binary (input path already
              substituted, same convention as _TargetCmd.build_cmd).
    Returns None if gcov support is not configured/usable.
    """
    if not CONFIG.gcov_build_dir or not Path(CONFIG.gcov_build_dir).is_dir():
        return None

    _reset_counters()
    # No GCOV_PREFIX: the target's cwd IS the gcov build dir, so .gcda files
    # are written next to their .gcno notes (required for gcov to match them).
    env = dict(os.environ)
    try:
        proc = subprocess.run(
            argv_cmd,
            cwd=CONFIG.gcov_build_dir,
            input=stdin_data,
            capture_output=True,
            timeout=CONFIG.exec_timeout + 5,
            env=env,
        )
    except subprocess.TimeoutExpired:
        # counters may still be flushed via atexit on SIGTERM; try anyway
        pass

    if not _find_gcda():
        return None  # target crashed before flushing counters

    result = GcovResult()
    for fdata in _run_gcov_json().get("files", []):
        result.source_files.append(fdata.get("file", "?"))
        for fn in fdata.get("functions", []):
            name = fn.get("name", "?")
            result.functions[name] = result.functions.get(name, 0) + fn.get("execution_count", 0)
        for ln in fdata.get("lines", []):
            if ln.get("count", 0) > 0:
                result.lines.add(ln.get("line_number"))
            for br in ln.get("branches", []):
                result.branches_total += 1
                if br.get("count", 0) > 0:
                    result.branches_taken += 1
    return result


@dataclass
class SourceCoverage:
    """Aggregated function/line coverage over the corpus."""
    per_input: dict[str, GcovResult] = field(default_factory=dict)  # seed name -> result
    all_functions: set[str] = field(default_factory=set)
    all_lines: set[tuple[str, int]] = field(default_factory=set)    # (source_file, line)
    zero_cov_functions: set[str] = field(default_factory=set)       # never executed

    def render_for_llm(self, max_lines: int = 40) -> str:
        """Compact textual summary for prompts."""
        parts = []
        for name, res in self.per_input.items():
            fns = ", ".join(res.function_names) or "(none)"
            br = f"{res.branches_taken}/{res.branches_total}" if res.branches_total else "n/a"
            parts.append(f"- {name}: functions=[{fns}] lines={sorted(res.lines)[:10]} branches_taken={br}")
        if self.zero_cov_functions:
            zc = ", ".join(sorted(self.zero_cov_functions)[:max_lines])
            parts.append(f"- NEVER-EXECUTED FUNCTIONS (across whole corpus): {zc}")
        return "\n".join(parts)


def analyze_source_coverage(seeds: list[tuple[str, bytes]], build_cmd_fn) -> SourceCoverage | None:
    """Compute per-input function/line coverage for the corpus.

    seeds: list of (name, data)
    build_cmd_fn: callable(input_path) -> argv list (from coverage.TARGET_CMD)
    Returns None if gcov build is not configured.
    """
    if not CONFIG.gcov_build_dir or not Path(CONFIG.gcov_build_dir).is_dir():
        return None

    cov = SourceCoverage()
    tmp = Path(CONFIG.gcov_build_dir) / "_agent_input.tmp"

    all_seen_functions: set[str] = set()
    declared_functions: set[str] = set()

    for name, data in seeds:
        tmp.write_bytes(data)
        argv_cmd = build_cmd_fn(str(tmp))
        res = run_with_gcov(argv_cmd)
        if res is None:
            continue
        cov.per_input[name] = res
        cov.all_functions.update(res.function_names)
        for sf in res.source_files:
            for ln in res.lines:
                cov.all_lines.add((sf, ln))
        all_seen_functions.update(res.functions.keys())

    # discover ALL functions declared in the binary (run once with garbage
    # input just to get the full function list from gcov notes)
    tmp.write_bytes(b"\x00\x01\x02agent-probe\x00")
    probe = run_with_gcov(build_cmd_fn(str(tmp)))
    if probe:
        declared_functions = set(probe.functions.keys())
    cov.zero_cov_functions = declared_functions - all_seen_functions

    try:
        tmp.unlink()
    except OSError:
        pass
    return cov
