"""Coverage-preserving corpus minimization via afl-cmin.

Given a directory of inputs (the campaign queue), produce a much smaller set
that preserves the SAME edge coverage. This is the standard AFL corpus
minimization step ("tmin"/"cmin"): dropping inputs that add no unique edges
shrinks the queue dramatically (e.g. 2000 -> 100) WITHOUT losing coverage, so
the fuzzer spends its energy mutating a tight, high-value set and the agent
sees a smaller, sharper seed set.

Runs afl-cmin inside the OSS-Fuzz base-runner container, mirroring the campaign
itself, so the same instrumented target is used for the coverage measurement.

Target-agnostic: the binary, its build dir, and the input placeholder all come
from config.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .config import CONFIG


class MinimizeError(RuntimeError):
    pass


def available() -> bool:
    """True when a minimization build + binary are configured."""
    return bool(CONFIG.minimize_build_dir and CONFIG.minimize_bin)


def _target_cmd() -> str:
    """The cmin target command (argv-file style), defaulting to <bin> @@."""
    if CONFIG.minimize_target_cmd:
        return CONFIG.minimize_target_cmd
    # target_cmd is remembered as e.g. ".../xml @@"; reuse just the "@@" style.
    base = CONFIG.target_cmd.split()
    if "@@" in base:
        return f"/out/{CONFIG.minimize_bin} @@"
    return f"/out/{CONFIG.minimize_bin}"


def minimize_corpus(in_dir: str, out_dir: str, timeout: int | None = None,
                    min_files: int | None = None) -> dict:
    """Minimize `in_dir` into `out_dir` keeping coverage identical.

    Returns {"input": n, "output": m, "skipped": bool}. `out_dir` is created
    fresh (afl-cmin refuses a non-empty output dir). When the input is already
    at/below `min_files` (default FUZZ_MINIMIZE_MIN_QUEUE) it is a no-op copy.
    """
    if not available():
        raise MinimizeError("minimization not configured "
                            "(FUZZ_MINIMIZE_BUILD_DIR/FUZZ_MINIMIZE_BIN)")
    in_path = Path(in_dir).resolve()
    out_path = Path(out_dir).resolve()
    if not in_path.is_dir():
        raise MinimizeError(f"input dir does not exist: {in_path}")

    files = [p for p in in_path.iterdir() if p.is_file()]
    n_in = len(files)
    min_files = CONFIG.minimize_min_queue if min_files is None else min_files

    if out_path.exists():
        shutil.rmtree(out_path, ignore_errors=True)
    out_path.mkdir(parents=True, exist_ok=True)

    # Already small enough: nothing to gain, just mirror the input.
    if n_in <= min_files:
        for f in files:
            shutil.copy2(f, out_path / f.name)
        return {"input": n_in, "output": n_in, "skipped": True}

    build = Path(CONFIG.minimize_build_dir).resolve()
    image = CONFIG.minimize_image
    # Mount the input read-only and a work dir (parent of out) writable so
    # afl-cmin can create its output subdir inside the container.
    work = out_path.parent / (out_path.name + "_work")
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    tmp_out = work / "cmin_out"

    cmd = [
        "docker", "run", "--rm",
        "-e", "AFL_SKIP_CPUFREQ=1",
        "-e", "AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1",
        "-e", "AFL_IGNORE_PROBLEMS=1",
        "-e", "AFL_NO_UI=1",
        "-e", "ASAN_OPTIONS=detect_leaks=0:symbolize=0",
        "-v", f"{build}:/out",
        "-v", f"{in_path}:/cmin_in:ro",
        "-v", f"{work}:/work",
        image, "bash", "-c",
        # clear any root-owned leftover from a previous run (afl-cmin refuses a
        # non-empty output dir), then run cmin and hand /work back to the host.
        f"rm -rf /work/cmin_out; "
        f"/out/afl-cmin -i /cmin_in -o /work/cmin_out -t {CONFIG.exec_timeout * 1000} "
        f"-- {_target_cmd()}; rc=$?; "
        f"chown -R {os.getuid()}:{os.getgid()} /work 2>/dev/null; exit $rc",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout or CONFIG.minimize_timeout)
    except subprocess.TimeoutExpired as e:
        raise MinimizeError(f"afl-cmin timed out after {timeout or CONFIG.minimize_timeout}s") from e
    if proc.returncode != 0 or not tmp_out.is_dir():
        tail = (proc.stdout or "")[-500:] + (proc.stderr or "")[-500:]
        raise MinimizeError(f"afl-cmin failed (rc={proc.returncode}): {tail}")

    # Copy the minimized set into out_dir (docker wrote it root-owned).
    n_out = 0
    for f in sorted(tmp_out.iterdir()):
        if f.is_file():
            shutil.copy2(f, out_path / f.name)
            n_out += 1
    shutil.rmtree(work, ignore_errors=True)
    return {"input": n_in, "output": n_out, "skipped": False}


def main() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="coverage-preserving corpus minimization")
    ap.add_argument("--in", dest="in_dir", required=True)
    ap.add_argument("--out", dest="out_dir", required=True)
    ap.add_argument("--min-files", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=None)
    args = ap.parse_args()

    try:
        res = minimize_corpus(args.in_dir, args.out_dir,
                              timeout=args.timeout, min_files=args.min_files)
    except MinimizeError as e:
        print(f"minimize error: {e}")
        raise SystemExit(2)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
