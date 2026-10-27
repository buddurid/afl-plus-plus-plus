"""Campaign status report — pure execution, zero LLM.

Reads the corpus (FUZZ_CORPUS_DIR), runs every seed through afl-showmap
and, if configured, the gcov build; prints a human-readable coverage
report: totals, per-file edge counts, functions reached, source files
touched, and never-executed functions.

Usage (from repo root):
    .venv/bin/python -m agent.orchestrator --status
    FUZZ_CORPUS_DIR=/work/proj/afl_out .venv/bin/python -m agent.orchestrator --status
"""

from __future__ import annotations

import os
from collections import Counter

from .config import CONFIG
from .coverage import analyze


def print_status(top_n: int = 15) -> None:
    CONFIG.ensure_dirs()
    print("=" * 62)
    print("FUZZING CAMPAIGN STATUS")
    print("=" * 62)
    print(f"corpus dir   : {os.path.abspath(CONFIG.corpus_dir)}")
    print(f"target       : {CONFIG.target_cmd}  (cwd={CONFIG.target_dir})")

    snap = analyze()
    if not snap.seeds:
        print("\n(no corpus inputs found — nothing to report)")
        return

    total_execs = len(snap.seeds)
    print(f"seeds        : {total_execs}")
    print(f"edges covered: {snap.num_edges}")

    # --- per-seed edge histogram -----------------------------------------
    counts = Counter(len(s.edges) for s in snap.seeds)
    print("\nseeds by edge count:")
    for k in sorted(counts, reverse=True)[:top_n]:
        print(f"  {k:>4} edges : {'#' * min(counts[k], 40)} {counts[k]} seed(s)")

    # --- most valuable seeds ---------------------------------------------
    print(f"\ntop seeds (by unique edges):")
    for s in snap.top_seeds(5):
        print(f"  {s.name:<40} {len(s.edges):>4} edges  {len(s.data):>7}B")

    # --- function / line coverage (gcov) ----------------------------------
    sc = snap.source_coverage
    if sc is None:
        print("\nfunction/line coverage: DISABLED")
        print("  (set FUZZ_GCOV_BUILD_DIR in agent/.env to enable)")
    else:
        print(f"\nfunctions reached ({len(sc.all_functions)}):")
        for fn in sorted(sc.all_functions)[:top_n]:
            n_inputs = sum(1 for r in sc.per_input.values() if fn in r.functions)
            print(f"  {fn:<45} hit by {n_inputs}/{total_execs} inputs")
        if len(sc.all_functions) > top_n:
            print(f"  ... and {len(sc.all_functions) - top_n} more")

        src_files = sorted({f for f, _ in sc.all_lines})
        print(f"\nsource files touched ({len(src_files)}):")
        for f in src_files[:top_n]:
            lines = sorted(l for sf, l in sc.all_lines if sf == f)
            head = lines[:8]
            more = f" ...+{len(lines) - 8}" if len(lines) > 8 else ""
            print(f"  {f}")
            print(f"      lines: {head}{more}")
        if len(src_files) > top_n:
            print(f"  ... and {len(src_files) - top_n} more files")

        if sc.zero_cov_functions:
            zc = sorted(sc.zero_cov_functions)
            print(f"\nNEVER-EXECUTED functions ({len(zc)}):")
            for fn in zc[:top_n]:
                print(f"  {fn}")
            if len(zc) > top_n:
                print(f"  ... and {len(zc) - top_n} more")

    # --- memory snapshot ---------------------------------------------------
    try:
        from .memory import Memory
        mem = Memory()
        cycles = mem.last_cycles(5)
        if cycles:
            print(f"\nrecent cycles ({len(cycles)} of last 5 shown):")
            for c in cycles:
                print(
                    f"  mode={c['mode']:<9} accepted={c['accepted']:>3} "
                    f"new_edges={c['new_edges']:>4} total_edges={c['total_edges']}"
                )
            print(f"plateaued: {mem.is_plateaued()}")
        else:
            print("\nno agent cycles recorded yet (memory file empty/missing)")
    except Exception:
        pass

    print("=" * 62)
