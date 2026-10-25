"""Coverage analysis: run seeds through afl-showmap and aggregate edge sets."""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass, field
from typing import Iterable

from .config import CONFIG


@dataclass
class Seed:
    path: str
    data: bytes
    edges: frozenset[int] = field(default_factory=frozenset)

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.data).hexdigest()[:16]


@dataclass
class CoverageSnapshot:
    seeds: list[Seed] = field(default_factory=list)
    all_edges: set[int] = field(default_factory=set)
    # Optional rich coverage (function/line level, gcov-backed). None when
    # FUZZ_GCOV_BUILD_DIR is not configured.
    source_coverage: "object | None" = None  # funcs_coverage.SourceCoverage

    @property
    def num_edges(self) -> int:
        return len(self.all_edges)

    def top_seeds(self, n: int) -> list[Seed]:
        """Seeds ranked by edge count (most coverage first)."""
        return sorted(self.seeds, key=lambda s: len(s.edges), reverse=True)[:n]

    def render_context(self) -> str:
        """Full context block for LLM prompts: edges + functions/lines."""
        ctx = [
            f"total corpus seeds: {len(self.seeds)}",
            f"total edges covered: {self.num_edges}",
        ]
        if self.source_coverage is not None:
            ctx.append("FUNCTION/LINE COVERAGE (gcov):")
            ctx.append(self.source_coverage.render_for_llm())
        return "\n".join(ctx)


class _TargetCmd:
    """Builds the per-input target command line from CONFIG.target_cmd."""

    def build_cmd(self, input_path: str) -> list[str]:
        """Assemble the target command line for one input.

        - "@@" present  -> replaced by the input path (argv mode).
        - "@-" present  -> replaced by the input path on stdin (fd mode).
        - neither       -> input piped to the target's stdin.
        """
        base = CONFIG.target_cmd.split()
        if "@@" in base:
            return base[: base.index("@@")] + [input_path] + base[base.index("@@") + 1 :]
        if "@-" in base:
            idx = base.index("@-")
            return base[:idx] + [input_path] + base[idx + 1 :]
        return base


TARGET_CMD = _TargetCmd()


def _run_showmap(input_path: str) -> tuple[frozenset[int], bool]:
    """Run afl-showmap on a single input. Returns (edges, crashed/hung)."""
    stdin_file = None
    base = CONFIG.target_cmd.split()
    uses_argv_file = "@@" in base or "@-" in base
    # Special case: target reads its input from a file it OPENS itself via
    # "-f <path>" (AFL++ test-instr style). Passing the file path as a bare
    # argv[1] would feed the target the path string as content. Detect the
    # "-f <marker>" pattern and substitute the real input path there.
    if "@@" in base:
        i = base.index("@@")
        if i > 0 and base[i - 1] == "-f":
            base = base[: i - 1] + ["-f", input_path] + base[i + 1 :]
            uses_argv_file = False  # target opens the file itself; no stdin
        else:
            base = base[:i] + [input_path] + base[i + 1 :]
    elif "@-" in base:
        i = base.index("@-")
        base = base[:i] + [input_path] + base[i + 1 :]
    cmd = [
        CONFIG.showmap_bin,
        "-q",                       # quiet
        "-t", str(CONFIG.exec_timeout * 1000),
        "-o", "-",                  # stdout
        "--",
    ] + base

    if not uses_argv_file and "-f" not in base:
        try:
            stdin_file = open(input_path, "rb")
        except OSError:
            stdin_file = None

    env = dict(os.environ, AFL_QUIET="1", AFL_SKIP_CPUFREQ="1")
    try:
        proc = subprocess.run(
            cmd,
            cwd=CONFIG.target_dir,
            env=env,
            stdin=stdin_file,
            capture_output=True,
            timeout=CONFIG.exec_timeout + 5,
        )
    except subprocess.TimeoutExpired:
        return frozenset(), True
    finally:
        if stdin_file:
            stdin_file.close()

    edges: set[int] = set()
    for line in proc.stdout.decode(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        # showmap output lines: "edge_id: hit_count" (possibly @tuple format)
        try:
            eid = int(line.split(":")[0].split("@")[0])
            edges.add(eid)
        except ValueError:
            continue
    return frozenset(edges), proc.returncode not in (0,)
    # NOTE: showmap returns non-zero on target crash; we still keep the edges.


def load_corpus(corpus_dir: str | None = None) -> list[str]:
    d = corpus_dir or CONFIG.corpus_dir
    if not os.path.isdir(d):
        return []
    out = []
    for root, _dirs, files in os.walk(d):
        for f in sorted(files):
            out.append(os.path.join(root, f))
    return out


def analyze(corpus_dir: str | None = None) -> CoverageSnapshot:
    """Build a full coverage snapshot of the current corpus.

    Deterministic and cheap to extend: per-seed edge sets allow the mapper
    and generator to reason about which seed reaches what.
    """
    snap = CoverageSnapshot()
    for path in load_corpus(corpus_dir):
        try:
            with open(path, "rb") as fh:
                data = fh.read(CONFIG.max_input_size * 4)
        except OSError:
            continue
        edges, _bad = _run_showmap(path)
        snap.seeds.append(Seed(path=path, data=data, edges=edges))
        snap.all_edges |= edges

    # Optional richer layer: functions/lines via a gcov-instrumented build
    if CONFIG.gcov_build_dir and snap.seeds:
        from .funcs_coverage import analyze_source_coverage, run_with_gcov
        from .config import CONFIG as C

        gcov_cmd = C.gcov_target_cmd or CONFIG.target_cmd

        def build_cmd_fn(input_path: str) -> list[str]:
            base = gcov_cmd.split()
            if "@@" in base:
                i = base.index("@@")
                if i > 0 and base[i - 1] == "-f":
                    return base[: i - 1] + ["-f", input_path] + base[i + 1 :]
                return base[:i] + [input_path] + base[i + 1 :]
            if "@-" in base:
                i = base.index("@-")
                return base[:i] + [input_path] + base[i + 1 :]
            return base

        snap.source_coverage = analyze_source_coverage(
            [(s.name, s.data) for s in snap.seeds], build_cmd_fn
        )
    return snap
