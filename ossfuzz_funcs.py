"""Function coverage from the OSS-Fuzz coverage pipeline (llvm-cov).

Consumes the digest.json produced by tools/ossfuzz_coverage.py digest:
clang source-based coverage of the whole campaign corpus (e.g. an AFL++
queue dir replayed through the sanitizer=coverage build).

Set FUZZ_OSSFUZZ_REPORT_DIR in agent/.env to the report directory and the
agent gains real function-level coverage for real-world targets, without
needing a gcov build.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .config import CONFIG


@dataclass
class OssfuzzFunction:
    name: str
    count: int
    file: str
    line: int


@dataclass
class OssfuzzCoverage:
    """Aggregated llvm-cov coverage of the corpus (from digest.json)."""
    target: str = ""
    totals: dict = field(default_factory=dict)          # metric -> {covered,total,pct}
    files: list[dict] = field(default_factory=list)     # per-file line coverage
    functions: list[OssfuzzFunction] = field(default_factory=list)  # all declared
    zero_cov_functions: list[str] = field(default_factory=list)

    @property
    def hit_functions(self) -> list[OssfuzzFunction]:
        return [f for f in self.functions if f.count > 0]

    def render_for_llm(self, max_functions: int = 40) -> str:
        """Compact textual summary for prompts."""
        parts = []
        t = self.totals
        if t:
            def fmt(metric: str) -> str:
                v = t.get(metric)
                if not v:
                    return f"{metric}: n/a"
                return f"{metric}: {v['covered']}/{v['total']} ({v['pct']}%)"
            parts.append("totals: " + ", ".join(fmt(m) for m in
                                                ("functions", "lines", "branches") if m in t))
        hot = self.hit_functions[:max_functions]
        if hot:
            fns = ", ".join(f"{f.name}({f.count})" for f in hot)
            parts.append(f"executed functions (hits): {fns}")
        if self.zero_cov_functions:
            zc = ", ".join(sorted(self.zero_cov_functions)[:max_functions])
            n = len(self.zero_cov_functions)
            parts.append(f"NEVER-EXECUTED functions ({n} total, sample): {zc}")
        return "\n".join(parts)


def load_ossfuzz_coverage(report_dir: str | None = None) -> OssfuzzCoverage | None:
    """Load digest.json written by tools/ossfuzz_coverage.py digest.

    Returns None when not configured or the digest is missing/unparsable.
    """
    d = Path(report_dir or CONFIG.ossfuzz_report_dir)
    if not str(d):
        return None
    path = d / "digest.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    functions = [
        OssfuzzFunction(
            name=fn.get("name", "?"),
            count=int(fn.get("count", 0)),
            file=fn.get("file", "?"),
            line=int(fn.get("line", 0)),
        )
        for fn in data.get("functions", [])
    ]
    return OssfuzzCoverage(
        target=data.get("target", ""),
        totals=data.get("totals", {}),
        files=data.get("files", []),
        functions=functions,
        zero_cov_functions=data.get("zero_cov_functions", []),
    )
