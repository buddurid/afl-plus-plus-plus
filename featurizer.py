"""Feature / subsystem gap analysis from llvm-cov coverage.

The agent should reason about *behaviour*, not just edge counts. This module
turns the OSS-Fuzz llvm-cov digest into:

  * per-subsystem (source file) coverage, ranked by how much code is dark
  * the entry-point-ish functions of each dark subsystem
  * real source excerpts of uncovered functions, so the LLM can read the
    code and infer WHICH feature it implements and WHAT input drives it
  * a compact, behaviour-oriented brief for prompts

Target-agnostic: nothing here knows about a specific project. Symbol/entry
point heuristics come from config (FUZZ_SYMBOL_PREFIXES) and degrade safely
to "treat everything as a candidate" when unset.

Everything is deterministic; only the interpretation is left to the LLM.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import CONFIG

# Functions that are rarely reachable from fuzzer input (allocators, teardown,
# debug/diagnostic helpers). Generic heuristics, not project specific.
_LOW_SIGNAL = re.compile(
    r"(_free|Free|Cleanup|Deinit|Teardown|Debug|Dump|Print|Malloc|Calloc|Realloc|"
    r"Strdup|Memcpy|MemSet|Deregister|Unregister|_new$|_delete$)", re.I)


def _short(name: str) -> str:
    """Strip the redundant 'file.c:' prefix llvm-cov adds to static funcs."""
    return name.split(":", 1)[-1] if ":" in name else name


def _symbol_prefixes() -> tuple[str, ...]:
    """Configured public-API prefixes (FUZZ_SYMBOL_PREFIXES). Empty = none."""
    raw = CONFIG.symbol_prefixes
    return tuple(p.strip() for p in raw.replace(",", " ").split() if p.strip())


def _looks_like_entry(name: str) -> bool:
    """Heuristic: is this function a plausible public entry point?

    With configured prefixes, match those. Without them, fall back to a
    language-neutral heuristic: not a low-signal name, and reasonably named.
    """
    n = _short(name)
    if _LOW_SIGNAL.search(n):
        return False
    prefixes = _symbol_prefixes()
    if prefixes:
        return n.startswith(prefixes)
    return True


@dataclass
class DarkFunction:
    name: str
    file: str
    line: int


@dataclass
class Subsystem:
    """Coverage of one source file = one feature area."""
    file: str
    lines_covered: int
    lines_total: int
    pct: float
    functions_total: int
    functions_covered: int
    dark_functions: list[DarkFunction] = field(default_factory=list)
    entry_points: list[str] = field(default_factory=list)

    @property
    def dark_lines(self) -> int:
        return max(0, self.lines_total - self.lines_covered)

    def render(self, max_dark: int = 6) -> str:
        ep = ", ".join(_short(e) for e in self.entry_points[:4]) if self.entry_points else "(none identified)"
        df = ", ".join(_short(f.name) for f in self.dark_functions[:max_dark]) or "(none)"
        return (f"{self.file}: {self.lines_covered}/{self.lines_total} lines "
                f"({self.pct}%), {self.functions_covered}/{self.functions_total} functions; "
                f"entry points: {ep}; unreached: {df}")


def build_subsystems(oc) -> list[Subsystem]:
    """Group an OssfuzzCoverage digest by source file, ranked darkest-first."""
    if oc is None:
        return []
    by_file: dict[str, Subsystem] = {}
    for fdef in oc.files:
        f = fdef.get("file", "?")
        by_file[f] = Subsystem(
            file=f,
            lines_covered=int(fdef.get("lines_covered", 0)),
            lines_total=int(fdef.get("lines_total", 0)),
            pct=float(fdef.get("pct", 0.0)),
            functions_total=0,
            functions_covered=0,
        )
    for fn in oc.functions:
        sub = by_file.get(fn.file)
        if sub is None:
            sub = by_file.setdefault(fn.file, Subsystem(fn.file, 0, 0, 0.0, 0, 0))
        sub.functions_total += 1
        if fn.count > 0:
            sub.functions_covered += 1
        else:
            sub.dark_functions.append(DarkFunction(fn.name, fn.file, fn.line))

    for sub in by_file.values():
        sub.dark_functions.sort(key=lambda d: (0 if _looks_like_entry(d.name) else 1, d.line))
        sub.entry_points = [d.name for d in sub.dark_functions if _looks_like_entry(d.name)][:6]
    # rank by dark lines (biggest unexercised feature first)
    return sorted(by_file.values(), key=lambda s: -s.dark_lines)


def _harness_source(harness_file: str, budget: int = 8000) -> str:
    """Bounded excerpt of the fuzz harness AND the local sources it #includes,
    so the LLM can judge reachability and learn the input format."""
    from .harness import load_harness_context

    text = load_harness_context(harness_file, total_budget=budget,
                                per_file=budget)
    return text[:budget]


def rank_subsystems(subs: list[Subsystem], harness_file: str = "",
                    driven_min_pct: float | None = None) -> tuple[list[Subsystem], list[Subsystem]]:
    """Split dark subsystems into (driven, untouched) by coverage proximity.

    Target-agnostic reachability signal: a subsystem the harness meaningfully
    exercises is definitely reachable and simply has room left — the strongest
    targets. A subsystem at ~0% is either outside this harness's scope (another
    fuzzer's domain) or gated behind a feature switch the input never sets; the
    LLM decides which, using the harness source.

    `driven_min_pct` guards against incidental coverage (a few init lines):
    files below it count as untouched. Defaults to FUZZ_SUBSYSTEM_DRIVEN_PCT.
    """
    if driven_min_pct is None:
        driven_min_pct = CONFIG.subsystem_driven_pct
    driven = [s for s in subs if s.lines_total > 0 and s.pct >= driven_min_pct]
    untouched = [s for s in subs if s.lines_total > 0 and s.pct < driven_min_pct]
    driven.sort(key=lambda s: -s.dark_lines)
    untouched.sort(key=lambda s: -s.dark_lines)
    return driven, untouched


def _read_source(source_dir: str, rel_file: str) -> str | None:
    """Read a source file, tolerating digest paths like 'proj/subdir/file.c'."""
    if not source_dir:
        return None
    candidates = [rel_file, rel_file.split("/", 1)[-1], os.path.basename(rel_file)]
    for cand in candidates:
        p = Path(source_dir) / cand
        if p.is_file():
            try:
                return p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return None
    return None


def extract_function_source(source_dir: str, sub: Subsystem, budget: int = 6000,
                            max_fns: int = 8) -> str:
    """Return real source excerpts for the subsystem's uncovered entry points.

    Excerpts are anchored at the function definition and read forward a bounded
    amount, so the LLM sees the actual parsing/feature logic to target.
    """
    text = _read_source(source_dir, sub.file)
    if not text:
        return ""
    lines = text.splitlines()
    wanted = [d for d in sub.dark_functions if _looks_like_entry(d.name)][:max_fns]
    if not wanted:
        wanted = sub.dark_functions[:max_fns]

    chunks: list[str] = []
    used = 0
    for d in wanted:
        start = max(0, d.line - 1)
        # find the opening brace forward from the declaration line
        end = start
        depth = 0
        seen_brace = False
        for i in range(start, min(len(lines), start + 120)):
            line = lines[i]
            depth += line.count("{") - line.count("}")
            if "{" in line:
                seen_brace = True
            end = i
            if seen_brace and depth <= 0:
                break
        body = "\n".join(lines[start:end + 1])
        if len(body) > 1600:
            body = body[:1600] + "\n    /* ...truncated... */"
        chunk = f"// {sub.file}:{d.line}  ({d.name})\n{body}"
        if used + len(chunk) > budget:
            break
        chunks.append(chunk)
        used += len(chunk)
    return "\n\n".join(chunks)


def behavior_brief(oc, source_dir: str, top_subsystems: int = 6,
                   max_zero_fns: int = 60, harness_file: str = "") -> str:
    """LLM-ready description of unreached behaviour, code-grounded."""
    subs = build_subsystems(oc)
    driven, untouched = rank_subsystems(subs, harness_file)
    if not driven and not untouched:
        return "(no llvm-cov digest available — falling back to edge analysis)"

    out = ["BEHAVIOURAL COVERAGE ANALYSIS (read before planning):", ""]

    out.append("A) SUBSYSTEMS THIS HARNESS ALREADY PARTLY DRIVES — "
               "the strongest targets: the code is reachable (coverage > 0%), "
               "there is just room left. Extending these needs the NEXT "
               "construct/flag/branch, closer to what the corpus already does:")
    for s in driven[:top_subsystems]:
        out.append("  - " + s.render())
    if not driven:
        out.append("  (none — the harness reaches nothing yet)")

    out.append("")
    out.append("B) SUBSYSTEMS AT 0% COVERAGE — harder call. Some are "
               "simply out of scope for THIS harness (driven by other fuzzers); "
               "others are in scope but gated behind a feature switch the input "
               "never sets. Use the harness source below to tell them apart. "
               "Only pursue one if you can name the exact precondition:")
    for s in untouched[:top_subsystems]:
        out.append("  - " + s.render())

    hs = _harness_source(harness_file)
    if hs:
        out.append("")
        out.append("THIS IS THE FUZZ HARNESS (its API calls define what is IN "
                   "scope; anything it never calls is out of scope):")
        out.append(hs)

    out.append("")
    out.append("UNREACHED FUNCTIONS THAT LOOK LIKE ENTRY POINTS "
               "(strongest candidates for a feature nobody drives):")
    entry_fns: list[str] = []
    for s in (driven + untouched):
        for ep in s.entry_points:
            entry_fns.append(f"{ep}  [{s.file}]")
    out.append("  " + ("\n  ".join(entry_fns[:max_zero_fns]) if entry_fns else "(none)"))

    zc = getattr(oc, "zero_cov_functions", []) or []
    if zc:
        out.append("")
        out.append(f"NEVER-EXECUTED FUNCTIONS ({len(zc)} total, sample): "
                   + ", ".join(sorted(zc)[:max_zero_fns]))

    # code excerpts for the darkest IN-SCOPE subsystems, so the agent can read them
    out.append("")
    out.append("SOURCE OF THE DARKEST IN-SCOPE SUBSYSTEMS (read this — decide "
               "what each function implements and what input would drive it):")
    budget_left = 9000
    for s in (driven + untouched)[:2]:
        src = extract_function_source(source_dir, s, budget=budget_left // 2)
        if src:
            out.append(f"\n===== {s.file} =====")
            out.append(src)
            budget_left -= len(src)
    return "\n".join(out)


def dark_feature_targets(oc, source_dir: str, limit: int = 5,
                         harness_file: str = "") -> list[dict]:
    """Structured targets passed to the generator: one per dark subsystem,
    in-scope (partly driven) subsystems first."""
    subs = build_subsystems(oc)
    driven, untouched = rank_subsystems(subs, harness_file)
    out: list[dict] = []
    for s in (driven + untouched):
        if s.lines_total == 0 or s.dark_lines == 0:
            continue
        src = extract_function_source(source_dir, s, budget=CONFIG.target_source_bytes)
        if not src:
            continue
        out.append({
            "file": s.file,
            "pct": s.pct,
            "dark_lines": s.dark_lines,
            "entry_points": s.entry_points[:6],
            "unreached_functions": [d.name for d in s.dark_functions[:12]],
            "source": src,
        })
        if len(out) >= limit:
            break
    return out
