"""Target Mapper: attribute coverage to code regions and emit hypotheses."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from .config import CONFIG
from .coverage import CoverageSnapshot
from .llm import LLM


MAPPER_SYSTEM = """\
You are a fuzzing coverage analyst. You receive:
1. A list of functions in the target (extracted from source or binary symbols).
2. Coverage statistics: which seeds hit how many edges, and how edges cluster.
3. Optional source code excerpts.

Your job: infer WHICH FUNCTIONALITY is already exercised and WHICH nearby
functionality is NOT exercised ("near-miss" targets), and produce concrete
hypotheses about what input characteristics would reach new code.

Be specific: name functions, flags, magic bytes, format fields, parsers.
Respond ONLY with JSON matching the schema:
{
  "covered_functionality": ["..."],
  "near_miss": [
    {
      "functions": ["..."],
      "reasoning": "why it's close but not reached",
      "hint": "input characteristics likely to reach it"
    }
  ],
  "summary": "one-paragraph state of coverage"
}
"""


SURFACE_SYSTEM = """\
You are a security researcher hunting for unexplored attack surface.
You receive the target's source code (or function/symbol list) and the
current coverage state. Coverage has PLATEAUED: mutations no longer find
new edges. Your job is to propose ENTIRELY NEW input classes that the
current corpus has never resembled — new file formats, protocol states,
rare CLI/env-driven paths, deep parser combinations, boundary encodings.

Respond ONLY with JSON matching the schema:
{
  "attack_surfaces": [
    {
      "name": "short label",
      "entry_point": "function or path that handles it",
      "rationale": "why it is likely uncovered",
      "input_description": "precise description of a triggering input"
    }
  ]
}
"""


@dataclass
class Hypothesis:
    functions: list[str] = field(default_factory=list)
    reasoning: str = ""
    hint: str = ""

    def render(self) -> str:
        fns = ", ".join(self.functions) if self.functions else "(unspecified)"
        return f"[{fns}] {self.hint} ({self.reasoning})"


@dataclass
class AttackSurface:
    name: str = ""
    entry_point: str = ""
    rationale: str = ""
    input_description: str = ""


def _collect_function_names() -> list[str]:
    """Best-effort function list: from source dir via grep, else empty."""
    src = CONFIG.source_dir
    if not src or not os.path.isdir(src):
        return []
    names: set[str] = set()
    for root, _dirs, files in os.walk(src):
        for f in files:
            if not f.endswith((".c", ".h", ".cc", ".cpp")):
                continue
            try:
                with open(os.path.join(root, f), encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
            except OSError:
                continue
            # crude C-style function definition match
            import re
            for m in re.finditer(
                r"^[A-Za-z_][A-Za-z0-9_\* \t]*\**([A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*$",
                text, re.MULTILINE,
            ):
                names.add(m.group(1))
    return sorted(names)[:300]


def map_coverage(llm: LLM, snap: CoverageSnapshot) -> dict:
    """LLM-assisted mapping of coverage to functionality. Returns raw dict."""
    top = snap.top_seeds(CONFIG.max_seed_samples)
    seed_lines = "\n".join(
        f"- {s.name}: {len(s.edges)} edges, first bytes: {s.data[:48]!r}"
        for s in top
    )
    fns = _collect_function_names()
    fn_block = "\n".join(fns) if fns else "(no source dir configured)"

    user = f"""\
FUNCTIONS IN TARGET:
{fn_block}

COVERAGE SNAPSHOT:
{snap.render_context()}
- top seeds:
{seed_lines}
"""

    return llm.chat_json(MAPPER_SYSTEM, user)


def propose_surfaces(llm: LLM, snap: CoverageSnapshot, prior_summary: str = "") -> list[AttackSurface]:
    """Plateau mode: propose wholly new attack surfaces."""
    context = []
    if CONFIG.source_dir and os.path.isdir(CONFIG.source_dir):
        # include a bounded sample of source for grounding
        budget = 8000
        collected: list[str] = []
        for root, _dirs, files in os.walk(CONFIG.source_dir):
            for f in sorted(files):
                if not f.endswith((".c", ".h", ".cc", ".cpp")):
                    continue
                p = os.path.join(root, f)
                try:
                    with open(p, encoding="utf-8", errors="replace") as fh:
                        collected.append(f"--- {p} ---\n" + fh.read(budget))
                except OSError:
                    continue
                if sum(len(c) for c in collected) > budget * 6:
                    break
        context.append("\n".join(collected)[: budget * 6])

    top = snap.top_seeds(CONFIG.max_seed_samples)
    seed_lines = "\n".join(
        f"- {s.name}: {len(s.edges)} edges, first bytes: {s.data[:48]!r}"
        for s in top
    )
    user = f"""\
CURRENT CORPUS (top seeds):
{seed_lines}
{snap.render_context()}

PREVIOUS CYCLES SUMMARY:
{prior_summary or "(first cycle)"}

{'SOURCE CODE:' + chr(10) + chr(10).join(context) if context else '(no source available)'}
"""
    data = llm.chat_json(SURFACE_SYSTEM, user)
    return [
        AttackSurface(
            name=a.get("name", ""),
            entry_point=a.get("entry_point", ""),
            rationale=a.get("rationale", ""),
            input_description=a.get("input_description", ""),
        )
        for a in data.get("attack_surfaces", [])
    ][:6]
