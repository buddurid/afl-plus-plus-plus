"""Target Mapper: attribute coverage to code regions and emit hypotheses."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from .config import CONFIG
from .coverage import CoverageSnapshot
from .featurizer import behavior_brief, build_subsystems
from .llm import LLM


MAPPER_SYSTEM = """\
You are a fuzzing coverage analyst specialised in reaching DARK CODE.

You receive llvm-cov coverage of the target, grouped by SUBSYSTEM (source
file), with the public entry points and functions that are NEVER executed,
and real source excerpts of the darkest functions.

Your job is NOT to enumerate functions. Your job is to read the code and
answer, in behavioural terms:
  1. WHAT KIND OF BEHAVIOUR have we been triggering so far? (describe the
     code paths the corpus actually drives)
  2. WHICH FEATURES / SUBSYSTEMS are we failing to reach? For each, name the
     concrete FEATURE (a capability the program has, in words — not just a
     symbol or file name).
  3. For each dark feature, explain the PRECONDITION that gates it: what makes
     the target *enter* that code? A control/option flag, a specific construct
     in the input, a referenced external resource, an encoding, a size or
     alignment boundary, a multi-chunk split, a required header, etc.
  4. Give a precise, format-aware INPUT RECIPE that satisfies that precondition
     for THIS harness, grounded in the source you were shown.

Think about the harness: fuzz inputs are usually STRUCTURED, not raw payloads.
A leading header of control fields, then a body; a feature is often gated by a
header bit/field OR by a construct in the body — identify which, and say the
exact value or construct to set.

Be concrete and code-grounded. Cite the function names and constants you saw.
Respond ONLY with JSON matching the schema:
{
  "triggered_behaviour": ["what behaviour we ARE exercising, in words"],
  "near_miss": [
    {
      "feature": "short name of the dark FEATURE (behaviour, not symbol)",
      "functions": ["entry point / key functions that implement it"],
      "precondition": "exact condition that gates entry to this code",
      "reasoning": "why it is currently unreached, grounded in the code",
      "hint": "precise input recipe (option/flag / construct / encoding) to reach it"
    }
  ],
  "summary": "state of coverage in behavioural terms"
}
"""


SURFACE_SYSTEM = """\
You are a security researcher hunting for unexplored attack surface.

You receive the target's source (excerpts of the darkest subsystems) and the
current behavioural coverage state. Coverage has PLATEAUED: mutation no longer
finds new edges. Your job is to propose ENTIRELY NEW input classes that drive
whole FEATURES the corpus has never triggered — features gated behind control
fields, rarely-used formats, alternate entry points, deeply nested structures,
extreme boundaries, encodings, or externally referenced resources.

For each surface, be specific about the PRECONDITION and the input recipe.
Respond ONLY with JSON matching the schema:
{
  "attack_surfaces": [
    {
      "name": "short label",
      "feature": "the behaviour this unlocks",
      "entry_point": "function or path that handles it",
      "precondition": "what must be true for the target to enter this code",
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
    feature: str = ""
    precondition: str = ""

    def render(self) -> str:
        fns = ", ".join(self.functions) if self.functions else "(unspecified)"
        parts = []
        if self.feature:
            parts.append(f"FEATURE: {self.feature}")
        parts.append(f"functions: {fns}")
        if self.precondition:
            parts.append(f"precondition: {self.precondition}")
        parts.append(f"inputs: {self.hint} ({self.reasoning})")
        return "\n    ".join(parts)


@dataclass
class AttackSurface:
    name: str = ""
    entry_point: str = ""
    rationale: str = ""
    input_description: str = ""
    feature: str = ""
    precondition: str = ""


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


def _function_block(snap: CoverageSnapshot) -> str:
    """Behaviour-oriented coverage context for prompts, best source first:
    1. feature/subsystem gap analysis from the llvm-cov digest (+ source)
    2. crude grep over FUZZ_SOURCE_DIR
    """
    oc = getattr(snap, "ossfuzz_coverage", None)
    if oc is not None and (oc.functions or oc.files):
        return behavior_brief(oc, CONFIG.source_dir,
                              harness_file=CONFIG.harness_file)
    fns = _collect_function_names()
    return "\n".join(fns) if fns else "(no source dir configured)"


def map_coverage(llm: LLM, snap: CoverageSnapshot) -> dict:
    """LLM-assisted mapping of coverage to functionality. Returns raw dict."""
    top = snap.top_seeds(CONFIG.max_seed_samples)
    seed_lines = "\n".join(
        f"- {s.name}: {len(s.edges)} edges, first bytes: {s.data[:48]!r}"
        for s in top
    )
    fn_block = _function_block(snap)

    user = f"""\
{fn_block}

COVERAGE SNAPSHOT:
{snap.render_context()}
- top seeds (these exist TODAY; your recipes must go BEYOND their shapes):
{seed_lines}
"""

    return llm.chat_json(MAPPER_SYSTEM, user)


def propose_surfaces(llm: LLM, snap: CoverageSnapshot, prior_summary: str = "") -> list[AttackSurface]:
    """Plateau mode: propose wholly new attack surfaces, grounded in dark code."""
    oc = getattr(snap, "ossfuzz_coverage", None)
    dark = behavior_brief(oc, CONFIG.source_dir, harness_file=CONFIG.harness_file) if oc is not None else ""
    if not dark:
        dark = _function_block(snap)

    top = snap.top_seeds(CONFIG.max_seed_samples)
    seed_lines = "\n".join(
        f"- {s.name}: {len(s.edges)} edges, first bytes: {s.data[:48]!r}"
        for s in top
    )
    user = f"""\
{dark}

CURRENT CORPUS (top seeds):
{seed_lines}
{snap.render_context()}

PREVIOUS CYCLES SUMMARY:
{prior_summary or "(first cycle)"}
"""
    data = llm.chat_json(SURFACE_SYSTEM, user)
    return [
        AttackSurface(
            name=a.get("name", ""),
            entry_point=a.get("entry_point", ""),
            rationale=a.get("rationale", ""),
            input_description=a.get("input_description", ""),
            feature=a.get("feature", ""),
            precondition=a.get("precondition", ""),
        )
        for a in data.get("attack_surfaces", [])
    ][:6]
