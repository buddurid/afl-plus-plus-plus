"""Generator Agent: LLM produces candidate inputs (neighbor or surface mode)."""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass

from .config import CONFIG
from .coverage import CoverageSnapshot, Seed
from .llm import LLM
from .mapper import AttackSurface, Hypothesis


NEIGHBOR_SYSTEM = """\
You are a fuzzing input generator. You are given existing seeds and coverage
hints about functionality that is CLOSE but not reached. Produce NEW input
files that are structural/semantic variants of the seeds, designed to hit
the hinted functionality: flip flags, change magic values, extend field
ranges, add boundary lengths (0, 1, -1, 0x7fffffff), alter encodings,
append/insert new sections.

Rules:
- Each input is at most {max_size} bytes.
- Inputs must be valid enough to parse, but push toward the uncovered logic.
- Diversity matters: do not output near-duplicates.

Respond ONLY with JSON:
{{
  "inputs": [
    {{
      "reasoning": "why this input might reach new code",
      "data_base64": "base64-encoded file content"
    }}
  ]
}}
Give exactly {count} inputs.
"""

SURFACE_SYSTEM = """\
You are a fuzzing input generator attacking an UNREACHED attack surface.
Coverage has plateaued; mutations of existing seeds no longer help.
You are given a description of a new attack surface (entry point, rationale,
input description) and examples of what the corpus already contains.

Produce inputs that match the attack surface description and are structurally
DIFFERENT from the existing corpus: new formats, new protocol states, new
field combinations, extreme boundary values.

Rules:
- Each input is at most {max_size} bytes.
- Inputs should be as valid as possible so the target's parsing runs deep.

Respond ONLY with JSON:
{{
  "inputs": [
    {{
      "reasoning": "why this input might reach the new surface",
      "data_base64": "base64-encoded file content"
    }}
  ]
}}
Give exactly {count} inputs.
"""


@dataclass
class Candidate:
    data: bytes
    reasoning: str


def _decode(data_b64: str) -> bytes | None:
    try:
        raw = base64.b64decode(data_b64, validate=True)
    except (binascii.Error, ValueError):
        return None
    if not raw or len(raw) > CONFIG.max_input_size:
        return None
    return raw


def _render_seeds(seeds: list[Seed]) -> str:
    lines = []
    for s in seeds:
        b64 = base64.b64encode(s.data[:CONFIG.max_input_size]).decode()
        lines.append(f"--- seed {s.name} ({len(s.edges)} edges) ---\nbase64: {b64}")
    return "\n".join(lines)


def generate_neighbors(
    llm: LLM, snap: CoverageSnapshot, hypotheses: list[Hypothesis]
) -> list[Candidate]:
    """Neighbor mode: variants of existing seeds targeting near-miss edges."""
    hints = "\n".join(f"{i+1}. {h.render()}" for i, h in enumerate(hypotheses))
    user = f"""\
EXISTING SEEDS:
{_render_seeds(snap.top_seeds(CONFIG.max_seed_samples))}

NEAR-MISS FUNCTIONALITY TO TARGET:
{hints or "(no specific hints — explore around the seeds)"}

Generate {CONFIG.gen_batch_size} new inputs as base64.
"""
    data = llm.chat_json(
        NEIGHBOR_SYSTEM.format(max_size=CONFIG.max_input_size, count=CONFIG.gen_batch_size),
        user,
    )
    out: list[Candidate] = []
    for item in data.get("inputs", []):
        raw = _decode(item.get("data_base64", ""))
        if raw is not None:
            out.append(Candidate(data=raw, reasoning=item.get("reasoning", "")))
    return out


def generate_for_surface(
    llm: LLM, snap: CoverageSnapshot, surfaces: list[AttackSurface]
) -> list[Candidate]:
    """Surface mode: structurally new inputs for a plateaued corpus."""
    per_surface = max(1, CONFIG.gen_batch_size // max(1, len(surfaces)))
    out: list[Candidate] = []

    for surface in surfaces:
        user = f"""\
ATTACK SURFACE:
name: {surface.name}
entry_point: {surface.entry_point}
rationale: {surface.rationale}
input description: {surface.input_description}

WHAT THE CORPUS ALREADY CONTAINS (do not reproduce these shapes):
{_render_seeds(snap.top_seeds(max(3, CONFIG.max_seed_samples // 2)))}

Generate {per_surface} new inputs as base64.
"""
        try:
            data = llm.chat_json(
                SURFACE_SYSTEM.format(max_size=CONFIG.max_input_size, count=per_surface),
                user,
            )
        except Exception:
            continue
        for item in data.get("inputs", []):
            raw = _decode(item.get("data_base64", ""))
            if raw is not None:
                out.append(Candidate(data=raw, reasoning=f"[{surface.name}] {item.get('reasoning', '')}"))
    return out
