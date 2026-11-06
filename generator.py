"""Generator Agent: LLM produces candidate inputs (neighbor or surface mode)."""

from __future__ import annotations

import base64
import binascii
import os
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


TARGET_SYSTEM = """\
You generate fuzzing inputs that make execution REACH specific uncovered code.

You are given the uncovered functions and their source, the fuzz HARNESS (plus
the local helpers it #includes — together they DEFINE the input format), and
corpus inputs that are already format-correct.

Method:
  1. Read the harness and helpers. Trace, byte by byte, how an input is
     consumed: control fields, widths, byte order, length prefixes, framing,
     delimiters. The input may be a structured record, not a plain file.
  2. State the precondition that gates the uncovered code, in the program's own
     terms (a flag/field the harness reads, a required construct, an encoding,
     a size boundary, a chunk split).
  3. Build the WHOLE record the harness expects, then the payload that reaches
     the code. Keep the exact framing seen in the corpus examples.

Rules:
- Each input is at most {max_size} bytes.
- Inputs MUST be well-formed for the harness (every field present and correctly
  encoded); a mis-framed input never parses far enough.
- Vary the construct you use across inputs, not just bytes.

Respond ONLY with JSON:
{{
  "precondition": "the condition you believe gates this code",
  "inputs": [
    {{"reasoning": "why this input reaches the target", "data_base64": "base64 bytes"}}
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
    """Render seeds with a bounded byte-level preview AND base64.

    The escaped preview lets the LLM see the real framing (control fields,
    delimiters, length prefixes) that base64 hides — essential when the input
    format uses non-printable structure. Previews are capped so prompts stay
    within provider request-time limits.
    """
    lines = []
    budget = CONFIG.seed_preview_bytes
    for s in seeds:
        data = s.data[:CONFIG.max_input_size]
        head = data[:budget]
        b64 = base64.b64encode(head).decode()
        esc = head.decode("latin-1").encode("unicode_escape").decode()
        more = "" if len(data) <= budget else f" …(+{len(data)-budget} bytes)"
        lines.append(
            f"--- seed {s.name} ({len(s.edges)} edges, {len(data)} bytes) ---\n"
            f"first bytes (escaped, \\\\xNN = non-printable): {esc}{more}\n"
            f"base64(first {len(head)}B): {b64}")
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

INPUTS THE CORPUS ALREADY CONTAINS (do not reproduce these shapes):
{_render_seeds(snap.top_seeds(min(3, CONFIG.seed_preview_count)))}

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


def _harness_source() -> str:
    """Return the fuzz harness source together with the local sources it
    #includes, so the LLM can deduce the exact input format (header fields,
    framing, any length prefixes) by reading the helper functions the harness
    calls.

    Nothing about the format is hardcoded: the harness's own `#include "..."`
    directives are followed (see agent/harness.py), which pulls in the local
    helper definitions. Bounded in size (FUZZ_HARNESS_CONTEXT_BYTES) so the
    generation request stays within provider request-time limits.
    """
    from .harness import load_harness_context

    text = load_harness_context(CONFIG.harness_file,
                                total_budget=CONFIG.harness_context_bytes)
    return text or ""


def generate_for_target(llm: LLM, target: dict, snap: CoverageSnapshot,
                        count: int | None = None) -> tuple[str, list[Candidate]]:
    """Generate inputs aimed at ONE dark subsystem/feature.

    `target` comes from featurizer.dark_feature_targets(): it carries the
    uncovered functions, their source excerpt, and the coverage %. Returns
    (precondition, candidates).
    """
    count = count or CONFIG.gen_batch_size
    ep = ", ".join(target.get("entry_points", [])) or "(none identified)"
    unreached = ", ".join(target.get("unreached_functions", [])) or "(none)"
    harness = _harness_source()
    seeds_block = _render_seeds(snap.top_seeds(min(4, CONFIG.seed_preview_count)))

    base = f"""\
TARGET SUBSYSTEM: {target.get('file', '?')}  ({target.get('pct', 0)}% line coverage, \
{target.get('dark_lines', 0)} dark lines)
UNCOVERED ENTRY POINTS: {ep}
UNREACHED FUNCTIONS: {unreached}

SOURCE OF THE UNCOVERED CODE (read this carefully):
{target.get('source', '(no source excerpt available)')}
"""
    tail = f"""\
INPUTS THE CORPUS ALREADY CONTAINS (your input must go BEYOND these shapes):
{seeds_block}

Construct {count} inputs that reach the uncovered code above.
"""
    harness_block = ""
    if harness:
        harness_block = (f"FUZZ HARNESS + INCLUDED HELPERS (this defines the "
                         f"input format — read every byte it consumes):\n"
                         f"{harness}\n")

    # Prompt variants, escalating DOWN to a minimal prompt. Providers (esp.
    # reasoning endpoints) spiral and return an EMPTY body on rich prompts —
    # and the usable prompt size can collapse drastically (e.g. from ~8 KB to
    # ~1.4 KB) during provider degradation. So try the richest informative
    # prompt first, then progressively smaller ones (smaller system too), and
    # take the first variant that yields candidates.
    full_sys = TARGET_SYSTEM.format(max_size=CONFIG.max_input_size, count=count)
    small_sys = ('You generate fuzzing inputs. Reply ONLY with JSON '
                 '{"precondition":str,"inputs":[{"reasoning":str,"data_base64":str}]}.')
    short_source = (target.get("source", "") or "")[:CONFIG.target_source_bytes // 2]
    one_seed = _render_seeds(snap.top_seeds(1))
    variants: list[tuple[str, str]] = [(full_sys, f"{base}\n{harness_block}\n{tail}")]
    if harness:
        variants.append((full_sys, f"{base}\n{tail}"))
    variants.append((
        small_sys,
        f"UNREACHED: {unreached}\nSOURCE:\n{short_source}\n\n"
        f"CORPUS EXAMPLE:\n{one_seed}\n\n"
        f"Construct {count} inputs that reach this code. Keep the same framing "
        f"as the corpus example. Reply JSON."))
    if harness:
        variants.append((
            small_sys,
            f"HARNESS (defines the input format):\n{harness[:800]}\n\n"
            f"UNREACHED: {unreached}\nConstruct {count} inputs. Reply JSON."))

    last: Exception | None = None
    import time

    deadline = time.monotonic() + CONFIG.gen_budget_sec
    for sys, user in variants:
        if time.monotonic() > deadline:
            break
        try:
            # attempts=2: `chat` already fans out across all fallback models,
            # so the outer variants loop is the real retry. Deeper nesting
            # multiplies HTTP calls and can stall a whole cycle.
            data = llm.chat_json(sys, user, attempts=2, deadline=deadline)
        except Exception as e:  # noqa: BLE001 - provider errors are opaque
            last = e
            continue
        if not isinstance(data, dict):
            continue
        out: list[Candidate] = []
        for item in data.get("inputs", []):
            raw = _decode(item.get("data_base64", ""))
            if raw is not None:
                out.append(Candidate(data=raw, reasoning=item.get("reasoning", "")))
        if out:
            return data.get("precondition", "") or "", out

    if last is not None:
        import logging

        logging.getLogger(__name__).warning(
            "generate_for_target(%s) produced no candidates: %s: %s",
            target.get("file", "?"), type(last).__name__, str(last)[:200],
        )
    return "", []
