"""Pre-campaign codebase research: an LLM studies the target it will fuzz.

Before the campaign starts (and opportunistically between cycles) the agent
runs a dedicated RESEARCH step: it reads the fuzz harness, the sources it
#includes, the darkest uncovered source files, and the coverage digest, and
produces durable, structured NOTES about the codebase:

  * what the program does / its main capabilities
  * the input format the harness consumes (byte layout)
  * per-feature notes: which entry points exist, what gates them, and a recipe
    for the input that would drive them
  * which modules are in scope for THIS harness vs out of scope

The notes are written to CONFIG.research_path (JSON) and read back into every
agent cycle's brief, so the agent starts each cycle already knowing the target
instead of re-deriving it from scratch.

Target-agnostic: nothing here is project specific. Everything is derived from
the harness #include graph, the coverage digest, and the configured source dir.
The LLM calls are kept SMALL (chunked) because some providers return empty
bodies on large prompts.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from .config import CONFIG
from .featurizer import build_subsystems, extract_function_source, rank_subsystems
from .harness import load_harness_context
from .llm import LLM, LLMError, get_llm


RESEARCH_SYSTEM = """\
You are reverse-engineering a C/C++ program and its fuzz harness so that a
fuzzer can reach DARK CODE (features the current campaign never triggers).

You are given: the fuzz harness source (plus the local helpers it #includes),
excerpts of the program's darkest uncovered source files, and coverage numbers.

Produce concise, CONCRETE notes a fuzzing agent can act on. Do not speculate
beyond the code shown. Respond ONLY with JSON:
{
  "program": "one-paragraph description of what this program does",
  "input_format": "the exact byte layout the harness consumes: field widths, byte order, length prefixes, delimiters, framing, and where the payload starts",
  "capabilities": ["named feature/capability the program implements", ...]
}
Be brief. This is a SMALL response: at most 12 capabilities, one sentence each.
"""

RESEARCH_DARK_SYSTEM = """\
You are analyzing the FUZZ HARNESS and ONE uncovered source file to find DARK
CODE: capabilities the fuzzer is not reaching, and how to reach them.

Respond ONLY with JSON:
{
  "dark_features": [
    {
      "feature": "short name of an untriggered capability implemented in THIS file",
      "functions": ["function names in THIS file that implement it"],
      "precondition": "the exact condition that gates it (a flag/field the harness reads, a required input construct, an encoding, a size boundary, ...)",
      "recipe": "how to build an input that satisfies the precondition",
      "confidence": "high|medium|low"
    }
  ]
}
Rules:
- Focus on features THIS harness can plausibly reach through the input format.
- Prefer specifics over generality; keep every string SHORT (one line).
- Give at most __MAX_FEATURES__ dark_features for this file.
- If this file has no untriggered feature the harness can reach, return
  {"dark_features": []}.
- Respond with JSON only, no prose.
"""


def _load_notes(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def load_research(path: str | None = None) -> dict | None:
    """Read the durable research notes (or None if not produced yet)."""
    return _load_notes(path or CONFIG.research_path)


def research_brief(path: str | None = None, max_features: int = 8) -> str:
    """Compact text form of the notes, for injection into the agent brief."""
    notes = load_research(path)
    if not notes:
        return ""
    out = ["CODEBASE RESEARCH NOTES (from the pre-campaign study — use these; "
           "you do not need to re-derive the format or scope):", ""]
    if notes.get("program"):
        out.append(f"PROGRAM: {notes['program']}")
    if notes.get("input_format"):
        out.append(f"INPUT FORMAT: {notes['input_format']}")
    caps = notes.get("capabilities") or []
    if caps:
        out.append("CAPABILITIES: " + "; ".join(str(c) for c in caps[:12]))
    if notes.get("in_scope"):
        out.append("IN SCOPE (this harness): " + "; ".join(
            str(c) for c in notes["in_scope"][:12]))
    if notes.get("out_of_scope"):
        out.append("OUT OF SCOPE: " + "; ".join(
            str(c) for c in notes["out_of_scope"][:12]))
    feats = notes.get("dark_features") or []
    if feats:
        out.append("")
        out.append("DARK FEATURES TO TARGET (precondition -> recipe):")
        for f in feats[:max_features]:
            if not isinstance(f, dict):
                continue
            out.append(
                f"  - {f.get('feature','?')} [{f.get('confidence','?')}]"
                f"  fns={', '.join(str(x) for x in (f.get('functions') or [])[:4])}"
                f"\n      gate: {f.get('precondition','?')}"
                f"\n      recipe: {f.get('recipe','?')}")
    return "\n".join(out)


def _dark_source_excerpts(source_dir: str, oc, budget: int, n_subs: int = 3) -> str:
    """Source excerpts of the darkest in-scope subsystems, for the LLM to read."""
    if not source_dir or oc is None:
        return ""
    subs = build_subsystems(oc)
    driven, untouched = rank_subsystems(subs, CONFIG.harness_file)
    out: list[str] = []
    left = budget
    for s in (driven + untouched)[:n_subs]:
        src = extract_function_source(source_dir, s, budget=max(800, left // 2))
        if not src:
            continue
        out.append(f"===== {s.file} ({s.pct}% covered) =====\n{src}")
        left -= len(src)
        if left <= 0:
            break
    return "\n\n".join(out)


def run_research(llm: LLM | None = None, source_dir: str | None = None,
                 oc=None, max_features: int = 8) -> dict:
    """Run one research pass and persist the notes. Returns the notes dict.

    Resilient: if the LLM fails entirely, write a notes file with whatever
    deterministic facts we have (harness present, covered modules) so downstream
    code always has something to read.
    """
    llm = llm or get_llm()
    source_dir = source_dir or CONFIG.source_dir

    # Deterministic fallback facts (always available).
    base: dict[str, Any] = {
        "program": "",
        "input_format": "",
        "capabilities": [],
        "in_scope": [],
        "out_of_scope": [],
        "dark_features": [],
        "generated_at": time.time(),
    }
    if oc is not None:
        subs = build_subsystems(oc)
        driven, untouched = rank_subsystems(subs, CONFIG.harness_file)
        base["in_scope"] = [f"{s.file} ({s.pct}%)" for s in driven[:12]]
        base["out_of_scope"] = [f"{s.file} ({s.pct}%)" for s in untouched[:12]]

    harness = load_harness_context(CONFIG.harness_file,
                                   total_budget=CONFIG.harness_context_bytes)
    dark_src = _dark_source_excerpts(source_dir, oc, budget=CONFIG.research_source_bytes)

    notes = dict(base)

    # Call 1 (SMALL output): what the program is + the harness input format.
    # Kept separate because asking for the full body in one call makes every
    # model exhaust its output budget and return an empty response.
    try:
        data = llm.chat_json(
            RESEARCH_SYSTEM,
            f"FUZZ HARNESS + INCLUDED HELPERS:\n{harness or '(none)'}\n\n"
            f"DARK UNCOVERED SOURCE EXCERPTS:\n{dark_src or '(none)'}\n\n"
            f"COVERAGE: {getattr(oc, 'totals', '') if oc is not None else '(none)'}",
            attempts=3,
        )
        if isinstance(data, dict):
            for k in ("program", "input_format", "capabilities"):
                if data.get(k):
                    notes[k] = data[k]
    except Exception as e:  # noqa: BLE001 - keep deterministic fallback
        notes["_llm_error"] = f"{type(e).__name__}: {str(e)[:300]}"

    # Call 2..N (one per dark file, SMALL output): the dark features + recipes.
    # A compact context (program + input format) is passed instead of the full
    # harness source: the full source makes the prompt large enough that the
    # provider drops the connection, and the model already saw it in call 1.
    features = _research_dark_features(llm, source_dir, oc, notes,
                                       max_features=max_features)
    if features:
        notes["dark_features"] = features

    path = CONFIG.research_path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(notes, fh, indent=2)
    return notes


def _research_dark_features(llm: LLM, source_dir: str, oc, notes: dict,
                            max_features: int = 8, n_files: int = 3) -> list[dict]:
    """One small LLM call per darkest subsystem file -> dark features.

    Chunking is essential: a single call spanning all files asks the model to
    emit a large JSON object, which exhausts its output budget and yields an
    empty response on reasoning endpoints. Small per-file calls are reliable.
    """
    if oc is None or not source_dir:
        return []
    # Compact context: enough for the model to judge reachability without
    # re-sending the whole harness (which makes the request too large).
    ctx = (f"PROGRAM: {notes.get('program','')}\n"
           f"INPUT FORMAT: {notes.get('input_format','')}\n"
           f"CAPABILITIES: {'; '.join(str(c) for c in (notes.get('capabilities') or [])[:8])}")
    subs = build_subsystems(oc)
    driven, untouched = rank_subsystems(subs, CONFIG.harness_file)
    out: list[dict] = []
    seen: set[str] = set()
    for s in (driven + untouched)[:n_files]:
        src = extract_function_source(source_dir, s, budget=CONFIG.research_source_bytes // n_files)
        if not src:
            continue
        try:
            data = llm.chat_json(
                RESEARCH_DARK_SYSTEM.replace("__MAX_FEATURES__", str(max_features)),
                f"{ctx}\n\nUNCOVERED SOURCE FILE {s.file} ({s.pct}% covered):\n{src}",
                attempts=3,
            )
        except Exception:  # noqa: BLE001 - skip this file, keep going
            continue
        for f in (data.get("dark_features") if isinstance(data, dict) else None) or []:
            if not isinstance(f, dict):
                continue
            name = str(f.get("feature", "")).strip().lower()
            if not name or name in seen:
                continue
            seen.add(name)
            out.append(f)
        if len(out) >= max_features:
            break
    return out[:max_features]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="pre-campaign codebase research")
    ap.add_argument("--max-features", type=int, default=8)
    ap.add_argument("--status", action="store_true",
                    help="print existing notes and exit (no LLM call)")
    args = ap.parse_args()

    if args.status:
        notes = load_research()
        print(research_brief() if notes else "(no research notes yet)")
        return

    oc = None
    if CONFIG.ossfuzz_report_dir:
        from .ossfuzz_funcs import load_ossfuzz_coverage
        try:
            oc = load_ossfuzz_coverage()
        except Exception:
            oc = None
    t0 = time.time()
    notes = run_research(oc=oc, max_features=args.max_features)
    n = len(notes.get("dark_features") or [])
    print(f"research written to {CONFIG.research_path}: "
          f"{n} dark features, format={'yes' if notes.get('input_format') else 'no'} "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

