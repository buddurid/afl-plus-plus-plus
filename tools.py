"""Tool layer: wraps the deterministic modules as LLM-callable tools.

Every tool returns (json_serializable_result, observation_string). The
observation string is what the LLM sees in its transcript.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from typing import Any, Callable

from .config import CONFIG
from .coverage import analyze
from .featurizer import _short, dark_feature_targets
from .generator import (Candidate, generate_for_surface, generate_for_target,
                        generate_neighbors)
from .mapper import Hypothesis, map_coverage, propose_surfaces
from .memory import Memory
from .structured import control_sweep
from .validator import validate_candidates
from . import cover_replay

# --------------------------------------------------------------------------- #
# Tool implementations. Each takes (llm, ctx) plus tool args, returns
# a JSON-serializable result for the transcript.
# --------------------------------------------------------------------------- #


def _tool_get_coverage(llm, ctx: dict, **args) -> Any:
    """Run showmap over the whole corpus and snapshot coverage."""
    snap = analyze()
    ctx["snapshot"] = snap
    top = snap.top_seeds(5)
    result = {
        "seeds": len(snap.seeds),
        "total_edges": snap.num_edges,
        "top_seeds": [
            {"name": s.name, "edges": len(s.edges), "head": s.data[:32].hex()}
            for s in top
        ],
    }
    oc = getattr(snap, "ossfuzz_coverage", None)
    if oc is not None and oc.totals:
        result["function_coverage"] = {
            metric: v for metric, v in oc.totals.items() if metric in
            ("functions", "lines", "branches")
        }
        result["never_executed_functions_sample"] = sorted(oc.zero_cov_functions)[:20]
    return result


def _tool_map_coverage(llm, ctx: dict, **args) -> Any:
    """Functionality mapping: dark features + near misses.

    If the pre-campaign research step already produced notes, return those
    INSTANTLY instead of spending ~6 min re-deriving the same thing with an
    LLM call (which previously ate the whole cycle budget before hunt ran).
    """
    snap = ctx.get("snapshot")
    if snap is None:
        snap = analyze()
        ctx["snapshot"] = snap

    cached = _cached_research_mapping(snap)
    if cached is not None:
        mapping = cached
    else:
        mapping = map_coverage(llm, snap)
    ctx["hypotheses"] = [
        Hypothesis(
            functions=h.get("functions", []),
            reasoning=h.get("reasoning", ""),
            hint=h.get("hint", ""),
            feature=h.get("feature", ""),
            precondition=h.get("precondition", ""),
        )
        for h in mapping.get("near_miss", [])
    ]
    ctx["coverage_summary"] = mapping.get("summary", "")
    return mapping


def _cached_research_mapping(snap) -> dict | None:
    """Build a map_coverage-style dict from the pre-campaign research notes."""
    try:
        from .research import load_research
        notes = load_research()
    except Exception:  # noqa: BLE001
        return None
    feats = (notes or {}).get("dark_features") or []
    if not feats:
        return None
    near_miss = [
        {
            "functions": f.get("functions", []),
            "feature": f.get("feature", ""),
            "precondition": f.get("precondition", ""),
            "hint": f.get("recipe", ""),
            "reasoning": f"dark feature ({f.get('confidence', '?')}) from "
                         "pre-campaign research",
        }
        for f in feats if isinstance(f, dict)
    ]
    summary = (
        f"{notes.get('program', '')} | Input format: {notes.get('input_format', '')} "
        f"| {len(near_miss)} dark features from pre-campaign research"
    )
    return {"summary": summary, "near_miss": near_miss,
            "_from_research": True}


def _tool_propose_surfaces(llm, ctx: dict, **args) -> Any:
    """LLM analysis: propose entirely new attack surfaces (plateau mode)."""
    snap = ctx.get("snapshot")
    if snap is None:
        snap = analyze()
        ctx["snapshot"] = snap
    surfaces = propose_surfaces(llm, snap, ctx.get("coverage_summary", ""))
    ctx["surfaces"] = surfaces
    return {
        "surfaces": [
            {
                "name": s.name,
                "entry_point": s.entry_point,
                "input_description": s.input_description,
            }
            for s in surfaces
        ]
    }


def _tool_generate_inputs(llm, ctx: dict, strategy: str = "neighbor", **args) -> Any:
    """Generate candidate inputs from current hypotheses or surfaces."""
    snap = ctx.get("snapshot")
    if snap is None:
        snap = analyze()
        ctx["snapshot"] = snap
    if strategy == "surface":
        surfaces = ctx.get("surfaces") or []
        if not surfaces:
            return {"error": "no surfaces proposed yet — call propose_new_attack_surfaces first"}
        cands = generate_for_surface(llm, snap, surfaces)
    else:
        hyps = ctx.get("hypotheses") or []
        cands = generate_neighbors(llm, snap, hyps)
    ctx.setdefault("candidates", []).extend(cands)
    return {"generated": len(cands), "strategy": strategy}


def _tool_hunt(llm, ctx: dict, max_targets: int | None = None,
               attempts_per_target: int | None = None, **args) -> Any:
    """Directed hunt: for each dark feature, construct an input that reaches it,
    RUN THE HARNESS NOW to confirm the function is reached, and only then stage
    it for injection.

    The mission loop: 1) current coverage, 2) pick an unreached feature
    (in-scope first), 3) build an input that reaches it, 4) EXECUTE the harness
    on that input immediately (function-level replay) and check whether the
    target function actually flipped from dark to covered; if not, move to the
    next feature and retry, within a bounded budget. Injection happens only
    after the target code is confirmed reached.
    """
    snap = ctx.get("snapshot")
    if snap is None:
        snap = analyze()
        ctx["snapshot"] = snap

    oc = getattr(snap, "ossfuzz_coverage", None)
    if oc is None:
        return {"error": "no llvm-cov digest — hunt needs function coverage "
                         "(set FUZZ_OSSFUZZ_REPORT_DIR)"}

    # Clamp to config: the LLM may ask for more than is affordable in one cycle.
    # Every attempt costs a full coverage replay (~5-15s), so cap the total work.
    max_targets = min(max_targets or CONFIG.hunt_max_targets, CONFIG.hunt_max_targets)
    attempts = min(attempts_per_target or CONFIG.hunt_attempts_per_target,
                   CONFIG.hunt_attempts_per_target)
    targets = dark_feature_targets(oc, CONFIG.source_dir, limit=max_targets,
                                   harness_file=CONFIG.harness_file)
    if not targets:
        return {"error": "no dark feature targets with source available"}

    # functions the campaign already reaches (bare names, matching the replay)
    already_covered = {_short(f.name) for f in oc.functions if f.count > 0}
    can_replay = cover_replay.available()

    known = set(snap.all_edges)
    tried: list[dict] = []
    accepted_total = 0
    new_edges_total = 0
    unlocked: list[str] = []
    deadline = time.monotonic() + CONFIG.hunt_budget_sec
    stopped_early = False

    for t in targets:
        if time.monotonic() > deadline:
            stopped_early = True
            break
        file_label = t.get("file", "?")
        # the specific dark functions we are trying to reach in this subsystem
        wanted = {_short(n) for n in t.get("unreached_functions", [])}
        target_accepted = 0
        last_precond = ""
        for attempt in range(1, attempts + 1):
            if time.monotonic() > deadline:
                stopped_early = True
                break
            precond, cands = generate_for_target(llm, t, snap)
            if precond:
                last_precond = precond
            if not cands:
                tried.append({"target": file_label, "attempt": attempt,
                              "generated": 0, "accepted": 0,
                              "reached": [], "precondition": last_precond})
                continue

            # --- IMMEDIATE CHECK: run the harness on these candidates now and
            # see whether any dark function of this subsystem got executed.
            reached: list[str] = []
            if can_replay:
                import tempfile
                tmp = tempfile.mkdtemp(prefix="hunt_", dir="/tmp/opencode"
                                       if os.path.isdir("/tmp/opencode") else None)
                paths = []
                for j, c in enumerate(cands[:CONFIG.hunt_replay_max]):
                    p = os.path.join(tmp, f"c{j:04d}.bin")
                    with open(p, "wb") as fh:
                        fh.write(c.data)
                    paths.append(p)
                try:
                    covered = cover_replay.run_coverage(paths)
                    reached = sorted(wanted & covered)
                except cover_replay.CoverageReplayError as e:
                    tried.append({"target": file_label, "attempt": attempt,
                                  "generated": len(cands), "accepted": 0,
                                  "reached": [], "error": str(e)[:160],
                                  "precondition": last_precond})
                    shutil.rmtree(tmp, ignore_errors=True)
                    continue
                finally:
                    shutil.rmtree(tmp, ignore_errors=True)

            ctx["generated_total"] = ctx.get("generated_total", 0) + len(cands)

            accepted: list = []
            before = set(known)
            # Immediate function-level gate: only run AFL's edge validator if the
            # harness actually reached a dark function of this subsystem.
            # (When replay is unavailable we fall back to pure edge coverage.)
            if reached or not can_replay:
                accepted, _verdicts, known = validate_candidates(cands, known)
                target_accepted += len(accepted)
                accepted_total += len(accepted)
                new_edges_total += len(known) - len(before)
                if accepted:
                    # Persist accepted inputs DURABLY (accepted_dir), not back
                    # into ctx["candidates"]: a later validate_candidates call
                    # would rewrite staging and delete these files (they add no
                    # edge once `known` has been updated).
                    os.makedirs(CONFIG.accepted_dir, exist_ok=True)
                    import time as _time
                    for path, _edges in accepted:
                        try:
                            with open(path, "rb") as fh:
                                data = fh.read()
                        except OSError:
                            continue
                        out = os.path.join(
                            CONFIG.accepted_dir,
                            f"hunt_{int(_time.time())}_{os.path.basename(path)}")
                        try:
                            with open(out, "wb") as fh:
                                fh.write(data)
                        except OSError:
                            continue
                    unlocked.append(
                        f"{file_label}: reached {', '.join(reached) or '(edge only)'} "
                        f"| precondition: {last_precond}")

            tried.append({"target": file_label, "attempt": attempt,
                          "generated": len(cands), "accepted": len(accepted),
                          "reached": reached, "precondition": last_precond})
            if target_accepted:
                break
        if target_accepted:
            break  # one unlocked feature is enough for this call

    ctx["accepted_this_cycle"] = ctx.get("accepted_this_cycle", 0) + accepted_total
    ctx["new_edges_this_cycle"] = ctx.get("new_edges_this_cycle", 0) + new_edges_total
    ctx.setdefault("all_known_edges", set()).update(known)

    if not accepted_total:
        fails = [f"{t['target']}: {t.get('precondition', '')}"
                 for t in tried if not t["accepted"] and t.get("precondition")]
        if fails:
            try:
                ctx["memory"].record_hypotheses(fails)
            except Exception:
                pass

    return {
        "targets_tried": len(tried),
        "accepted": accepted_total,
        "new_edges": new_edges_total,
        "unlocked": unlocked,
        "attempts": tried[-12:],
        "stopped_early": stopped_early,
        "note": ("STOPPED: feature unlocked and function reached"
                 if accepted_total else
                 ("budget exhausted — try fewer targets or a different feature"
                  if stopped_early else
                  "no feature unlocked — all preconditions tried failed")),
    }


def _tool_validate(llm, ctx: dict, **args) -> Any:
    """Replay staged candidates against the target; keep coverage-adding ones."""
    snap = ctx.get("snapshot")
    cands = ctx.pop("candidates", [])
    if not cands:
        return {"error": "no staged candidates to validate"}
    known = set(snap.all_edges) if snap else set()
    accepted, verdicts, known = validate_candidates(cands, known)
    new_edges = len(known) - len(snap.all_edges) if snap else 0
    ctx.setdefault("all_known_edges", set()).update(known)
    ctx["accepted_this_cycle"] = ctx.get("accepted_this_cycle", 0) + len(accepted)
    ctx["new_edges_this_cycle"] = ctx.get("new_edges_this_cycle", 0) + new_edges
    return {
        "candidates_validated": len(verdicts),
        "accepted": len(accepted),
        "new_edges": new_edges,
        "verdicts": [
            {"file": v.path, "accepted": v.accepted, "reason": v.reasoning}
            for v in verdicts[:20]
        ],
    }


def _configured_options() -> list[int]:
    """Parse FUZZ_HARNESS_OPTIONS (comma/space list of ints, e.g. '0 1 16 256')."""
    raw = CONFIG.harness_options
    out: list[int] = []
    for tok in raw.replace(",", " ").split():
        try:
            out.append(int(tok, 0))
        except ValueError:
            continue
    return out


def _default_control_field() -> tuple[int, int]:
    """Parse FUZZ_HARNESS_CONTROL_FIELD ('offset:width'); default 0:4."""
    raw = CONFIG.harness_control_field or "0:4"
    try:
        off, width = raw.split(":", 1)
        return int(off), int(width)
    except (ValueError, AttributeError):
        return 0, 4


def _tool_control_sweep(llm, ctx: dict, option_values: list | None = None,
                        offset: int | None = None, width: int | None = None,
                        **args) -> Any:
    """Deterministically set a harness header control field on existing seeds.

    Some harnesses gate whole features behind a control field in the input
    header (a mode selector, flags, an options value). Byte mutation rarely
    sets such fields coherently; this rewrites the configured field on real
    seeds, keeping the body valid. Which field and which values are target
    config (FUZZ_HARNESS_*), not hardcoded here.
    """
    snap = ctx.get("snapshot")
    if snap is None:
        snap = analyze()
        ctx["snapshot"] = snap
    header_bytes = CONFIG.harness_header_bytes
    if header_bytes <= 0:
        return {"error": "no harness header configured (FUZZ_HARNESS_HEADER_BYTES)"}
    vals = option_values or _configured_options()
    if not vals:
        return {"error": "no option_values given and none configured "
                         "(FUZZ_HARNESS_OPTIONS)"}
    try:
        vals = [int(v, 0) if isinstance(v, str) else int(v) for v in vals]
    except (TypeError, ValueError):
        return {"error": "option_values must be integers"}
    if offset is None or width is None:
        doff, dwidth = _default_control_field()
        offset = doff if offset is None else offset
        width = dwidth if width is None else width
    cands = control_sweep(snap.top_seeds(CONFIG.harness_sweep_seeds),
                          header_bytes, vals, offset=offset, width=width)
    ctx.setdefault("candidates", []).extend(cands)
    return {"generated": len(cands), "strategy": "control_sweep",
            "header_bytes": header_bytes, "control_field": f"{offset}:{width}",
            "option_values": vals}


def _tool_finalize_cycle(llm, ctx: dict, summary: str = "", **args) -> Any:
    """Record cycle results in memory. MUST be called to end the cycle."""
    import os
    import shutil
    import time

    mem: Memory = ctx["memory"]

    # Move validated candidates from staging to accepted_dir
    accepted_dir = CONFIG.accepted_dir
    os.makedirs(accepted_dir, exist_ok=True)
    moved = 0
    staging = CONFIG.staging_dir
    if os.path.isdir(staging):
        for f in sorted(os.listdir(staging)):
            src = os.path.join(staging, f)
            if os.path.isfile(src):
                shutil.move(src, os.path.join(accepted_dir, f"gen_{int(time.time())}_{f}"))
                moved += 1

    mem.record_cycle(
        mode=ctx.get("mode", "agentic"),
        num_candidates=ctx.get("generated_total", 0),
        num_accepted=ctx.get("accepted_this_cycle", 0),
        new_edges=ctx.get("new_edges_this_cycle", 0),
        total_edges=len(ctx.get("all_known_edges", set())),
        summary=summary or ctx.get("coverage_summary", ""),
    )
    return {"recorded": True, "accepted": ctx.get("accepted_this_cycle", 0),
            "new_edges": ctx.get("new_edges_this_cycle", 0), "moved_to_accepted": moved}


# --------------------------------------------------------------------------- #
# OpenAI tool schemas
# --------------------------------------------------------------------------- #

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_corpus_coverage",
            "description": (
                "Run every corpus seed through afl-showmap and snapshot current "
                "edge coverage. Always start the cycle with this."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "map_coverage_to_functionality",
            "description": (
                "Analyze which functionality is covered and identify near-miss "
                "code (close but unreached). Produces hypotheses for neighbor-mode "
                "generation."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_new_attack_surfaces",
            "description": (
                "Propose wholly NEW input classes / attack surfaces, for when "
                "coverage has plateaued and mutations of existing seeds are futile."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_inputs",
            "description": (
                "Generate candidate inputs. strategy='neighbor' mutates existing "
                "seeds toward near-miss hypotheses (needs map_coverage first). "
                "strategy='surface' creates inputs for new attack surfaces "
                "(needs propose_new_attack_surfaces first)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "strategy": {
                        "type": "string",
                        "enum": ["neighbor", "surface"],
                        "description": "Generation strategy to use",
                    }
                },
                "required": ["strategy"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hunt",
            "description": (
                "DIRECTED hunt (preferred): pick unreached in-scope features, "
                "construct an input that reaches each one, validate against the "
                "live campaign, and STOP as soon as a feature is unlocked. Loops "
                "over several dark features automatically. Use this instead of "
                "guessy generate_inputs when you want to reach specific dark code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "max_targets": {
                        "type": "integer",
                        "description": "how many dark features to try (default "
                                       "from config)",
                    },
                    "attempts_per_target": {
                        "type": "integer",
                        "description": "input-construction attempts per feature",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_candidates",
            "description": (
                "Replay staged candidates through the target and keep only "
                "inputs that add new coverage. Call after generate_inputs/"
                "control_sweep; call again if you generate more."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "control_sweep",
            "description": (
                "Deterministically rewrite a harness HEADER control field on "
                "existing seeds, keeping their body valid. Use this to switch ON "
                "features gated behind a header field — things byte mutation "
                "almost never sets coherently. Optionally pass option_values; "
                "otherwise the configured defaults are swept."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "option_values": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "control-field values to write",
                    },
                    "offset": {"type": "integer",
                               "description": "header offset (default 0)"},
                    "width": {"type": "integer",
                              "description": "field width in bytes (default 4)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize_cycle",
            "description": (
                "Record the cycle's results in long-term memory and end the "
                "cycle. You MUST call this exactly once when done."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "One-paragraph summary of what was tried and found",
                    }
                },
                "required": ["summary"],
            },
        },
    },
]

TOOL_IMPLS: dict[str, Callable] = {
    "get_corpus_coverage": _tool_get_coverage,
    "map_coverage_to_functionality": _tool_map_coverage,
    "propose_new_attack_surfaces": _tool_propose_surfaces,
    "generate_inputs": _tool_generate_inputs,
    "hunt": _tool_hunt,
    "control_sweep": _tool_control_sweep,
    "validate_candidates": _tool_validate,
    "finalize_cycle": _tool_finalize_cycle,
}
