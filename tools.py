"""Tool layer: wraps the deterministic modules as LLM-callable tools.

Every tool returns (json_serializable_result, observation_string). The
observation string is what the LLM sees in its transcript.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from .config import CONFIG
from .coverage import analyze
from .generator import Candidate, generate_for_surface, generate_neighbors
from .mapper import Hypothesis, map_coverage, propose_surfaces
from .memory import Memory
from .validator import validate_candidates

# --------------------------------------------------------------------------- #
# Tool implementations. Each takes (llm, ctx) plus tool args, returns
# a JSON-serializable result for the transcript.
# --------------------------------------------------------------------------- #


def _tool_get_coverage(llm, ctx: dict, **args) -> Any:
    """Run showmap over the whole corpus and snapshot coverage."""
    snap = analyze()
    ctx["snapshot"] = snap
    top = snap.top_seeds(5)
    return {
        "seeds": len(snap.seeds),
        "total_edges": snap.num_edges,
        "top_seeds": [
            {"name": s.name, "edges": len(s.edges), "head": s.data[:32].hex()}
            for s in top
        ],
    }


def _tool_map_coverage(llm, ctx: dict, **args) -> Any:
    """LLM analysis: what functionality is covered, what is a near miss."""
    snap = ctx.get("snapshot")
    if snap is None:
        snap = analyze()
        ctx["snapshot"] = snap
    mapping = map_coverage(llm, snap)
    ctx["hypotheses"] = [
        Hypothesis(
            functions=h.get("functions", []),
            reasoning=h.get("reasoning", ""),
            hint=h.get("hint", ""),
        )
        for h in mapping.get("near_miss", [])
    ]
    ctx["coverage_summary"] = mapping.get("summary", "")
    return mapping


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
            "name": "validate_candidates",
            "description": (
                "Replay generated candidates through the target and keep only "
                "inputs that add new coverage. Call after generating; call again "
                "if you generate more."
            ),
            "parameters": {"type": "object", "properties": {}},
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
    "validate_candidates": _tool_validate,
    "finalize_cycle": _tool_finalize_cycle,
}
