"""Agentic ReAct loop: the LLM decides the next action via tool calls.

The orchestrator no longer prescribes the pipeline. It gives the agent a
mission brief (coverage state, plateau status, past failures) and a toolset;
the agent reasons over each tool result and chooses its own next action.
Deterministic guardrails cap the loop and force finalize_cycle.
"""

from __future__ import annotations

import json
import time
from typing import Any

from .config import CONFIG
from .coverage import analyze
from .memory import Memory
from .tools import TOOL_IMPLS, TOOL_SCHEMAS


AGENT_SYSTEM = """\
You are the orchestrator agent of an LLM-guided fuzzing system built on AFL++.
You run one CYCLE now, autonomously, by choosing which tools to call and when.

MISSION: reach DARK CODE — subsystems and features of the target that the
current campaign never triggers. You are not maximising raw edge count; you
are unlocking whole behaviours that are currently unreachable by mutation.

THINK IN TERMS OF BEHAVIOUR AND FEATURES, NOT EDGES:
- Do not just ask "what edges are missing". Ask "what does this program DO
  that we never see it do?" — name the untriggered capability in words.
- Read the source you are given. For each dark subsystem, identify the exact
  PRECONDITION that gates entry (a control/option flag, a required construct
  in the input, an external reference, an encoding, a size/alignment
  boundary, a multi-chunk split) and produce the INPUT that satisfies it.
- Fuzz inputs are usually STRUCTURED: a header of control fields then a body.
  A feature is often gated by a header bit/field, not the body alone.
- Prefer inputs that are VALID enough to drive the target deep (so the gated
  code actually runs) over random bytes.

YOUR TOOLSET:
- get_corpus_coverage: real edge data for the current corpus.
- hunt: DIRECTED hunt — the primary weapon. It picks unreached in-scope
  features, reads their source, constructs an input that reaches each one,
  validates against the live campaign, and STOPS as soon as a feature is
  unlocked. It loops over several dark features by itself; if all fail it
  reports which preconditions were wrong.
- map_coverage_to_functionality: reads the coverage digest + source, and
  returns DARK FEATURES with their gating preconditions and input recipes.
- propose_new_attack_surfaces: whole new input classes when plateaued.
- generate_inputs(strategy): LLM-authored candidates ("neighbor"/"surface").
- control_sweep: convert a feature hypothesis into concrete inputs by setting
  the harness HEADER control field on real seeds, keeping their body valid. Use
  this for features gated behind header bits/control fields — byte mutation
  almost never sets those coherently. (The header layout is configured per
  target; read the harness source to know which field to set.)
- validate_candidates: ground truth — a candidate is accepted ONLY if it adds
  edges the ENTIRE live campaign does not already have.
- finalize_cycle(summary): persist results. REQUIRED to end the cycle.

DECISION GUIDELINES (you decide, these are not hard rules):
- Ground yourself first: get_corpus_coverage before reasoning.
- THEN PREFER hunt: it is the whole point — reach specific unreached features
  by reasoning about their code, not by guessing. Call hunt once or twice.
- hunt stops early on success. If it reports "unlocked", you are DONE for this
  cycle — call finalize_cycle immediately (the fuzzer will mutate the rest).
- If hunt finds nothing, that is real signal: read its "attempts" and the
  failed preconditions, then call it again with a different max_targets, or
  use control_sweep for a header-gated feature, or map_coverage_to_functionality
  to get fresh recipes. Change your precondition each round; never repeat it.
- Budget your effort: a few well-targeted, validated attempts beat many blind
  ones. When satisfied, call finalize_cycle with an honest summary of which
  FEATURE you tried to unlock and whether it worked.

CONTEXT YOU WILL RECEIVE: coverage numbers, dark-feature analysis, plateau
status from past cycles, and hypotheses that previously failed. Use it to avoid
repeating failed strategies.
"""


class AgentError(RuntimeError):
    pass


def build_brief(mem: Memory, snapshot) -> str:
    """Initial user message: everything the agent needs to plan its cycle."""
    plateau = mem.is_plateaued()
    recent = mem.summary()
    failed = getattr(mem, "_data", {}).get("failed_hypotheses", [])
    failed_txt = "\n".join(f"- {h}" for h in failed[-10:]) if failed else "(none)"
    # Durable pre-campaign research notes (codebase + harness understanding).
    try:
        from .research import research_brief
        notes = research_brief()
    except Exception:
        notes = ""
    notes_block = f"\n{notes}\n" if notes else ""
    return f"""\
CYCLE BRIEF — {time.strftime('%Y-%m-%d %H:%M:%S')}

CORPUS: {len(snapshot.seeds)} seeds, {snapshot.num_edges} edges currently covered.
{notes_block}
PLATEAU STATUS: {"PLATEAUED — recent cycles produced almost no new edges; mutations of existing seeds are futile. Strongly consider propose_new_attack_surfaces." if plateau else "not plateaued — near-miss exploration is viable."}

RECENT CYCLES:
{recent}

PREVIOUSLY FAILED HYPOTHESES (do not repeat these verbatim):
{failed_txt}

Begin your cycle. Ground yourself with tools, iterate, and call
finalize_cycle when done. You have at most {CONFIG.agent_max_steps} actions.
"""


def run_agentic_cycle(llm, mem: Memory) -> dict:
    """One full agent-driven cycle. Returns a report dict."""
    CONFIG.ensure_dirs()

    # Ground-truth snapshot for the brief (cheap: also warms up ctx lazily)
    snapshot = analyze()
    ctx: dict[str, Any] = {"memory": mem, "mode": "agentic"}
    from . import orchestrator as _orc_mod
    _orc_mod._cycle_ctx = ctx

    messages: list[dict] = [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": build_brief(mem, snapshot)},
    ]

    tool_trace: list[dict] = []
    finalized = False
    steps = 0
    t0 = time.time()

    while steps < CONFIG.agent_max_steps:
        steps += 1
        resp = llm.raw_chat(messages, tools=TOOL_SCHEMAS)
        msg = resp.choices[0].message

        # No tool call => agent wants to talk; nudge it to act or finalize.
        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content or ""})
            messages.append({
                "role": "user",
                "content": (
                    "Respond only by calling tools. If you are done, call "
                    "finalize_cycle with your summary."
                ),
            })
            continue

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                tc.model_dump() if hasattr(tc, "model_dump") else tc
                for tc in msg.tool_calls
            ],
        })

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            impl = TOOL_IMPLS.get(name)
            if impl is None:
                observation = {"error": f"unknown tool {name}"}
            else:
                if name == "generate_inputs":
                    ctx["generated_total"] = ctx.get("generated_total", 0) + CONFIG.gen_batch_size
                try:
                    observation = impl(llm, ctx, **args)
                except Exception as e:  # tool failure must not kill the cycle
                    observation = {"error": f"{type(e).__name__}: {e}"}

            if name == "finalize_cycle":
                finalized = True

            tool_trace.append({"step": steps, "tool": name, "args": args})
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(observation, default=str)[:8000],
            })

        if finalized:
            break

    if not finalized:
        # Guardrail: force-finalize so memory is always updated.
        try:
            TOOL_IMPLS["finalize_cycle"](llm, ctx,
                                         summary="(forced) agent hit step budget")
        except Exception:
            pass
        forced = True
    else:
        forced = False

    return {
        "mode": "agentic",
        "steps": steps,
        "forced_finalize": forced,
        "tool_trace": tool_trace,
        "accepted": ctx.get("accepted_this_cycle", 0),
        "new_edges": ctx.get("new_edges_this_cycle", 0),
        "candidates": ctx.get("generated_total", 0),
        "duration_sec": round(time.time() - t0, 1),
    }
