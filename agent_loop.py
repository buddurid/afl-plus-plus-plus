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

MISSION: maximize new edge coverage of the target by intelligently creating
new corpus inputs.

YOUR TOOLSET:
- get_corpus_coverage: execute every seed via afl-showmap, get real edge data.
- map_coverage_to_functionality: LLM analysis -> which functionality is
  covered, which is a NEAR MISS (close but unreached), with input hints.
- propose_new_attack_surfaces: LLM analysis -> wholly new input classes
  (use when near-miss exploration is exhausted / plateaued).
- generate_inputs(strategy): create candidates from hypotheses ("neighbor")
  or from new surfaces ("surface").
- validate_candidates: execute candidates, keep only coverage-adding ones.
  Ground truth. LLM guesses are worthless until validated.
- finalize_cycle(summary): persist results. REQUIRED to end the cycle.

DECISION GUIDELINES (you decide, these are not hard rules):
- Always ground yourself first: get_corpus_coverage before reasoning.
- Work in iterations: generate -> validate -> look at the verdicts -> adapt.
  If validation shows your hypothesis was wrong, change approach rather than
  generating more of the same.
- If a hypothesis yields nothing new, do not repeat it; pivot to surfaces.
- Budget your effort: a few well-targeted iterations beat many blind ones.
- When satisfied (or when nothing more is worth trying), call finalize_cycle
  with an honest summary of what worked and what did not.

CONTEXT YOU WILL RECEIVE: current coverage numbers, plateau status from past
cycles, and hypotheses that previously failed. Use it to avoid repeating
failed strategies.
"""


class AgentError(RuntimeError):
    pass


def build_brief(mem: Memory, snapshot) -> str:
    """Initial user message: everything the agent needs to plan its cycle."""
    plateau = mem.is_plateaued()
    recent = mem.summary()
    failed = getattr(mem, "_data", {}).get("failed_hypotheses", [])
    failed_txt = "\n".join(f"- {h}" for h in failed[-10:]) if failed else "(none)"
    return f"""\
CYCLE BRIEF — {time.strftime('%Y-%m-%d %H:%M:%S')}

CORPUS: {len(snapshot.seeds)} seeds, {snapshot.num_edges} edges currently covered.

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
