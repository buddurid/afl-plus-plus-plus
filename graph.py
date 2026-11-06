"""LangGraph-based agentic fuzzing workflow.

Every step is an LLM decision. The graph provides only:
  agent node   -> LLM chooses a tool (bound via tool-calling)
  tools node   -> executes the chosen tool(s)
  router       -> back to agent, or END after finalize_cycle / budget

State is a MessagesState (full LLM transcript) plus a side-channel for
cycle metadata. A SqliteSaver checkpointer persists the conversation so
cycles can be inspected/resumed.

Requires: langgraph, langchain-core, langchain-openai (see requirements.txt)
"""

from __future__ import annotations

import json
import os
import time
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .config import CONFIG
from .coverage import analyze
from .memory import Memory
from .tools import TOOL_IMPLS
from .agent_loop import AGENT_SYSTEM, build_brief

# --------------------------------------------------------------------------- #
# Cycle log: step-by-step timing, enabled via FUZZ_CYCLE_LOG env var
# --------------------------------------------------------------------------- #


def _cycle_log(msg: str) -> None:
    path = os.environ.get("FUZZ_CYCLE_LOG", "")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(time.strftime("[%H:%M:%S] ") + msg + "\n")
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #


class FuzzState(TypedDict):
    # LLM transcript (messages accumulate via add_messages)
    messages: Annotated[list[AnyMessage], add_messages]
    # side-channel metadata, not shown to the LLM
    meta: dict[str, Any]


# --------------------------------------------------------------------------- #
# Tool executor node
# --------------------------------------------------------------------------- #


def make_tool_node(ctx: dict, llm=None):
    """ToolNode variant that routes calls through our TOOL_IMPLS so the
    deterministic modules (coverage, validator, memory) stay shared with
    the non-LangGraph modes."""

    def _execute(state: FuzzState) -> dict:
        last: AIMessage = state["messages"][-1]
        meta = state.get("meta", {})
        observations: list[ToolMessage] = []

        for tc in last.tool_calls:
            name = tc["name"] if isinstance(tc, dict) else tc.name
            args = tc["args"] if isinstance(tc, dict) else tc.args
            call_id = tc["id"] if isinstance(tc, dict) else tc.id
            _cycle_log(f"tool {name} start")

            impl = TOOL_IMPLS.get(name)
            if impl is None:
                result: Any = {"error": f"unknown tool {name}"}
            else:
                if name == "generate_inputs":
                    meta["generated_total"] = meta.get("generated_total", 0) + CONFIG.gen_batch_size
                    ctx["generated_total"] = meta["generated_total"]
                try:
                    result = impl(_LLMShim(llm), ctx, **args)
                except Exception as e:
                    result = {"error": f"{type(e).__name__}: {e}"}
            _cycle_log(f"tool {name} done ({type(result).__name__})")

            if name == "finalize_cycle":
                meta["finalized"] = True

            meta["steps"] = meta.get("steps", 0) + 1
            meta.setdefault("tool_trace", []).append(
                {"step": meta["steps"], "tool": name, "args": args}
            )

            observations.append(
                ToolMessage(
                    content=json.dumps(result, default=str)[:8000],
                    tool_call_id=call_id,
                )
            )

        return {"messages": observations, "meta": meta}

    return _execute


class _LLMShim:
    """The mapper/generator tools expect an object with chat_json(). Inside
    the LangGraph workflow those sub-LLM calls go through the same chat model
    (real or injected for tests) in JSON mode.

    For the REAL langchain model we delegate to agent.llm.LLM.chat_json, which
    has the model-fallback chain, empty-response retry, and hardened JSON
    extraction. Invoking the single langchain model directly (as before) gave
    up on the provider's intermittent empty responses and produced len=0 JSON
    errors. Injected/fake models (tests) keep the direct path.
    """

    def __init__(self, llm):
        self._llm = llm
        self._robust = None
        # only the real langchain ChatOpenAI should route through the robust
        # multi-model client; injected fakes must keep calling invoke().
        if llm is not None and _is_real_chat_model(llm):
            try:
                from .llm import LLM
                self._robust = LLM()
            except Exception:
                self._robust = None

    def chat_json(self, system: str, user: str, **kw: Any) -> Any:
        if self._robust is not None:
            return self._robust.chat_json(system, user, **kw)
        bound = self._llm.bind(response_format={"type": "json_object"})
        resp = _llm_retry(bound, [SystemMessage(content=system), _human(user)])
        from .llm import _extract_json
        return _extract_json(resp.content)


def _is_real_chat_model(llm) -> bool:
    """True for a real langchain ChatOpenAI (not a test double)."""
    try:
        from langchain_openai import ChatOpenAI as _CO
        return isinstance(llm, _CO)
    except Exception:
        return False


from langchain_core.messages import HumanMessage as _human  # noqa: E402


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #


def _make_chat_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=CONFIG.openai_model,
        api_key=CONFIG.openai_api_key,
        base_url=CONFIG.openai_base_url or None,
        temperature=CONFIG.llm_temperature,
        timeout=200,          # hard cap per LLM call (vyce Cloudflare: 120s gate)
        max_retries=1,        # rely on _llm_retry for explicit retries
    )


def _llm_retry(llm, messages, attempts: int = 3, backoff: float = 25.0):
    """Invoke the LLM, retrying transient gateway errors and rate limits.

    Vyce (Cloudflare-fronted) kills requests after 120s with a 524 and
    throttles bursts with 429/402/403. Rate limits get more, shorter
    retries; 5xx/timeouts get fewer, longer ones.
    Worst case per invocation: ~3 calls + backoffs ≈ 5-6 min.
    """
    from openai import APIError, APIConnectionError, APITimeoutError

    rate_limited = {402, 403, 429}
    max_attempts = 4

    for i in range(max_attempts):
        try:
            return llm.invoke(messages)
        except (APITimeoutError, APIConnectionError) as e:
            if i == attempts - 1:
                raise
            time.sleep(backoff * (i + 1))
        except APIError as e:
            status = getattr(getattr(e, "response", None), "status_code", None) or 0
            is_limit = status in rate_limited
            if (is_limit and i == max_attempts - 1) or (not is_limit and (status < 500 or i == attempts - 1)):
                raise
            time.sleep(30 if is_limit else backoff * (i + 1))
    raise RuntimeError("unreachable")


def make_agent_node(llm: ChatOpenAI):
    def agent(state: FuzzState) -> dict:
        meta = dict(state.get("meta", {}))
        meta["turns"] = meta.get("turns", 0) + 1
        # keep transcript bounded
        trimmed = trim_messages(
            state["messages"],
            max_tokens=60000,
            token_counter=len,  # message count as cheap proxy
            strategy="last",
            start_on="human",
            allow_partial=False,
        )
        t0 = time.time()
        try:
            resp = _llm_retry(llm, trimmed)
        except Exception as e:
            # permanent LLM failure: gracefully finalize the cycle instead of
            # crashing the run
            _cycle_log(f"agent turn FAILED: {type(e).__name__}")
            emergency = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "finalize_cycle",
                        "args": {"summary": f"LLM unavailable ({type(e).__name__} "
                                             f"{getattr(getattr(e, 'response', None), 'status_code', '?')}); "
                                             "cycle finalized without generation."},
                        "id": "emergency-finalize",
                        "type": "tool_call",
                    }
                ],
            )
            return {"messages": [emergency], "meta": meta}
        finally:
            _cycle_log(f"agent turn took {time.time() - t0:.1f}s")
        return {"messages": [resp], "meta": meta}

    return agent


def make_router(max_steps: int):
    def route(state: FuzzState) -> str:
        meta = state.get("meta", {})
        if meta.get("finalized"):
            return END
        # wall-clock ceiling: stop before the driver's hard cap so the cycle can
        # finalize gracefully instead of being SIGTERM'd mid-generation
        budget = CONFIG.cycle_budget_sec
        t0 = meta.get("t0")
        if budget and t0 and (time.time() - t0) >= budget:
            return END
        # hard ceiling on total agent turns, independent of tool steps, so a
        # chatty model that never calls a tool can't loop forever
        if meta.get("turns", 0) >= max_steps * 2 + 4:
            return END
        if meta.get("steps", 0) >= max_steps:
            return END
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "agent"  # nudge loop: agent talked without calling tools

    return route


def build_graph(llm: Optional[ChatOpenAI] = None, ctx: Optional[dict] = None,
                checkpointer=None):
    """Compile the fuzzing workflow graph.

        START -> agent <-> tools -> (router) -> agent ... -> END
    """
    llm = llm or _make_chat_model()
    ctx = ctx or {"memory": Memory()}
    llm_with_tools = llm.bind_tools(
        # bind by name via pydantic-free simple schemas
        _tool_specs()
    )

    g = StateGraph(FuzzState)
    g.add_node("agent", make_agent_node(llm_with_tools))
    g.add_node("tools", make_tool_node(ctx, llm=llm))

    g.add_edge(START, "agent")
    g.add_conditional_edges(
        "agent",
        make_router(CONFIG.agent_max_steps),
        {
            "tools": "tools",
            "agent": "agent",
            "agent_with_force": "agent",
            END: END,
        },
    )
    g.add_edge("tools", "agent")

    return g.compile(checkpointer=checkpointer)


def _tool_specs():
    """Minimal OpenAI-style schemas (same as TOOL_SCHEMAS from tools.py)."""
    from .tools import TOOL_SCHEMAS
    return TOOL_SCHEMAS


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def run_langgraph_cycle(mem: Memory, thread_id: str | None = None,
                        _llm_override=None) -> dict:
    CONFIG.ensure_dirs()
    thread_id = thread_id or f"cycle-{int(time.time())}"

    snapshot = analyze()
    ctx = {"memory": mem, "mode": "langgraph"}
    from . import orchestrator as _orc_mod
    _orc_mod._cycle_ctx = ctx

    graph = build_graph(llm=_llm_override, ctx=ctx, checkpointer=MemorySaver())

    initial: FuzzState = {
        "messages": [
            SystemMessage(content=AGENT_SYSTEM),
            _human(build_brief(mem, snapshot)),
        ],
        "meta": {"mode": "langgraph", "steps": 0, "tool_trace": [], "t0": time.time()},
    }

    t0 = time.time()
    final_state = graph.invoke(initial, config={"configurable": {"thread_id": thread_id}})
    meta = final_state["meta"]

    if not meta.get("finalized"):
        # force finalize through the tool impl directly
        TOOL_IMPLS["finalize_cycle"](None, ctx, summary="(forced) budget exhausted")

    return {
        "mode": "langgraph",
        "thread_id": thread_id,
        "steps": meta.get("steps", 0),
        "forced_finalize": not meta.get("finalized", False),
        "tool_trace": meta.get("tool_trace", []),
        "accepted": ctx.get("accepted_this_cycle", 0),
        "new_edges": ctx.get("new_edges_this_cycle", 0),
        "candidates": meta.get("generated_total", 0),
        "duration_sec": round(time.time() - t0, 1),
    }
