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

            impl = TOOL_IMPLS.get(name)
            if impl is None:
                result: Any = {"error": f"unknown tool {name}"}
            else:
                if name == "generate_inputs":
                    meta["generated_total"] = meta.get("generated_total", 0) + CONFIG.gen_batch_size
                try:
                    result = impl(_LLMShim(llm), ctx, **args)
                except Exception as e:
                    result = {"error": f"{type(e).__name__}: {e}"}

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
    (real or injected for tests) in JSON mode."""

    def __init__(self, llm):
        self._llm = llm

    def chat_json(self, system: str, user: str, **kw) -> Any:
        bound = self._llm.bind(response_format={"type": "json_object"})
        resp = bound.invoke([SystemMessage(content=system), _human(user)])
        from .llm import _extract_json
        return _extract_json(resp.content)


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
    )


def make_agent_node(llm: ChatOpenAI):
    def agent(state: FuzzState) -> dict:
        # keep transcript bounded
        trimmed = trim_messages(
            state["messages"],
            max_tokens=60000,
            token_counter=len,  # message count as cheap proxy
            strategy="last",
            start_on="human",
            allow_partial=False,
        )
        resp = llm.invoke(trimmed)
        return {"messages": [resp]}

    return agent


def make_router(max_steps: int):
    def route(state: FuzzState) -> str:
        meta = state.get("meta", {})
        if meta.get("finalized"):
            return END
        if meta.get("steps", 0) >= max_steps:
            # budget exhausted -> force finalize next round
            force = SystemMessage(
                content="Step budget exhausted. Call finalize_cycle now with your summary."
            )
            return "agent_with_force"
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

    graph = build_graph(llm=_llm_override, ctx=ctx, checkpointer=MemorySaver())

    initial: FuzzState = {
        "messages": [
            SystemMessage(content=AGENT_SYSTEM),
            _human(build_brief(mem, snapshot)),
        ],
        "meta": {"mode": "langgraph", "steps": 0, "tool_trace": []},
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
