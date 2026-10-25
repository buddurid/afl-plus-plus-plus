"""Thin OpenAI chat-completion wrapper with JSON-mode helpers."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

from .config import CONFIG


class LLMError(RuntimeError):
    pass


class LLM:
    """Minimal OpenAI client wrapper (uses the official `openai` package)."""

    def __init__(self) -> None:
        if not CONFIG.openai_api_key:
            raise LLMError(
                "OPENAI_API_KEY is not set. Export it before running the agent."
            )
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise LLMError(
                "The 'openai' package is required: pip install openai"
            ) from e

        self._client = OpenAI(
            api_key=CONFIG.openai_api_key,
            base_url=CONFIG.openai_base_url or None,
        )
        self.model = CONFIG.openai_model

    def chat(
        self,
        system: str,
        user: str,
        json_mode: bool = False,
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": (
                CONFIG.llm_temperature if temperature is None else temperature
            ),
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = self._client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content or ""
        if not content.strip():
            raise LLMError("LLM returned empty response")
        return content

    def raw_chat(self, messages: list[dict], tools: Optional[list] = None,
                 temperature: Optional[float] = None) -> Any:
        """Full-control chat call for the agentic loop (returns raw response
        so the caller can inspect tool_calls)."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": (
                CONFIG.llm_temperature if temperature is None else temperature
            ),
        }
        if tools:
            kwargs["tools"] = tools
        return self._client.chat.completions.create(**kwargs)

    def chat_json(self, system: str, user: str, **kw: Any) -> Any:
        raw = self.chat(system, user, json_mode=True, **kw)
        return _extract_json(raw)


def _extract_json(raw: str) -> Any:
    """Parse JSON, tolerating code fences or stray text around it."""
    text = raw.strip()
    # strip ```json ... ``` fences
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # last resort: first {...} or [...] block
        for opener, closer in (("{", "}"), ("[", "]")):
            start = text.find(opener)
            end = text.rfind(closer)
            if start != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    continue
        raise LLMError(f"Could not parse JSON from LLM response:\n{raw[:500]}")


def get_llm() -> LLM:
    return LLM()
