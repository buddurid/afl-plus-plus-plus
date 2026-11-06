"""Thin OpenAI chat-completion wrapper with JSON-mode helpers."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Optional

from .config import CONFIG


def _now() -> float:
    return time.monotonic()


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
            timeout=CONFIG.llm_timeout,
            max_retries=0,
        )
        self.model = CONFIG.openai_model
        # Fallback chain: the primary model first, then any configured
        # alternates. The provider's availability is model-specific and
        # changes minute to minute (one model 400s/empties while another
        # works), so try each in turn on failure.
        chain = [CONFIG.openai_model] + [
            m.strip() for m in CONFIG.llm_fallback_models.split(",")
            if m.strip() and m.strip() != CONFIG.openai_model
        ]
        self.models = chain

    def chat(
        self,
        system: str,
        user: str,
        json_mode: bool = False,
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
        deadline: Optional[float] = None,
    ) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        last: Exception | None = None
        for model in self.models:
            if deadline is not None and _now() > deadline:
                break
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": (
                    CONFIG.llm_temperature if temperature is None else temperature
                ),
                "max_tokens": max_tokens,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            try:
                resp = self._create_with_retry(kwargs, attempts=1)
            except Exception as e:  # noqa: BLE001 - provider errors are opaque
                last = e
                continue
            content = resp.choices[0].message.content or ""
            if content.strip():
                return content
            last = LLMError("LLM returned empty response")
        raise last or LLMError("LLM returned empty response")

    def _create_with_retry(self, kwargs: dict[str, Any], attempts: int = 3) -> Any:
        """Create a completion, retrying transient provider failures.

        Free endpoints emit 5xx / Cloudflare 524 (origin timeout) and
        connection errors under load. Back off and retry a bounded number of
        times; re-raise the last error.
        """
        import time

        last: Exception | None = None
        for i in range(max(1, attempts)):
            try:
                return self._client.chat.completions.create(**kwargs)
            except Exception as e:  # noqa: BLE001 - provider errors are opaque
                status = getattr(getattr(e, "response", None), "status_code", None)
                retryable = status in (408, 409, 425, 429, 500, 502, 503, 504,
                                       522, 524) or status is None
                if not retryable or i == attempts - 1:
                    raise
                last = e
                time.sleep(CONFIG.llm_retry_backoff * (i + 1))
        raise last  # pragma: no cover

    def raw_chat(self, messages: list[dict], tools: Optional[list] = None,
                 temperature: Optional[float] = None) -> Any:
        """Full-control chat call for the agentic loop (returns raw response
        so the caller can inspect tool_calls). Tries each fallback model."""
        last: Exception | None = None
        for model in self.models:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": (
                    CONFIG.llm_temperature if temperature is None else temperature
                ),
            }
            if tools:
                kwargs["tools"] = tools
            try:
                resp = self._create_with_retry(kwargs, attempts=2)
            except Exception as e:  # noqa: BLE001 - provider errors are opaque
                last = e
                continue
            if resp.choices and resp.choices[0].message.content:
                return resp
            if resp.choices and getattr(resp.choices[0].message, "tool_calls", None):
                return resp
            last = LLMError("LLM returned empty response")
        raise last or LLMError("LLM returned empty response")

    def chat_json(self, system: str, user: str, attempts: int = 4,
                  deadline: Optional[float] = None, **kw: Any) -> Any:
        """Chat and parse JSON, robust to flaky providers.

        Reasoning endpoints intermittently return an empty body (and no error)
        for hard generation prompts: the model spends its whole hidden
        reasoning budget and stops with no content. These failures come in
        time-clustered waves, so retry both across models (handled by chat())
        AND across time (backoff here), alternating json_mode and nudging
        temperature, before giving up. `deadline` (monotonic seconds) bounds
        the total wall time across all retries.
        """
        last: Exception | None = None
        for i in range(max(1, attempts)):
            if deadline is not None and _now() > deadline:
                break
            json_mode = (i % 2 == 0)
            # Vary temperature a little on later rounds to escape a bad window.
            round_kw = dict(kw)
            if i > 0 and "temperature" not in round_kw:
                round_kw["temperature"] = 0.8 if i % 2 else 0.4
            try:
                raw = self.chat(system, user, json_mode=json_mode,
                                deadline=deadline, **round_kw)
                return _extract_json(raw)
            except Exception as e:  # noqa: BLE001
                # Includes provider 5xx (Cloudflare 502/503/524), empty bodies
                # (LLMError) and malformed JSON. These come in time-clustered
                # waves that hit ALL fallback models at once, so retry across
                # TIME here as well as across models inside chat().
                last = e
                if deadline is not None and _now() > deadline:
                    break
                if i < attempts - 1:
                    time.sleep(CONFIG.llm_retry_backoff * (i + 1))
                continue
        if isinstance(last, LLMError):
            raise last
        raise LLMError(f"chat_json failed after {attempts} attempts: {last}")


def _balanced_slices(text: str) -> list[str]:
    """Return top-level balanced {...} and [...] substrings, in order.

    Scans the text tracking string state so braces inside JSON strings do not
    confuse the depth counter. This recovers JSON embedded in prose far more
    reliably than a naive find/rfind.
    """
    out: list[str] = []
    for opener, closer in (("{", "}"), ("[", "]")):
        depth = 0
        start = -1
        in_str = False
        esc = False
        for i, ch in enumerate(text):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                if depth == 0:
                    start = i
                depth += 1
            elif ch == closer and depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    out.append(text[start : i + 1])
    return out


def _extract_json(raw: str) -> Any:
    """Parse JSON, tolerating the many forms a chat model may emit.

    Handles: code fences anywhere, leading/trailing prose (including reasoning
    text before the JSON), unescaped newlines/control chars in strings, and
    stray string concatenation (``"a" + "b"``). Tries progressively more
    aggressive repairs and returns the first that parses.
    """
    candidates: list[str] = []
    text = raw.strip()

    # fenced blocks (anywhere)
    for m in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.DOTALL):
        candidates.append(m.group(1).strip())
    # the whole text, and every balanced brace/bracket group within it
    candidates.append(text)
    candidates.extend(_balanced_slices(text))

    seen: set[str] = set()
    for cand in candidates:
        if not cand or cand in seen:
            continue
        seen.add(cand)
        # repair Python-style string concatenation: "abc" + "def" -> "abcdef"
        joined = re.sub(r'"\s*\+\s*"', "", cand)
        for candidate in (cand, joined):
            for kwargs in ({"strict": False}, {}):
                try:
                    return json.loads(candidate, **kwargs)
                except (json.JSONDecodeError, ValueError):
                    continue

    raise LLMError(
        f"Could not parse JSON from LLM response (len={len(raw)}):\n{raw[:800]}")


def get_llm() -> LLM:
    return LLM()
