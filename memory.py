"""Persistent memory: hypothesis history across orchestrator cycles."""

from __future__ import annotations

import json
import os
import time
from typing import Any

from .config import CONFIG


class Memory:
    """JSON-backed log of cycles: mode, hypotheses, results. Enables the
    orchestrator to avoid repeating failed hypotheses and to compute
    plateau statistics."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path or CONFIG.memory_path
        self._data: dict[str, Any] = {"cycles": []}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as fh:
                    self._data = json.load(fh)
            except (OSError, json.JSONDecodeError):
                self._data = {"cycles": []}

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2)

    def record_cycle(
        self,
        mode: str,
        num_candidates: int,
        num_accepted: int,
        new_edges: int,
        total_edges: int,
        summary: str = "",
    ) -> None:
        self._data["cycles"].append(
            {
                "ts": time.time(),
                "mode": mode,
                "candidates": num_candidates,
                "accepted": num_accepted,
                "new_edges": new_edges,
                "total_edges": total_edges,
                "summary": summary,
            }
        )
        self.save()

    def record_hypotheses(self, hypotheses: list[str]) -> None:
        self._data.setdefault("failed_hypotheses", [])
        # dedup
        for h in hypotheses:
            if h not in self._data["failed_hypotheses"]:
                self._data["failed_hypotheses"].append(h)
        self._data["failed_hypotheses"] = self._data["failed_hypotheses"][-100:]
        self.save()

    def last_cycles(self, n: int) -> list[dict]:
        return self._data["cycles"][-n:]

    def new_edges_last_cycles(self, n: int) -> int:
        return sum(c.get("new_edges", 0) for c in self.last_cycles(n))

    def is_plateaued(self) -> bool:
        """Plateau if we have at least `plateau_cycles` recorded cycles and
        the recent ones produced fewer than `plateau_new_edges` new edges."""
        cycles = self.last_cycles(CONFIG.plateau_cycles)
        if len(cycles) < CONFIG.plateau_cycles:
            return False
        return self.new_edges_last_cycles(CONFIG.plateau_cycles) < CONFIG.plateau_new_edges

    def summary(self) -> str:
        cycles = self.last_cycles(5)
        if not cycles:
            return "(no previous cycles)"
        lines = [
            f"cycle {c['ts']:.0f}: mode={c['mode']} accepted={c['accepted']} "
            f"new_edges={c['new_edges']} total_edges={c['total_edges']}"
            for c in cycles
        ]
        return "\n".join(lines)
