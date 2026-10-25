"""Validator Agent: replay candidates, keep only coverage-adding inputs."""

from __future__ import annotations

import os
from dataclasses import dataclass

from .config import CONFIG
from .coverage import _run_showmap


@dataclass
class Verdict:
    path: str
    accepted: bool
    new_edges: int
    total_edges: int
    reasoning: str = ""


def validate_candidates(candidates, known_edges: set[int], staging_dir: str | None = None):
    """Write candidates to staging, run each through showmap, accept inputs
    that contribute at least one edge not in `known_edges`.

    Returns (accepted: list[(path, edges)], verdicts: list[Verdict],
             updated_known_edges: set[int]).
    """
    staging = staging_dir or CONFIG.staging_dir
    os.makedirs(staging, exist_ok=True)

    accepted = []
    verdicts = []
    known = set(known_edges)
    budget = CONFIG.validate_budget

    for i, cand in enumerate(candidates[:budget]):
        path = os.path.join(staging, f"cand_{i:05d}.bin")
        with open(path, "wb") as fh:
            fh.write(cand.data)

        edges, bad = _run_showmap(path)
        new = edges - known

        if bad and not edges:
            verdict = Verdict(path, False, 0, len(edges), "target failed/hung with no coverage")
        elif new:
            verdict = Verdict(path, True, len(new), len(edges), f"adds {len(new)} new edge(s)")
            known |= new
            accepted.append((path, edges))
        else:
            verdict = Verdict(path, False, 0, len(edges), "no new coverage")

        verdicts.append(verdict)
        # Clean rejected candidates from staging to avoid clutter
        if not verdict.accepted:
            try:
                os.remove(path)
            except OSError:
                pass

    return accepted, verdicts, known
