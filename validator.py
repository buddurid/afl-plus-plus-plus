"""Validator Agent: replay candidates, keep only coverage-adding inputs."""

from __future__ import annotations

import os
from dataclasses import dataclass

from .config import CONFIG
from .coverage import _run_showmap


def _functions_for(input_path: str) -> list[str]:
    """Best-effort function attribution for an accepted input via the gcov
    build. Returns [] when gcov is not configured."""
    try:
        from .funcs_coverage import run_with_gcov
        from .coverage import TARGET_CMD
        import os

        if not CONFIG.gcov_build_dir:
            return []
        gcov_cmd = CONFIG.gcov_target_cmd or CONFIG.target_cmd
        base = gcov_cmd.split()
        if "@@" in base:
            i = base.index("@@")
            if i > 0 and base[i - 1] == "-f":
                argv = base[: i - 1] + ["-f", input_path] + base[i + 1 :]
            else:
                argv = base[:i] + [input_path] + base[i + 1 :]
        elif "@-" in base:
            i = base.index("@-")
            argv = base[:i] + [input_path] + base[i + 1 :]
        else:
            argv = base
        res = run_with_gcov(argv)
        return res.function_names if res else []
    except Exception:
        return []


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
            detail = f"adds {len(new)} new edge(s)"
            fc = _functions_for(path)
            if fc:
                detail += f", functions=[{', '.join(fc)}]"
            verdict = Verdict(path, True, len(new), len(edges), detail)
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
