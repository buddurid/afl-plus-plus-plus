"""Orchestrator Agent: the main cycle.

    snapshot -> plateau? -> mode -> mapper -> generator -> validator -> report

Run once:            python -m agent.orchestrator --once
Run every N seconds: python -m agent.orchestrator --interval 3600
"""

from __future__ import annotations

import argparse
import json
import shutil
import time

from .config import CONFIG
from .coverage import analyze
from .generator import generate_for_surface, generate_neighbors
from .llm import get_llm
from .mapper import Hypothesis, map_coverage, propose_surfaces
from .memory import Memory
from .validator import validate_candidates


def run_cycle(llm, mem: Memory) -> dict:
    CONFIG.ensure_dirs()
    t0 = time.time()

    # 1. Snapshot
    print("[1/5] Analyzing corpus coverage...")
    snap = analyze()
    print(f"      {len(snap.seeds)} seeds, {snap.num_edges} edges covered")
    known = set(snap.all_edges)

    # 2. Mode selection
    plateau = mem.is_plateaued()
    mode = "surface" if plateau else "neighbor"
    print(f"[2/5] Mode: {mode} (plateau={plateau})")

    accepted_paths = []
    n_candidates = 0
    verdicts = []

    if not snap.seeds:
        print("      Corpus is empty — nothing to analyze. Add seeds and retry.")
        report = {"mode": mode, "skipped": "empty corpus"}
        _write_report(report)
        return report

    try:
        if mode == "neighbor":
            # 3. Map coverage to functionality -> hypotheses
            print("[3/5] Mapping coverage to functionality (LLM)...")
            mapping = map_coverage(llm, snap)
            hypotheses = [
                Hypothesis(
                    functions=h.get("functions", []),
                    reasoning=h.get("reasoning", ""),
                    hint=h.get("hint", ""),
                )
                for h in mapping.get("near_miss", [])
            ]
            summary_text = mapping.get("summary", "")

            # 4. Generate
            print("[4/5] Generating neighbor candidates (LLM)...")
            candidates = generate_neighbors(llm, snap, hypotheses)
        else:
            # Plateau: hunt new attack surfaces
            print("[3/5] Proposing new attack surfaces (LLM)...")
            surfaces = propose_surfaces(llm, snap, mem.summary())
            summary_text = "; ".join(s.name for s in surfaces)

            print("[4/5] Generating surface candidates (LLM)...")
            candidates = generate_for_surface(llm, snap, surfaces)

        n_candidates = len(candidates)
        print(f"      {n_candidates} candidates generated")

        # 5. Validate
        print("[5/5] Validating candidates...")
        accepted, verdicts, known = validate_candidates(candidates, known)
        new_edges = known - snap.all_edges

        # Move accepted inputs to accepted_dir (injection point TBD)
        dest_dir = CONFIG.accepted_dir
        for path, _edges in accepted:
            dest = dest_path(dest_dir, path)
            shutil.move(path, dest)
            accepted_paths.append(dest)

        delta = len(known) - snap.num_edges
        print(
            f"      accepted {len(accepted)}/{n_candidates}, "
            f"+{delta} new edges -> {dest_dir}"
        )

        mem.record_cycle(
            mode=mode,
            num_candidates=n_candidates,
            num_accepted=len(accepted),
            new_edges=delta,
            total_edges=len(known),
            summary=summary_text,
        )

        # Failed hypotheses that produced nothing → remember to avoid repeats
        if mode == "neighbor" and not accepted:
            mem.record_hypotheses([h.hint for h in hypotheses])

        report = {
            "mode": mode,
            "plateau": plateau,
            "seeds": len(snap.seeds),
            "edges_before": snap.num_edges,
            "edges_after": len(known),
            "new_edges": delta,
            "candidates": n_candidates,
            "accepted": len(accepted),
            "accepted_paths": accepted_paths,
            "verdicts": [vars(v) for v in verdicts],
            "duration_sec": round(time.time() - t0, 1),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        report = {"mode": mode, "error": str(e), "duration_sec": round(time.time() - t0, 1)}

    _write_report(report)
    return report


def dest_path(dest_dir: str, src: str) -> str:
    import os
    name = os.path.basename(src)
    return os.path.join(dest_dir, f"gen_{int(time.time())}_{name}")


def _write_report(report: dict) -> None:
    with open(CONFIG.report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"Report written to {CONFIG.report_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM-guided fuzzing orchestrator")
    ap.add_argument("--once", action="store_true", help="run a single cycle and exit")
    ap.add_argument("--interval", type=int, default=3600, help="seconds between cycles")
    ap.add_argument(
        "--agentic", action="store_true",
        help="LLM decides its own workflow via tool calls (ReAct loop) "
             "instead of the fixed pipeline",
    )
    ap.add_argument(
        "--langgraph", action="store_true",
        help="run the workflow as a LangGraph state machine (agent<->tools "
             "graph with checkpointing)",
    )
    args = ap.parse_args()

    llm = get_llm()
    mem = Memory()

    def _cycle() -> dict:
        if args.langgraph:
            from .graph import run_langgraph_cycle
            return run_langgraph_cycle(mem)
        if args.agentic:
            from .agent_loop import run_agentic_cycle
            return run_agentic_cycle(llm, mem)
        return run_cycle(llm, mem)

    if args.once:
        _cycle()
        return

    while True:
        try:
            _cycle()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"cycle failed: {e}")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
