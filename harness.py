"""Load a fuzz harness together with the local sources it depends on.

A harness rarely defines its own input format inline: it calls helper functions
declared in a local header (`#include "fuzz.h"`) whose implementation lives in
a sibling .c file. Given only the harness, an LLM cannot deduce the byte format
of those helpers — so we follow the harness's own quoted #includes, resolve
them relative to the harness directory, and pull in the matching implementation
files (the common `name.h` <-> `name.c` convention).

This is pure C-preprocessor semantics — no project-specific knowledge — so the
agent stays target-agnostic: it simply reads whatever the harness includes.

Bounded by a total byte budget so prompts stay a reasonable size.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# #include "local/thing.h"  (quoted include = local file)
_QUOTED_INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.MULTILINE)

_EXTS = (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hxx", ".inc", ".def")


def _read(path: Path, limit: int) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:limit]
    except OSError:
        return ""


def _resolve(include_name: str, base_dir: Path, roots: list[Path]) -> Path | None:
    """Resolve one quoted include against the harness dir and extra roots."""
    for root in [base_dir] + roots:
        cand = (root / include_name).resolve()
        if cand.is_file():
            return cand
    return None


def _sibling_impl(header: Path) -> Path | None:
    """For a resolved header, find the sibling .c that implements it."""
    stem = header.with_suffix("")
    for ext in (".c", ".cc", ".cpp", ".cxx"):
        cand = Path(str(stem) + ext)
        if cand.is_file():
            return cand
    return None


def load_harness_context(harness_file: str, extra_roots: list[str] | None = None,
                         total_budget: int = 20000, per_file: int = 12000,
                         max_files: int = 8) -> str:
    """Return the harness source plus the local sources it #includes.

    Follows quoted includes transitively (bounded), and for each header also
    pulls in a sibling implementation file when one exists. Each file is
    prefixed with a marker so the LLM can tell them apart. Returns "" when no
    harness file is configured or readable.

    The budget is split EVENLY across the discovered files (rather than letting
    a large harness file consume it all), so the include files that define the
    actual input framing are always represented — even when the top-level
    harness is bigger than the whole budget.
    """
    if not harness_file:
        return ""
    root = Path(harness_file)
    if not root.is_file():
        return ""

    roots = [Path(r) for r in (extra_roots or []) if r and Path(r).is_dir()]

    # Pass 1: discover the transitive closure of local includes (paths only).
    order: list[Path] = []
    seen: set[Path] = set()
    queue: list[Path] = [root]
    while queue and len(order) < max_files:
        path = queue.pop(0)
        try:
            path = path.resolve()
        except OSError:
            continue
        if path in seen:
            continue
        seen.add(path)
        order.append(path)
        # follow this file's own quoted includes (read a small head to find them)
        head = _read(path, per_file)
        for inc in _QUOTED_INCLUDE.findall(head):
            resolved = _resolve(inc, path.parent, roots)
            if resolved is None or resolved in seen:
                continue
            queue.append(resolved)
            # headers usually declare; the sibling .c defines the format/logic
            if resolved.suffix in (".h", ".hpp", ".hxx"):
                impl = _sibling_impl(resolved)
                if impl is not None and impl not in seen:
                    queue.append(impl)

    if not order:
        return ""

    # Pass 2: split the budget, giving the INCLUDED files priority over the
    # entry harness. The entry file usually just orchestrates; the actual input
    # format is defined by the helper functions it #includes, so those are the
    # most valuable bytes to show, and rich prompts make some providers return
    # empty — so keep the whole thing small. Include files come FIRST; the
    # entry harness gets a capped slice at the end.
    out: list[str] = []
    remaining = total_budget
    include_files = [p for p in order[1:]]

    # include files first, equally (their total is ~3/4 of the budget)
    include_budget = remaining if not include_files else max(
        512, int(remaining * 0.75))
    for i, path in enumerate(include_files):
        if include_budget <= 0:
            break
        files_left = len(include_files) - i
        share = max(256, include_budget // files_left)
        text = _read(path, min(per_file, share))
        if not text:
            continue
        chunk = f"\n// ===== {path} =====\n{text}"
        if len(chunk) > include_budget:
            chunk = chunk[:include_budget]
        out.append(chunk)
        include_budget -= len(chunk)
        remaining -= len(chunk)

    # entry harness last, capped
    if remaining > 0:
        entry_cap = max(600, total_budget - len("".join(out)))
        text = _read(order[0], min(per_file, entry_cap, remaining))
        if text:
            chunk = f"\n// ===== {order[0]} (entry harness) =====\n{text}"
            if len(chunk) > remaining:
                chunk = chunk[:remaining]
            out.append(chunk)
            remaining -= len(chunk)

    return "".join(out)
