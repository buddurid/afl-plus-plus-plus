"""Structured / harness-aware input generation.

Many fuzzing harnesses consume a STRUCTURED input: a header of control fields
(mode selectors, flags, control values, chunk sizes) followed by the payload.
Coverage of whole subsystems is often gated by those header fields, yet
byte-level mutation almost never sets them in a coherent way while keeping the
payload valid.

This module lets the agent exploit that structure deliberately:

  control_sweep(seeds, header_bytes, option_values, ...)

takes existing seeds, keeps their payload, and rewrites a header control field
with each requested value. With `option_values` derived from the source (e.g.
a control value) this reaches features that edge-guided mutation
misses — the deterministic, directed counterpart to the LLM's hypotheses.

Target-agnostic: the header size, field offset/width and candidate values all
come from config (see FUZZ_HARNESS_* in agent/.env).
"""

from __future__ import annotations

from dataclasses import dataclass

from .coverage import CoverageSnapshot, Seed


@dataclass
class Candidate:
    data: bytes
    reasoning: str


def _apply_control(seed_data: bytes, header_bytes: int, offset: int,
                   width: int, value: int) -> bytes | None:
    """Return a copy of seed_data with `value` written at header offset."""
    if len(seed_data) < header_bytes or header_bytes <= 0:
        return None
    if offset < 0 or offset + width > header_bytes:
        return None
    b = bytearray(seed_data)
    for i in range(width):
        b[offset + i] = (value >> (8 * i)) & 0xFF
    return bytes(b)


def control_sweep(seeds: list[Seed], header_bytes: int,
                  option_values: list[int], offset: int = 0, width: int = 4,
                  max_per_seed: int = 12) -> list[Candidate]:
    """Rewrite the harness header's control field on each seed.

    For every (seed, option_value) pair, produce a variant that keeps the seed
    body but sets the control field. Deduplicated by full content.
    """
    out: list[Candidate] = []
    seen: set[bytes] = set()
    for s in seeds:
        produced = 0
        for val in option_values:
            if produced >= max_per_seed:
                break
            data = _apply_control(s.data, header_bytes, offset, width, val)
            if data is None or data in seen:
                continue
            seen.add(data)
            out.append(Candidate(
                data=data,
                reasoning=(f"control sweep: seed {s.name} with header field "
                           f"[{offset}:{offset+width}]={val} (0x{val:x})"),
            ))
            produced += 1
    return out


def header_echo_candidates(snapshot: CoverageSnapshot, header_bytes: int,
                           limit: int = 6) -> str:
    """Describe the current header values of the top seeds (for the LLM).

    Helps the model see which control values the corpus already uses, so it can
    propose ones that enable untouched features. The field layout comes from
    config (FUZZ_HARNESS_HEADER_FIELDS); nothing is hardcoded to a target.
    """
    from .config import CONFIG

    fields_spec = CONFIG.harness_header_fields or CONFIG.harness_control_field or "0:4"
    layout: list[tuple[int, int]] = []
    for part in fields_spec.replace(",", " ").split():
        try:
            off_s, width_s = part.split(":", 1)
            layout.append((int(off_s), int(width_s)))
        except ValueError:
            continue
    if not layout:
        layout = [(0, 4)]

    lines = []
    for s in snapshot.top_seeds(limit):
        if len(s.data) < header_bytes or header_bytes <= 0:
            continue
        fields = []
        for off, width in layout:
            if off + width <= header_bytes:
                val = int.from_bytes(s.data[off:off + width], "little")
                fields.append(f"[{off}:{off+width}]={val} (0x{val:x})")
        lines.append(f"- {s.name}: " + ", ".join(fields))
    return "\n".join(lines) if lines else "(no structured header configured)"
