"""Audit whether a RAG answer discloses assumptions."""

from __future__ import annotations

import re
from typing import Any

_EXPLICIT_SECTION_RE = re.compile(r"(?:^|\n)\s*(?:[-*]\s*)?(?:#+\s*)?(assumptions?|key assumptions?)\s*:?", re.I)
_EXPLICIT_PHRASES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("section", _EXPLICIT_SECTION_RE),
    ("assumption_phrase", re.compile(r"\b(?:assume|assumption|assumptions|if\s+we\s+assume|on\s+the\s+assumption\s+that|provided\s+that)\b", re.I)),
    ("held_constant", re.compile(r"\bholding\s+[^.?!;]{1,80}?\s+constant\b", re.I)),
)
_IMPLICIT_CUES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("assuming", re.compile(r"\bassuming\b", re.I)),
    ("if_we_treat", re.compile(r"\bif\s+we\s+treat\b", re.I)),
    ("this_implies", re.compile(r"\bthis\s+implies\b", re.I)),
    ("likely_means", re.compile(r"\blikely\s+means\b", re.I)),
)


def audit_answer_assumption_disclosure(answer: str, legacy_answer: str | None = None) -> dict[str, Any]:
    """Return explicit assumptions and implicit assumption cues.

    ``legacy_answer`` keeps older two-argument callers working; new callers pass
    only the answer text.
    """
    text = _inline_text(answer if legacy_answer is None else legacy_answer)
    disclosed = _disclosed_assumptions(text)
    implicit = _implicit_cues(text)
    return {
        "assumption_count": len(disclosed),
        "disclosed_assumptions": disclosed,
        "implicit_cues": implicit,
        "needs_disclosure": bool(implicit and not disclosed),
    }


def _disclosed_assumptions(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cue_type, pattern in _EXPLICIT_PHRASES:
        for match in pattern.finditer(text):
            rows.append({"type": cue_type, "cue": match.group(0).strip(" \t:"), "span": [match.start(), match.end()]})
    return _remove_overlaps(sorted(_dedupe(rows), key=lambda row: (row["span"][0], -(row["span"][1] - row["span"][0]), row["type"])))


def _implicit_cues(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cue_type, pattern in _IMPLICIT_CUES:
        for match in pattern.finditer(text):
            rows.append({"type": cue_type, "cue": match.group(0).strip(" \t:"), "span": [match.start(), match.end()]})
    return sorted(_dedupe(rows), key=lambda row: (row["span"][0], row["span"][1], row["type"]))


def _dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, int, int]] = set()
    unique: list[dict[str, Any]] = []
    for row in rows:
        key = (row["type"], row["cue"].casefold(), row["span"][0], row["span"][1])
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def _remove_overlaps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for row in rows:
        start, end = row["span"]
        if any(start < kept_row["span"][1] and end > kept_row["span"][0] for kept_row in kept):
            continue
        kept.append(row)
    return sorted(kept, key=lambda row: (row["span"][0], row["span"][1], row["type"]))


def _inline_text(value: object) -> str:
    return " ".join(("" if value is None else str(value)).split())
