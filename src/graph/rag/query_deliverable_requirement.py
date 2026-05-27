"""Detect requested deliverable forms in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_DELIVERABLE_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("checklist", ("checklist", "check list")),
    ("table", ("table", "tabular")),
    ("timeline", ("timeline", "chronology")),
    ("summary", ("summary", "summarize", "summarise")),
    ("brief", ("brief", "one pager", "one-pager")),
    ("plan", ("plan", "roadmap")),
    ("comparison_matrix", ("comparison matrix", "matrix")),
    ("json", ("json",)),
    ("csv", ("csv", "comma separated", "comma-separated")),
    ("bullet_list", ("bullet list", "bullets", "bullet points")),
)


def detect_query_deliverable_requirement(query: str) -> dict[str, Any]:
    """Return deliverable forms requested by the query."""
    normalized_query = _normalize(query)
    matched_cues = _matched_cues(normalized_query)
    deliverables = list(dict.fromkeys(cue["deliverable"] for cue in matched_cues))
    return {
        "normalized_query": normalized_query,
        "deliverables": deliverables,
        "primary_deliverable": deliverables[0] if deliverables else None,
        "confidence": _confidence(deliverables, matched_cues),
        "matched_cues": matched_cues,
    }


def _matched_cues(normalized_query: str) -> list[dict[str, Any]]:
    rows = []
    seen: set[tuple[str, str]] = set()
    for deliverable, cues in _DELIVERABLE_SPECS:
        for cue in cues:
            for match in re.finditer(rf"\b{re.escape(cue)}\b", normalized_query):
                key = (deliverable, cue)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"deliverable": deliverable, "cue": cue, "span": [match.start(), match.end()]})
    rows.sort(key=lambda row: (row["span"][0], row["span"][1], row["deliverable"]))
    return rows


def _confidence(deliverables: list[str], matched_cues: list[dict[str, Any]]) -> float:
    if not deliverables:
        return 0.0
    if any(row["deliverable"] in {"json", "csv", "comparison_matrix"} for row in matched_cues):
        return 0.9
    if len(deliverables) > 1:
        return 0.85
    return 0.75


def _normalize(query: str) -> str:
    return " ".join(str(query or "").casefold().split())
