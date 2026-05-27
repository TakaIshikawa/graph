"""Classify context source roles and report balance warnings."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from graph.rag._analysis_utils import content_text, result_id, value

_ROLES = ("primary", "secondary", "background", "opinion", "data", "unknown")
_HINTS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("data", re.compile(r"\b(?:dataset|data table|raw data|statistics|measurements)\b", re.I)),
    ("primary", re.compile(r"\b(?:primary source|official record|original report|transcript|filing)\b", re.I)),
    ("secondary", re.compile(r"\b(?:review|analysis|summary|synthesis|secondary source)\b", re.I)),
    ("opinion", re.compile(r"\b(?:opinion|editorial|commentary|blog post)\b", re.I)),
    ("background", re.compile(r"\b(?:background|overview|primer|explainer)\b", re.I)),
)


def analyze_context_source_role_balance(context_items: list[dict]) -> dict[str, Any]:
    rows = list(context_items or [])
    counts: Counter[str] = Counter({role: 0 for role in _ROLES})
    items = []
    for index, item in enumerate(rows):
        role = _role(item)
        counts[role] += 1
        items.append({"id": result_id(item, index), "role": role})
    warnings = []
    evidence_heavy = counts["secondary"] + counts["opinion"] + counts["background"] >= 2 or len(rows) >= 3
    if evidence_heavy and counts["primary"] == 0:
        warnings.append("missing_primary_sources")
    if evidence_heavy and counts["data"] == 0:
        warnings.append("missing_data_sources")
    return {"total_items": len(rows), "counts": dict(counts), "items": items, "warnings": warnings}


def _role(item: Any) -> str:
    explicit = value(item, "role")
    if isinstance(explicit, str) and explicit.casefold().replace("-", "_") in _ROLES:
        return explicit.casefold().replace("-", "_")
    source_type = value(item, "source_role")
    if isinstance(source_type, str) and source_type.casefold().replace("-", "_") in _ROLES:
        return source_type.casefold().replace("-", "_")
    text = content_text(item)
    for role, pattern in _HINTS:
        if pattern.search(text):
            return role
    return "unknown"
