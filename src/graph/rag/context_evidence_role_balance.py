"""Analyze role balance in RAG context evidence."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import string, value

_ROLES = {"primary", "secondary", "background", "counterexample", "methodology"}


def analyze_context_evidence_role_balance(items: Iterable[Any]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    total = 0
    missing = 0
    for item in items or []:
        total += 1
        role = _role(item)
        if role == "unknown":
            missing += 1
        counts[role] += 1

    role_counts = {role: counts[role] for role in sorted(set(counts) | _ROLES | {"unknown"}) if counts[role]}
    dominant = None
    if counts:
        dominant = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
    flags = []
    if counts["primary"] == 0 and total:
        flags.append("absent_primary_evidence")
    if total and counts["background"] / total >= 0.6:
        flags.append("background_heavy_context")
    return {"total_items": total, "role_counts": role_counts, "dominant_role": dominant, "missing_role_count": missing, "imbalance_flags": flags}


def _role(item: Any) -> str:
    for key in ("role", "evidence_role"):
        text = string(value(item, key))
        if text:
            normalized = text.strip().casefold().replace("-", "_")
            return normalized if normalized in _ROLES else "unknown"
    return "unknown"
