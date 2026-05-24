"""Score whether evidence matches query scope dimensions."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, metadata, result_id

_DIMENSIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("entity", ("customer", "account", "product", "team", "company", "vendor")),
    ("time", ("today", "weekly", "monthly", "quarter", "year", "2024", "2025", "2026")),
    ("geography", ("us", "u.s.", "europe", "emea", "apac", "japan", "california", "global")),
    ("population", ("enterprise", "smb", "consumer", "patients", "employees", "users")),
    ("metric", ("revenue", "latency", "cost", "conversion", "retention", "accuracy", "margin")),
)


def score_evidence_scope_match(query: str, evidence: Iterable[Any]) -> dict[str, Any]:
    """Return per-evidence deterministic scope match rows."""
    qtext = " ".join(str(query or "").split())
    qscope = _scope_terms(qtext)
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(evidence or []):
        etext = " ".join([content_text(item), " ".join(str(v) for v in metadata(item).values())])
        escope = _scope_terms(etext)
        required = sorted(qscope)
        matched = sorted(dim for dim in required if qscope[dim] & escope.get(dim, set()))
        missing = sorted(dim for dim in required if dim not in matched)
        score = 1.0 if not required else round(len(matched) / len(required), 2)
        rows.append({"result_id": result_id(item, index), "scope_score": score, "matched_dimensions": matched, "missing_dimensions": missing})
    return {"query_scope": {dim: sorted(values) for dim, values in sorted(qscope.items())}, "evidence": rows}


def _scope_terms(text: str) -> dict[str, set[str]]:
    folded = text.casefold()
    scope: dict[str, set[str]] = {}
    for dimension, terms in _DIMENSIONS:
        for term in terms:
            if re.search(rf"\b{re.escape(term)}\b", folded):
                scope.setdefault(dimension, set()).add(term)
    return scope
