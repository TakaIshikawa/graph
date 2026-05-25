"""Detect negative, null, failed, inconclusive, and regression evidence."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, result_id

_RESULT_CONTEXT_RE = re.compile(r"\b(?:result|finding|study|test|trial|experiment|analysis|evaluation|outcome|metric)\b", re.I)
_SPECS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("negative", re.compile(r"\b(?:negative\s+(?:result|finding|effect)|harmful|worse\s+outcome|decrease(?:d)?)\b", re.I)),
    ("null", re.compile(r"\b(?:null\s+(?:result|finding)|no\s+(?:effect|impact|difference|change)|not\s+statistically\s+significant)\b", re.I)),
    ("failed", re.compile(r"\b(?:failed|failure|did\s+not\s+meet|missed\s+the\s+target)\b", re.I)),
    ("inconclusive", re.compile(r"\b(?:inconclusive|unclear|mixed\s+results|insufficient\s+evidence)\b", re.I)),
    ("regression", re.compile(r"\b(?:regression|regressed|worsened|backslide|declined)\b", re.I)),
)


def detect_evidence_negative_results(evidence: Iterable[Any]) -> dict[str, Any]:
    """Return item-level negative-result matches and aggregate counts."""
    items: list[dict[str, Any]] = []
    counts = {name: 0 for name, _ in _SPECS}
    for index, item in enumerate(evidence or []):
        text = content_text(item)
        item_matches = []
        has_context = bool(_RESULT_CONTEXT_RE.search(text))
        for category, pattern in _SPECS:
            for match in pattern.finditer(text):
                if category in {"failed", "regression"} or has_context:
                    item_matches.append({"category": category, "cue": match.group(0).strip(), "span": [match.start(), match.end()]})
        if item_matches:
            for category in sorted({row["category"] for row in item_matches}):
                counts[category] += 1
            items.append({"result_id": result_id(item, index), "matches": item_matches})
    return {"category_counts": counts, "items": items}
