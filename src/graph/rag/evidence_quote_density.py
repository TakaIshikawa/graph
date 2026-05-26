"""Quoted-text density scoring for evidence snippets."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

_QUOTE_RE = re.compile(r'"([^"]+)"|“([^”]+)”|‘([^’]+)’')


def score_evidence_quote_density(evidence: Iterable[Any]) -> dict[str, Any]:
    texts = [_text(item) for item in evidence]
    total_chars = sum(len(text) for text in texts)
    quoted_spans = [match.group(1) or match.group(2) or match.group(3) or "" for text in texts for match in _QUOTE_RE.finditer(text)]
    quoted_chars = sum(len(span) for span in quoted_spans)
    density = quoted_chars / total_chars if total_chars else 0.0
    return {
        "total_chars": total_chars,
        "quoted_chars": quoted_chars,
        "quote_density": round(density, 4),
        "quote_count": len(quoted_spans),
        "density_bucket": _bucket(density),
    }


def _text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, Mapping):
        for key in ("snippet", "text", "content", "quote"):
            value = item.get(key)
            if value not in (None, ""):
                return str(value)
    for key in ("snippet", "text", "content", "quote"):
        value = getattr(item, key, None)
        if value not in (None, ""):
            return str(value)
    return ""


def _bucket(density: float) -> str:
    if density == 0:
        return "none"
    if density < 0.25:
        return "low"
    if density < 0.5:
        return "medium"
    return "high"
