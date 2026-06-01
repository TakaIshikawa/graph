"""Analyze evidence records for access-date signals."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from graph.rag._record_text import first, record_id, text_blob, value

_ACCESS_KEYS = ("accessed", "access_date", "retrieved_at", "last_checked")
_ACCESS_RE = re.compile(r"\b(?:accessed|retrieved|last\s+checked)\s*(?:on|at|:)?\s*\d{4}-\d{2}-\d{2}\b", re.I)


def analyze_evidence_access_date_signals(evidence: Iterable[Any]) -> dict[str, Any]:
    total = 0
    with_access = 0
    missing = []
    for index, item in enumerate(evidence):
        total += 1
        metadata = value(item, "metadata")
        has_key = isinstance(metadata, Mapping) and any(metadata.get(key) for key in _ACCESS_KEYS)
        if has_key or first(item, _ACCESS_KEYS) or _ACCESS_RE.search(text_blob(item)):
            with_access += 1
        else:
            missing.append({"index": index, "source_id": record_id(item, index, "evidence"), "title": first(item, ("title",))})
    return {
        "total_evidence": total,
        "with_access_date": with_access,
        "missing_access_date": total - with_access,
        "samples": missing[:5],
    }
