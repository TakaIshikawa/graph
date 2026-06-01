"""Audit temporal scope alignment between evidence and answers."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._record_text import record_id, text_blob

_TEMPORAL = re.compile(r"\b(?:19|20)\d{2}\b|\b\d{4}-\d{2}-\d{2}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b", re.I)
_CAVEAT = re.compile(r"\b(?:as of|currently|latest|at the time|evidence may be stale|freshness)\b", re.I)


def audit_answer_temporal_scope_alignment(answer: str, evidence: Iterable[Any] | None = None, sample_limit: int = 5) -> dict[str, Any]:
    samples = []
    evidence_count = 0
    for index, item in enumerate(evidence or ()):
        for match in _TEMPORAL.finditer(text_blob(item)):
            evidence_count += 1
            if len(samples) < sample_limit:
                samples.append({"result_id": record_id(item, index), "matched_temporal_marker": match.group(0)})
    answer_markers = _TEMPORAL.findall(str(answer or ""))
    has_caveat = bool(_CAVEAT.search(str(answer or "")))
    return {
        "evidence_temporal_marker_count": evidence_count,
        "answer_temporal_marker_count": len(answer_markers),
        "answer_has_freshness_caveat": has_caveat,
        "missing_temporal_scope": evidence_count > 0 and not answer_markers and not has_caveat,
        "samples": samples,
    }
