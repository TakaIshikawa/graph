"""Audit whether an answer discloses evidence gaps found in records."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import result_id
from graph.rag._record_text import text_blob

_CUES = ("missing", "unavailable", "not found", "no evidence", "insufficient evidence", "unclear", "unknown")
_CUE_RE = re.compile(r"\b(?:missing|unavailable|not\s+found|no\s+evidence|insufficient\s+evidence|unclear|unknown)\b", re.I)


def audit_answer_evidence_gap_disclosure(
    answer: str,
    evidence: Iterable[Any] | None = None,
    sample_limit: int = 5,
) -> dict[str, Any]:
    """Return evidence-gap cue counts and whether the answer discloses them."""
    limit = max(0, int(sample_limit))
    cue_counts: Counter[str] = Counter({cue: 0 for cue in _CUES})
    samples: list[dict[str, str]] = []
    gap_count = 0

    for index, record in enumerate(evidence or []):
        text = text_blob(record)
        cues = _matched_cues(text)
        if not cues:
            continue
        gap_count += 1
        for cue in cues:
            cue_counts[cue] += 1
        if len(samples) < limit:
            samples.append({"result_id": result_id(record, index), "cue": cues[0]})

    discloses = bool(_CUE_RE.search("" if answer is None else str(answer)))
    return {
        "evidence_gap_count": gap_count,
        "answer_discloses_gap": discloses,
        "missing_gap_disclosure": gap_count > 0 and not discloses,
        "cue_counts": dict(cue_counts),
        "samples": samples,
    }


def _matched_cues(text: str) -> list[str]:
    found: list[str] = []
    lowered = text.casefold()
    for cue in _CUES:
        pattern = r"\b" + re.escape(cue).replace(r"\ ", r"\s+") + r"\b"
        if re.search(pattern, lowered):
            found.append(cue)
    return found
