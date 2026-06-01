"""Audit whether answers acknowledge counterevidence found in records."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import result_id
from graph.rag._record_text import text_blob

_CUES = ("however", "but", "contradicts", "contrary", "exception", "limitation", "caveat", "on the other hand")
_CUE_RE = re.compile(
    r"\b(?:however|but|contradicts|contrary|exception|limitation|caveat|on\s+the\s+other\s+hand)\b",
    re.I,
)


def audit_answer_counterevidence_handling(
    answer: str,
    evidence: Iterable[Any] | None = None,
    sample_limit: int = 5,
) -> dict[str, Any]:
    """Return counterevidence cue counts and answer acknowledgement status."""
    limit = max(0, int(sample_limit))
    cue_counts: Counter[str] = Counter({cue: 0 for cue in _CUES})
    samples: list[dict[str, str]] = []
    counterevidence_count = 0

    for index, record in enumerate(evidence or []):
        text = text_blob(record)
        cues = _matched_cues(text)
        if not cues:
            continue
        counterevidence_count += 1
        for cue in cues:
            cue_counts[cue] += 1
        if len(samples) < limit:
            samples.append({"result_id": result_id(record, index), "cue": cues[0]})

    acknowledges = bool(_CUE_RE.search("" if answer is None else str(answer)))
    return {
        "counterevidence_count": counterevidence_count,
        "answer_acknowledges_counterevidence": acknowledges,
        "missing_counterevidence_handling": counterevidence_count > 0 and not acknowledges,
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
