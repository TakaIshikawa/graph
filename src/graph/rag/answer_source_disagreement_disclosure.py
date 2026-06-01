"""Audit whether answers disclose disagreement present in evidence."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.rag._record_text import record_id, text_blob

_EVIDENCE_CUES = ("conflicting", "conflict", "disagreement", "disputed", "contradicts", "mixed", "sources differ")
_ANSWER_RE = re.compile(r"\b(conflicting|mixed|disagreement|sources differ|differ(?:ing)? sources)\b", re.I)


def audit_answer_source_disagreement_disclosure(
    answer: str, evidence: Iterable[Any] | None = None, sample_limit: int = 5
) -> dict[str, Any]:
    limit = max(0, sample_limit)
    cue_counts: Counter[str] = Counter()
    samples: list[dict[str, str]] = []
    for index, result in enumerate(evidence or []):
        text = text_blob(result)
        for cue in _EVIDENCE_CUES:
            if re.search(rf"\b{re.escape(cue)}\b", text, re.I):
                cue_counts[cue] += 1
                if len(samples) < limit:
                    samples.append({"result_id": record_id(result, index), "cue": cue})
    samples.sort(key=lambda row: (row["result_id"], row["cue"]))
    has_evidence_disagreement = bool(cue_counts)
    answer_discloses_disagreement = bool(_ANSWER_RE.search(str(answer or "")))
    return {
        "has_evidence_disagreement": has_evidence_disagreement,
        "answer_discloses_disagreement": answer_discloses_disagreement,
        "missing_disclosure": has_evidence_disagreement and not answer_discloses_disagreement,
        "cue_counts": dict(sorted(cue_counts.items())),
        "samples": samples[:limit],
    }
