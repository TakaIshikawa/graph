"""Detect answer superlatives that lack ranking or comparative evidence."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, result_id

_SUPERLATIVE_RE = re.compile(r"\b(best|worst|most|least|largest|smallest|fastest|safest|highest|lowest|leading|top-ranked)\b", re.I)
_SUPPORT_RE = re.compile(r"\b(rank(?:ed|ing)?|top[- ]?\d+|#\d+|highest|lowest|largest|smallest|fastest|safest|best|worst|most|least|leading|compared|benchmark)\b", re.I)
_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")


def detect_unsupported_superlatives(answer: str, results: Iterable[Any]) -> dict[str, Any]:
    """Return superlative answer claims and evidence support status."""
    evidence = [{"result_id": result_id(result, index), "text": content_text(result)} for index, result in enumerate(results or [])]
    claims = []
    for index, sentence in enumerate(_sentences(str(answer or ""))):
        cues = sorted({match.group(1).casefold() for match in _SUPERLATIVE_RE.finditer(sentence)})
        if not cues:
            continue
        matched_ids = [row["result_id"] for row in evidence if _SUPPORT_RE.search(row["text"]) and any(cue in row["text"].casefold() for cue in cues)]
        reasons = [] if matched_ids else ["missing_comparative_or_ranking_evidence"]
        claims.append(
            {
                "sentence_index": index,
                "claim": sentence[:200],
                "superlatives": cues,
                "support_status": "supported" if matched_ids else "unsupported",
                "matched_result_ids": matched_ids,
                "reasons": reasons,
            }
        )

    reason_counts = Counter(reason for claim in claims for reason in claim["reasons"])
    warnings = []
    if not str(answer or "").strip():
        warnings.append("no_answer")
    if not evidence:
        warnings.append("no_results")
    if reason_counts:
        warnings.append("unsupported_superlatives")
    return {
        "total_superlatives": len(claims),
        "supported_superlatives": sum(1 for claim in claims if claim["support_status"] == "supported"),
        "unsupported_superlatives": sum(1 for claim in claims if claim["support_status"] == "unsupported"),
        "claims": claims,
        "reason_counts": dict(sorted(reason_counts.items())),
        "warnings": warnings,
    }


def _sentences(text: str) -> list[str]:
    return [match.group(0).strip() for match in _SENTENCE_RE.finditer(text) if match.group(0).strip()]
