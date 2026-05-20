"""Find likely counterexamples and exception passages in retrieved evidence."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, result_id

_CUES = (
    ("exception", re.compile(r"\b(?:except|exception|unless|other than|with the exception of)\b", re.I)),
    ("limitation", re.compile(r"\b(?:limited|limitation|caveat|constraint|not enough evidence|small sample)\b", re.I)),
    ("negative_finding", re.compile(r"\b(?:no evidence|did not|does not|failed to|not significant|without improvement)\b", re.I)),
    ("conflicting_condition", re.compile(r"\b(?:however|although|whereas|in contrast|conflicts? with|contrary to)\b", re.I)),
)
_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")


def find_evidence_counterexamples(results: Iterable[Any], *, max_passages_per_result: int = 3) -> dict[str, Any]:
    """Return bounded counterexample cue passages grouped by result."""
    if not isinstance(max_passages_per_result, int) or isinstance(max_passages_per_result, bool) or max_passages_per_result < 0:
        raise ValueError("max_passages_per_result must be a non-negative integer")

    items = list(results)
    rows = []
    for index, result in enumerate(items):
        matches = []
        seen = set()
        for sentence in _sentences(content_text(result)):
            for cue_type, pattern in _CUES:
                if len(matches) >= max_passages_per_result:
                    break
                if not pattern.search(sentence):
                    continue
                key = (cue_type, sentence.casefold())
                if key in seen:
                    continue
                seen.add(key)
                matches.append({"cue_type": cue_type, "passage": sentence[:220]})
            if len(matches) >= max_passages_per_result:
                break
        if matches:
            rows.append({"result_id": result_id(result, index), "matches": matches, "match_count": len(matches)})
    counts = Counter(match["cue_type"] for row in rows for match in row["matches"])
    return {
        "total_results": len(items),
        "result_count": len(rows),
        "counterexample_count": sum(row["match_count"] for row in rows),
        "cue_type_counts": {cue_type: counts.get(cue_type, 0) for cue_type, _ in _CUES},
        "results": rows,
        "warnings": ["counterexamples_found"] if rows else [],
    }


def _sentences(text: str) -> list[str]:
    return [match.group(0).strip() for match in _SENTENCE_RE.finditer(text) if match.group(0).strip()]
