"""Extract quote-worthy evidence spans from RAG results."""

from __future__ import annotations

import re
from collections.abc import Sequence, Mapping
from typing import Any

from graph.rag._analysis_utils import content_text, result_id

_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]|$)")
_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*(?:%|ms|s|kg|km|m|x)?\b")
_DATE_RE = re.compile(r"(?i)\b(?:\d{4}-\d{2}-\d{2}|Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b")
_ENTITY_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")
_QUOTE_RE = re.compile(r'"[^"]{6,}"|“[^”]{6,}”|\'[^\']{6,}\'')
_METHOD_RE = re.compile(r"(?i)\b(?:according to|measured|using|based on|sample|dataset|table|figure|survey)\b")

_CUES = {
    "number": _NUMBER_RE,
    "date": _DATE_RE,
    "named_reference": _ENTITY_RE,
    "quoted_phrase": _QUOTE_RE,
    "method_detail": _METHOD_RE,
}


def _validate_max(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("max_spans_per_result must be a non-negative integer")
    return value


def _sentence_candidates(text: str) -> list[tuple[int, int, str, list[str], int]]:
    rows = []
    for match in _SENTENCE_RE.finditer(text):
        span = match.group(0).strip()
        if not span:
            continue
        cues = [name for name, pattern in _CUES.items() if pattern.search(span)]
        score = len(cues)
        rows.append((match.start() + len(match.group(0)) - len(match.group(0).lstrip()), match.end(), span, cues, score))
    rows.sort(key=lambda item: (-item[4], item[0]))
    return rows


def extract_evidence_quote_spans(
    results: Sequence[Mapping[str, Any]],
    max_spans_per_result: int = 3,
) -> dict[str, Any]:
    """Return bounded quote-worthy spans with offsets into the original text."""
    limit = _validate_max(max_spans_per_result)
    rows: list[dict[str, Any]] = []
    sparse = 0
    for index, result in enumerate(results):
        text = content_text(result)
        spans = []
        for start, end, span_text, cues, _score in _sentence_candidates(text):
            if not cues:
                continue
            spans.append({"start": start, "end": end, "text": span_text, "cues": cues})
            if len(spans) >= limit:
                break
        if not spans:
            sparse += 1
        rows.append({"result_id": result_id(result, index), "spans": spans, "span_count": len(spans)})
    warnings = []
    if not rows:
        warnings.append("no_results")
    elif sparse == len(rows):
        warnings.append("sparse_evidence_spans")
    elif sparse:
        warnings.append("some_results_without_spans")
    return {"result_count": len(rows), "results": rows, "warnings": warnings}
