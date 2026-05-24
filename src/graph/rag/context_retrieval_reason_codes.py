"""Assign compact reason codes to retrieved RAG context records."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import any_present, content_text, number, result_date, result_id, tokens, value

_LABELS = {
    "SCORE_HIGH": "High retrieval score",
    "QUERY_TERM": "Matches query keywords",
    "RECENT": "Recent or dated context",
    "AUTHORITY": "Authority metadata present",
    "CITATION": "Citation or URL present",
    "ENTITY": "Overlaps query entities",
}


def assign_context_retrieval_reason_codes(query: str, context_records: Iterable[Any], *, score_threshold: float = 0.75) -> dict[str, Any]:
    """Return reason code arrays and labels for each retrieved context record."""
    query_terms = tokens(query, min_length=4)
    query_entities = {match.group(0).casefold() for match in re.finditer(r"\b[A-Z][A-Za-z0-9&.-]*\b", str(query or ""))}
    rows = []
    for index, record in enumerate(context_records):
        codes = []
        if (number(value(record, "score")) or 0.0) >= score_threshold:
            codes.append("SCORE_HIGH")
        record_terms = tokens(content_text(record), min_length=4)
        if query_terms & record_terms:
            codes.append("QUERY_TERM")
        if result_date(record) is not None:
            codes.append("RECENT")
        if any_present(record, ("authority", "source_rank", "publisher", "author")):
            codes.append("AUTHORITY")
        if any_present(record, ("citation", "citations", "url", "source_url", "doi")):
            codes.append("CITATION")
        if query_entities and query_entities & record_terms:
            codes.append("ENTITY")
        rows.append({"result_id": result_id(record, index), "reason_codes": codes, "labels": [_LABELS[code] for code in codes]})
    return {"records": rows, "label_map": _LABELS}
