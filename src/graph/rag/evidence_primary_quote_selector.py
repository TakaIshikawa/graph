"""Select primary short quotes from retrieved evidence."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import any_present, content_text, result_id, tokens

_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?")


def select_primary_evidence_quotes(
    query: str,
    evidence: Iterable[Any],
    *,
    limit: int = 3,
    max_quote_length: int = 180,
) -> dict[str, Any]:
    """Return strongest evidence quotes ranked by overlap, metadata, and stability."""
    query_terms = tokens(query)
    candidates: list[dict[str, Any]] = []
    for index, item in enumerate(evidence):
        rid = result_id(item, index)
        citation_available = any_present(item, ("citation", "url", "source_url", "id"))
        for quote_index, quote in enumerate(_quotes(content_text(item), max_quote_length)):
            overlap = len(tokens(quote) & query_terms)
            score = overlap * 2 + (1 if citation_available else 0)
            reasons = []
            if overlap:
                reasons.append("query term overlap")
            if citation_available:
                reasons.append("citation available")
            candidates.append(
                {
                    "result_id": rid,
                    "quote": quote,
                    "score": score,
                    "reasons": reasons,
                    "_order": (index, quote_index),
                }
            )

    selected = sorted(candidates, key=lambda row: (-row["score"], row["_order"]))[: max(0, limit)]
    for row in selected:
        row.pop("_order", None)
    return {"selected_quotes": selected}


def _quotes(text: str, max_length: int) -> list[str]:
    quotes: list[str] = []
    for match in _SENTENCE_RE.finditer(text):
        quote = " ".join(match.group(0).split())
        if not quote:
            continue
        if len(quote) > max_length:
            quote = quote[: max(0, max_length - 3)].rstrip() + "..."
        quotes.append(quote)
    return quotes
