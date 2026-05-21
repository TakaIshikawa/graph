"""Analyze quoted material density in RAG context."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import rounded_ratio, string

_QUOTE_RE = re.compile(r'"[^"]+"|\'[^\']+\'|“[^”]+”|‘[^’]+’')


def analyze_context_quote_density(context: Any) -> dict[str, Any]:
    """Estimate quote density for a context string or iterable of chunks."""
    chunks = _chunks(context)
    rows = [_metrics(index, chunk) for index, chunk in enumerate(chunks)]
    total_length = sum(row["text_length"] for row in rows)
    quoted_length = sum(row["quoted_length"] for row in rows)
    ratio = rounded_ratio(quoted_length, total_length)
    warnings = []
    if total_length == 0:
        warnings.append("empty_context")
    elif ratio < 0.02:
        warnings.append("under_quoted_context")
    elif ratio > 0.6:
        warnings.append("over_quoted_context")
    return {
        "chunk_count": len(rows),
        "text_length": total_length,
        "quoted_length": quoted_length,
        "quote_density": ratio,
        "chunks": rows,
        "warnings": warnings,
    }


def _chunks(context: Any) -> list[str]:
    if isinstance(context, str) or context is None:
        return [string(context) or ""]
    if isinstance(context, Iterable):
        return [string(chunk) or "" for chunk in context]
    return [string(context) or ""]


def _metrics(index: int, text: str) -> dict[str, Any]:
    quoted_spans = _QUOTE_RE.findall(text)
    blockquote_count = sum(1 for line in text.splitlines() if line.lstrip().startswith(">"))
    quoted_length = sum(len(span) for span in quoted_spans) + sum(len(line.lstrip()[1:].strip()) for line in text.splitlines() if line.lstrip().startswith(">"))
    return {
        "chunk_index": index,
        "text_length": len(text),
        "quoted_span_count": len(quoted_spans),
        "blockquote_count": blockquote_count,
        "quoted_length": quoted_length,
        "quote_density": rounded_ratio(quoted_length, len(text)),
    }
