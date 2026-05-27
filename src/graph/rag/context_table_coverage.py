"""Audit table evidence coverage in RAG context items."""

from __future__ import annotations

import re
from typing import Any

from graph.rag._analysis_utils import MISSING, result_id, string, value

_TABLE_QUERY_RE = re.compile(
    r"\b(?:table|tabular|rows?|columns?|csv|spreadsheet|dataset|data|metric|metrics|number|numbers|numeric|"
    r"stats?|statistics|percent|percentage|rate|ratio|average|median|total|count|trend|compare|comparison|"
    r"versus|vs|benchmark|ranking|ranked|price|cost|revenue|growth|latency)\b",
    re.IGNORECASE,
)
_NUMERIC_RE = re.compile(r"(?:\d|[%$€£¥]|\b(?:p\d{2}|q[1-4])\b)", re.IGNORECASE)


def analyze_context_table_coverage(context_items: list[dict[str, Any]], query: str = "") -> dict[str, Any]:
    """Return table evidence coverage for retrieved context items."""
    items = context_items or []
    table_items = []
    for index, item in enumerate(items):
        text = _raw_text(item)
        table_type = _table_type(text)
        if table_type:
            table_items.append(
                {
                    "item_id": result_id(item, index),
                    "index": index,
                    "table_type": table_type,
                }
            )

    total = len(items)
    table_count = len(table_items)
    recommendation = None
    if total and table_count == 0 and _is_table_or_numeric_query(query):
        recommendation = "Add table-structured context for numeric or comparative queries before answering."

    return {
        "total_items": total,
        "table_item_count": table_count,
        "table_ratio": round(table_count / total, 3) if total else 0.0,
        "table_items": table_items,
        "recommendation": recommendation,
    }


def _table_type(text: str) -> str | None:
    if _has_markdown_table(text):
        return "markdown"
    if _has_delimited_rows(text):
        return "delimited"
    return None


def _has_markdown_table(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for left, right in zip(lines, lines[1:]):
        if "|" not in left or "|" not in right:
            continue
        separator = right.replace("|", "").replace(" ", "")
        if separator and set(separator) <= {"-", ":"} and left.strip("|").count("|") >= 1:
            return True
    return False


def _has_delimited_rows(text: str) -> bool:
    rows = [line.strip() for line in text.splitlines() if line.strip()]
    for delimiter in (",", "\t", ";"):
        counts = [row.count(delimiter) for row in rows if row.count(delimiter) >= 2]
        if len(counts) >= 2 and len(set(counts[:2])) == 1:
            return True
    return False


def _raw_text(item: Any) -> str:
    parts = []
    for key in ("title", "content", "text", "summary", "snippet"):
        raw = value(item, key)
        if raw is MISSING or raw is None:
            continue
        text = str(getattr(raw, "value", raw)).strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


def _is_table_or_numeric_query(query: str) -> bool:
    text = string(query) or ""
    return bool(text and (_TABLE_QUERY_RE.search(text) or _NUMERIC_RE.search(text)))
