"""Extract table-like signals from retrieved RAG evidence."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from graph.rag._analysis_utils import MISSING, result_id, tokens, value


def extract_evidence_table_signals(query: str, evidence: Iterable[Any]) -> dict[str, Any]:
    """Detect markdown, delimited, and metadata table signals in evidence."""
    query_terms = tokens(query)
    rows: list[dict[str, Any]] = []
    columns: list[str] = []
    row_counts: dict[str, int] = {}

    for index, item in enumerate(evidence):
        rid = result_id(item, index)
        text = _raw_text(item)
        metadata_columns = _metadata_columns(value(item, "columns"))
        markdown_columns = _markdown_columns(text)
        delimiter_columns = _delimiter_columns(text)
        item_columns = _unique([*metadata_columns, *markdown_columns, *delimiter_columns])
        row_count = _row_count(text)
        if item_columns or row_count:
            rows.append({"result_id": rid, "columns": item_columns, "row_count": row_count})
            row_counts[rid] = row_count
            columns.extend(item_columns)

    detected_columns = _unique(columns)
    overlap = len(query_terms & {column.casefold() for column in detected_columns})
    suitability = 0.0 if not rows else min(1.0, 0.45 + len(rows) * 0.15 + overlap * 0.1)
    return {
        "table_like_results": rows,
        "detected_columns": detected_columns,
        "row_count_estimates": row_counts,
        "suitability_score": round(suitability, 2),
    }


def _metadata_columns(value_: Any) -> list[str]:
    if isinstance(value_, str):
        return [value_]
    if isinstance(value_, Iterable) and not isinstance(value_, bytes | str | Mapping):
        return [_clean(part) for part in value_ if _clean(part)]
    return []


def _markdown_columns(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for left, right in zip(lines, lines[1:]):
        if "|" in left and set(right.replace("|", "").strip()) <= {"-", ":"}:
            return [_clean(part) for part in left.strip("|").split("|") if _clean(part)]
    return []


def _delimiter_columns(text: str) -> list[str]:
    for line in text.splitlines():
        delimiter = "," if line.count(",") >= 2 else "\t" if line.count("\t") >= 2 else ""
        if delimiter:
            return [_clean(part) for part in line.split(delimiter) if _clean(part)]
    return []


def _row_count(text: str) -> int:
    count = 0
    for line in text.splitlines():
        stripped = line.strip()
        if "|" in stripped or stripped.count(",") >= 2 or stripped.count("\t") >= 2:
            count += 1
    return max(0, count - 1) if count else 0


def _raw_text(item: Any) -> str:
    parts = []
    for key in ("title", "content", "text", "summary", "snippet"):
        raw = value(item, key)
        if raw is not MISSING and raw is not None:
            text = str(getattr(raw, "value", raw)).strip()
            if text:
                parts.append(text)
    return "\n".join(parts)


def _clean(value_: Any) -> str:
    return " ".join(str(value_).strip().strip("|").split())


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value_ in values:
        key = value_.casefold()
        if value_ and key not in seen:
            seen.add(key)
            out.append(value_)
    return out
