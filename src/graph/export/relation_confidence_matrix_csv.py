"""CSV export for relation confidence bucket counts."""

from __future__ import annotations

import csv
import re
from collections import Counter
from collections.abc import Iterable
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeEdge

_FIELDNAMES = [
    "relation",
    "low_count",
    "medium_count",
    "high_count",
    "unknown_count",
    "total_count",
]
_BUCKETS = ("low", "medium", "high", "unknown")
_WHITESPACE_RE = re.compile(r"\s+")


def export_relation_confidence_matrix_csv(
    edges: Iterable[KnowledgeEdge],
    path: str | Path | None = None,
    *,
    low_threshold: float = 0.5,
    high_threshold: float = 0.8,
) -> str | dict[str, Any]:
    """Return or write edge confidence bucket counts grouped by relation."""
    _validate_thresholds(low_threshold, high_threshold)

    edge_list = list(edges)
    rows = _matrix_rows(edge_list, low_threshold=low_threshold, high_threshold=high_threshold)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "edge_count": len(edge_list),
        "relation_count": len(rows),
        "rows_exported": len(rows),
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "bytes_written": output_path.stat().st_size,
    }


def _matrix_rows(
    edges: list[KnowledgeEdge],
    *,
    low_threshold: float,
    high_threshold: float,
) -> list[dict[str, str | int]]:
    counts: Counter[tuple[str, str]] = Counter()

    for edge in edges:
        relation = _field_value(edge.relation) or "Unknown"
        bucket = _confidence_bucket(
            _edge_confidence(edge),
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )
        counts[(relation, bucket)] += 1

    rows: list[dict[str, str | int]] = []
    relations = sorted({relation for relation, _bucket in counts}, key=_sort_key)
    for relation in relations:
        bucket_counts = {bucket: counts[(relation, bucket)] for bucket in _BUCKETS}
        rows.append(
            {
                "relation": relation,
                "low_count": bucket_counts["low"],
                "medium_count": bucket_counts["medium"],
                "high_count": bucket_counts["high"],
                "unknown_count": bucket_counts["unknown"],
                "total_count": sum(bucket_counts.values()),
            }
        )
    return rows


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _validate_thresholds(low_threshold: float, high_threshold: float) -> None:
    if not _is_number(low_threshold):
        raise ValueError("low_threshold must be a number between 0 and 1")
    if not _is_number(high_threshold):
        raise ValueError("high_threshold must be a number between 0 and 1")
    if not 0 <= low_threshold <= 1:
        raise ValueError("low_threshold must be between 0 and 1")
    if not 0 <= high_threshold <= 1:
        raise ValueError("high_threshold must be between 0 and 1")
    if low_threshold >= high_threshold:
        raise ValueError("low_threshold must be less than high_threshold")


def _edge_confidence(edge: KnowledgeEdge) -> object:
    value = getattr(edge, "confidence", None)
    if value is not None:
        return value
    metadata = edge.metadata if isinstance(edge.metadata, dict) else {}
    return metadata.get("confidence")


def _confidence_bucket(
    value: object,
    *,
    low_threshold: float,
    high_threshold: float,
) -> str:
    confidence = _confidence_value(value)
    if confidence is None:
        return "unknown"
    if confidence < low_threshold:
        return "low"
    if confidence >= high_threshold:
        return "high"
    return "medium"


def _confidence_value(value: object) -> float | None:
    if not _is_number(value):
        return None
    return float(value)


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
