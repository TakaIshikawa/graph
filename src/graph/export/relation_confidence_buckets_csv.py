"""CSV export for relation confidence buckets."""

from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from collections.abc import Iterable
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeEdge

_FIELDNAMES = [
    "relation",
    "bucket",
    "edge_count",
    "average_confidence",
    "unknown_confidence_count",
]
_UNKNOWN_BUCKET = "Unknown"
_WHITESPACE_RE = re.compile(r"\s+")


def export_relation_confidence_buckets_csv(
    edges: Iterable[KnowledgeEdge],
    path: str | Path | None = None,
    *,
    bucket_size: float = 0.25,
) -> str | dict[str, Any]:
    """Return or write confidence bucket counts grouped by relation."""
    _validate_bucket_size(bucket_size)

    edge_list = list(edges)
    rows = _bucket_rows(edge_list, bucket_size=float(bucket_size))
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "edge_count": len(edge_list),
        "relation_count": len({row["relation"] for row in rows}),
        "rows_exported": len(rows),
        "bucket_size": bucket_size,
        "bytes_written": output_path.stat().st_size,
    }


def _bucket_rows(
    edges: list[KnowledgeEdge],
    *,
    bucket_size: float,
) -> list[dict[str, str | int]]:
    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    unknown_counts: dict[str, int] = defaultdict(int)

    for edge in edges:
        relation = _field_value(edge.relation) or "Unknown"
        confidence = _confidence_value(_edge_confidence(edge))
        if confidence is None:
            unknown_counts[relation] += 1
            continue
        buckets[relation][_bucket_label(confidence, bucket_size)].append(confidence)

    rows: list[dict[str, str | int]] = []
    relations = sorted(set(buckets) | set(unknown_counts), key=_sort_key)
    for relation in relations:
        relation_unknown_count = unknown_counts[relation]
        if buckets[relation]:
            for bucket in sorted(buckets[relation], key=_bucket_sort_key):
                values = buckets[relation][bucket]
                rows.append(
                    {
                        "relation": relation,
                        "bucket": bucket,
                        "edge_count": len(values),
                        "average_confidence": _decimal(sum(values) / len(values)),
                        "unknown_confidence_count": relation_unknown_count,
                    }
                )
        elif relation_unknown_count:
            rows.append(
                {
                    "relation": relation,
                    "bucket": _UNKNOWN_BUCKET,
                    "edge_count": 0,
                    "average_confidence": "",
                    "unknown_confidence_count": relation_unknown_count,
                }
            )
    return rows


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _validate_bucket_size(bucket_size: float) -> None:
    if not _is_number(bucket_size) or not 0 < float(bucket_size) <= 1:
        raise ValueError("bucket_size must be a positive number not greater than 1")


def _edge_confidence(edge: KnowledgeEdge) -> object:
    value = getattr(edge, "confidence", None)
    if value is not None:
        return value
    metadata = edge.metadata if isinstance(edge.metadata, dict) else {}
    return metadata.get("confidence")


def _confidence_value(value: object) -> float | None:
    if not _is_number(value):
        return None
    confidence = float(value)
    if not 0 <= confidence <= 1:
        return None
    return confidence


def _bucket_label(confidence: float, bucket_size: float) -> str:
    if confidence == 1:
        index = math.ceil(1 / bucket_size) - 1
    else:
        index = math.floor(confidence / bucket_size)
    start = index * bucket_size
    end = min(start + bucket_size, 1)
    return f"{start:.2f}-{end:.2f}"


def _bucket_sort_key(bucket: str) -> float:
    if bucket == _UNKNOWN_BUCKET:
        return math.inf
    return float(bucket.split("-", 1)[0])


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


def _decimal(value: float) -> str:
    return f"{value:.2f}"
