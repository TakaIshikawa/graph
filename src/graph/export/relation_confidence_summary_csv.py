"""CSV export for relation confidence summaries."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeEdge

_FIELDNAMES = [
    "relation",
    "source",
    "source_project",
    "edge_count",
    "confidence_count",
    "missing_confidence_count",
    "min_confidence",
    "max_confidence",
    "average_confidence",
    "low_confidence_count",
    "medium_confidence_count",
    "high_confidence_count",
]
_PROJECT_KEYS = ("source_project", "project")
_WHITESPACE_RE = re.compile(r"\s+")


def export_relation_confidence_summary_csv(
    edges: Iterable[KnowledgeEdge],
    path: str | Path | None = None,
    *,
    low_threshold: float = 0.5,
    high_threshold: float = 0.8,
) -> str | dict[str, Any]:
    """Return or write confidence statistics grouped by relation, source, and project."""
    _validate_thresholds(low_threshold, high_threshold)

    edge_list = list(edges)
    rows = _summary_rows(edge_list, low_threshold=low_threshold, high_threshold=high_threshold)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "edge_count": len(edge_list),
        "group_count": len(rows),
        "rows_exported": len(rows),
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "bytes_written": output_path.stat().st_size,
    }


def _summary_rows(
    edges: list[KnowledgeEdge],
    *,
    low_threshold: float,
    high_threshold: float,
) -> list[dict[str, str | int]]:
    groups: dict[tuple[str, str, str], list[float | None]] = defaultdict(list)
    for edge in edges:
        groups[(_edge_relation(edge), _edge_source(edge), _edge_project(edge))].append(_edge_confidence(edge))

    rows: list[dict[str, str | int]] = []
    for relation, source, source_project in sorted(
        groups, key=lambda key: (_sort_key(key[0]), _sort_key(key[1]), _sort_key(key[2]))
    ):
        values = groups[(relation, source, source_project)]
        confidences = [value for value in values if value is not None]
        rows.append(
            {
                "relation": relation,
                "source": source,
                "source_project": source_project,
                "edge_count": len(values),
                "confidence_count": len(confidences),
                "missing_confidence_count": len(values) - len(confidences),
                "min_confidence": _decimal(min(confidences)) if confidences else "",
                "max_confidence": _decimal(max(confidences)) if confidences else "",
                "average_confidence": _decimal(sum(confidences) / len(confidences)) if confidences else "",
                "low_confidence_count": sum(1 for value in confidences if value < low_threshold),
                "medium_confidence_count": sum(
                    1 for value in confidences if low_threshold <= value < high_threshold
                ),
                "high_confidence_count": sum(1 for value in confidences if value >= high_threshold),
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


def _edge_confidence(edge: KnowledgeEdge) -> float | None:
    metadata = edge.metadata if isinstance(edge.metadata, Mapping) else {}
    if "confidence" in metadata:
        return _number_or_none(metadata.get("confidence"))
    if "score" in metadata:
        return _number_or_none(metadata.get("score"))
    return _number_or_none(edge.weight)


def _number_or_none(value: object) -> float | None:
    if not _is_number(value):
        return None
    return float(value)


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _edge_relation(edge: KnowledgeEdge) -> str:
    return _field_value(edge.relation) or "Unknown"


def _edge_source(edge: KnowledgeEdge) -> str:
    return _field_value(edge.source) or "Unknown"


def _edge_project(edge: KnowledgeEdge) -> str:
    metadata = edge.metadata if isinstance(edge.metadata, Mapping) else {}
    for key in _PROJECT_KEYS:
        value = _field_value(metadata.get(key))
        if value:
            return value
    return "Unknown"


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
