"""CSV export for edge weight summaries by relation and source."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeEdge

_FIELDNAMES = [
    "relation",
    "source",
    "edge_count",
    "min_weight",
    "max_weight",
    "average_weight",
    "weak_edge_count",
    "strong_edge_count",
]
_WHITESPACE_RE = re.compile(r"\s+")


def export_edge_weight_summary_csv(
    edges: Iterable[KnowledgeEdge],
    path: str | Path | None = None,
    *,
    weak_threshold: float = 0.25,
    strong_threshold: float = 0.75,
) -> str | dict[str, Any]:
    """Return or write edge weight statistics grouped by relation and source."""
    _validate_thresholds(weak_threshold, strong_threshold)

    edge_list = list(edges)
    rows = _summary_rows(edge_list, weak_threshold=weak_threshold, strong_threshold=strong_threshold)
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
        "weak_threshold": weak_threshold,
        "strong_threshold": strong_threshold,
        "bytes_written": output_path.stat().st_size,
    }


def _summary_rows(
    edges: list[KnowledgeEdge],
    *,
    weak_threshold: float,
    strong_threshold: float,
) -> list[dict[str, str | int]]:
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for edge in edges:
        groups[(_edge_relation(edge), _edge_source(edge))].append(_weight(edge.weight))

    rows: list[dict[str, str | int]] = []
    for relation, source in sorted(groups, key=lambda key: (_sort_key(key[0]), _sort_key(key[1]))):
        weights = groups[(relation, source)]
        rows.append(
            {
                "relation": relation,
                "source": source,
                "edge_count": len(weights),
                "min_weight": _decimal(min(weights)),
                "max_weight": _decimal(max(weights)),
                "average_weight": _decimal(sum(weights) / len(weights)),
                "weak_edge_count": sum(1 for weight in weights if weight < weak_threshold),
                "strong_edge_count": sum(1 for weight in weights if weight >= strong_threshold),
            }
        )
    return rows


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _validate_thresholds(weak_threshold: float, strong_threshold: float) -> None:
    if not _is_number(weak_threshold):
        raise ValueError("weak_threshold must be a number between 0 and 1")
    if not _is_number(strong_threshold):
        raise ValueError("strong_threshold must be a number between 0 and 1")
    if not 0 <= weak_threshold <= 1:
        raise ValueError("weak_threshold must be between 0 and 1")
    if not 0 <= strong_threshold <= 1:
        raise ValueError("strong_threshold must be between 0 and 1")
    if weak_threshold >= strong_threshold:
        raise ValueError("weak_threshold must be less than strong_threshold")


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _weight(value: object) -> float:
    if not _is_number(value):
        return 0.0
    return float(value)


def _edge_relation(edge: KnowledgeEdge) -> str:
    return _field_value(edge.relation) or "Unknown"


def _edge_source(edge: KnowledgeEdge) -> str:
    return _field_value(edge.source) or "Unknown"


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
