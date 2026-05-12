"""CSV export for relation weight distribution summaries."""

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
    "zero_weight_count",
    "negative_weight_count",
    "strong_edge_count",
]
_WHITESPACE_RE = re.compile(r"\s+")


def export_relation_weight_distribution_csv(
    edges: Iterable[KnowledgeEdge],
    path: str | Path | None = None,
    *,
    strong_threshold: float = 1.0,
) -> str | dict[str, Any]:
    """Return or write relation/source weight distribution rows."""
    if not _is_number(strong_threshold):
        raise ValueError("strong_threshold must be numeric")

    threshold = float(strong_threshold)
    edge_list = list(edges)
    rows = _distribution_rows(edge_list, strong_threshold=threshold)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "edge_count": len(edge_list),
        "rows_exported": len(rows),
        "strong_threshold": threshold,
        "bytes_written": output_path.stat().st_size,
    }


def _distribution_rows(edges: list[KnowledgeEdge], *, strong_threshold: float) -> list[dict[str, str | int]]:
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for edge in edges:
        groups[(_field_value(edge.relation) or "Unknown", _field_value(edge.source) or "Unknown")].append(
            _weight(edge.weight)
        )

    rows: list[dict[str, str | int]] = []
    for (relation, source), weights in sorted(groups.items(), key=lambda item: (_sort_key(item[0][0]), _sort_key(item[0][1]))):
        rows.append(
            {
                "relation": relation,
                "source": source,
                "edge_count": len(weights),
                "min_weight": _decimal(min(weights)),
                "max_weight": _decimal(max(weights)),
                "average_weight": _decimal(sum(weights) / len(weights)),
                "zero_weight_count": sum(1 for weight in weights if weight == 0),
                "negative_weight_count": sum(1 for weight in weights if weight < 0),
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


def _weight(value: object) -> float:
    if _is_number(value):
        return float(value)
    return 0.0


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
