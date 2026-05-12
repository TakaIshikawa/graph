"""CSV export for edge relation/source provenance."""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeEdge

_FIELDNAMES = [
    "relation",
    "edge_source",
    "edge_count",
    "total_weight",
    "average_weight",
    "percent_of_relation",
]
_WHITESPACE_RE = re.compile(r"\s+")


def export_edge_relation_source_csv(
    edges: Iterable[KnowledgeEdge],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write edge counts and weights by relation and edge source."""
    edge_list = list(edges)
    rows = _relation_source_rows(edge_list)
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
        "relation_count": len({_field_value(edge.relation) for edge in edge_list}),
        "bytes_written": output_path.stat().st_size,
    }


def _relation_source_rows(edges: list[KnowledgeEdge]) -> list[dict[str, str | int]]:
    counts: Counter[tuple[str, str]] = Counter()
    weights: dict[tuple[str, str], float] = defaultdict(float)
    relation_totals: Counter[str] = Counter()

    for edge in edges:
        relation = _field_value(edge.relation) or "Unknown"
        edge_source = _field_value(edge.source) or "Unknown"
        key = (relation, edge_source)
        counts[key] += 1
        relation_totals[relation] += 1
        weights[key] += _weight(edge.weight)

    rows: list[dict[str, str | int]] = []
    for relation, edge_source in sorted(counts, key=lambda item: (_sort_key(item[0]), _sort_key(item[1]))):
        edge_count = counts[(relation, edge_source)]
        total_weight = weights[(relation, edge_source)]
        rows.append(
            {
                "relation": relation,
                "edge_source": edge_source,
                "edge_count": edge_count,
                "total_weight": _decimal(total_weight),
                "average_weight": _decimal(total_weight / edge_count),
                "percent_of_relation": _decimal(edge_count * 100 / relation_totals[relation]),
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
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    return 0.0


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
