"""CSV export for repeated ordered relationship pairs."""

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
    "from_unit_id",
    "to_unit_id",
    "relation_count",
    "edge_count",
    "sources",
    "relations",
    "min_weight",
    "max_weight",
    "average_weight",
    "bidirectional_pair_key",
]
_WHITESPACE_RE = re.compile(r"\s+")


def export_relation_pair_summary_csv(
    edges: Iterable[KnowledgeEdge],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write summary rows for edges grouped by ordered endpoint pair."""
    edge_list = list(edges)
    rows = _summary_rows(edge_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "edge_count": len(edge_list),
        "pair_count": len(rows),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _summary_rows(edges: list[KnowledgeEdge]) -> list[dict[str, str | int]]:
    groups: dict[tuple[str, str], list[KnowledgeEdge]] = defaultdict(list)
    for edge in edges:
        groups[(_inline_text(edge.from_unit_id), _inline_text(edge.to_unit_id))].append(edge)

    rows: list[dict[str, str | int]] = []
    for (from_unit_id, to_unit_id), pair_edges in sorted(
        groups.items(), key=lambda item: (_sort_key(item[0][0]), _sort_key(item[0][1]))
    ):
        relation_counts = Counter(_field_value(edge.relation) for edge in pair_edges)
        source_counts = Counter(_field_value(edge.source) for edge in pair_edges)
        weights = [_weight(edge.weight) for edge in pair_edges]
        rows.append(
            {
                "from_unit_id": from_unit_id,
                "to_unit_id": to_unit_id,
                "relation_count": len(relation_counts),
                "edge_count": len(pair_edges),
                "sources": _render_counts(source_counts),
                "relations": _render_counts(relation_counts),
                "min_weight": _decimal(min(weights)),
                "max_weight": _decimal(max(weights)),
                "average_weight": _decimal(sum(weights) / len(weights)),
                "bidirectional_pair_key": _bidirectional_pair_key(from_unit_id, to_unit_id),
            }
        )
    return rows


def _render_counts(counts: Counter[str]) -> str:
    return "; ".join(
        f"{label} ({count})" if count > 1 else label
        for label, count in sorted(counts.items(), key=lambda item: (_sort_key(item[0]), item[1]))
    )


def _bidirectional_pair_key(from_unit_id: str, to_unit_id: str) -> str:
    left, right = sorted([from_unit_id, to_unit_id], key=_sort_key)
    return f"{left}|{right}"


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _weight(value: object) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
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
