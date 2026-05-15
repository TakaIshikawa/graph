"""CSV export for reciprocal and one-way relation pairs."""

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
    "forward_relations",
    "reverse_relations",
    "forward_edge_count",
    "reverse_edge_count",
    "reciprocity_status",
]
_WHITESPACE_RE = re.compile(r"\s+")


def export_relation_reciprocity_csv(
    edges: Iterable[KnowledgeEdge],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write relation reciprocity rows grouped by unordered endpoint pair."""
    edge_list = list(edges)
    rows = _reciprocity_rows(edge_list)
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


def _reciprocity_rows(edges: list[KnowledgeEdge]) -> list[dict[str, str | int]]:
    groups: dict[tuple[str, str], dict[str, list[KnowledgeEdge]]] = defaultdict(lambda: {"forward": [], "reverse": []})
    for edge in edges:
        from_unit_id = _field_value(edge.from_unit_id)
        to_unit_id = _field_value(edge.to_unit_id)
        canonical_from, canonical_to = _canonical_pair(from_unit_id, to_unit_id)
        direction = "forward" if (from_unit_id, to_unit_id) == (canonical_from, canonical_to) else "reverse"
        groups[(canonical_from, canonical_to)][direction].append(edge)

    rows: list[dict[str, str | int]] = []
    for (from_unit_id, to_unit_id), directed_edges in groups.items():
        forward_edges = directed_edges["forward"]
        reverse_edges = directed_edges["reverse"]
        rows.append(
            {
                "from_unit_id": from_unit_id,
                "to_unit_id": to_unit_id,
                "forward_relations": _relation_counts(forward_edges),
                "reverse_relations": _relation_counts(reverse_edges),
                "forward_edge_count": len(forward_edges),
                "reverse_edge_count": len(reverse_edges),
                "reciprocity_status": "reciprocal" if forward_edges and reverse_edges else "one_way",
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            _sort_key(row["reciprocity_status"]),
            _sort_key(row["from_unit_id"]),
            _sort_key(row["to_unit_id"]),
        ),
    )


def _relation_counts(edges: list[KnowledgeEdge]) -> str:
    counts = Counter(_field_value(edge.relation) for edge in edges)
    return "; ".join(
        f"{relation} ({count})" if count > 1 else relation
        for relation, count in sorted(counts.items(), key=lambda item: (_sort_key(item[0]), item[1]))
    )


def _canonical_pair(from_unit_id: str, to_unit_id: str) -> tuple[str, str]:
    return tuple(sorted([from_unit_id, to_unit_id], key=_sort_key))  # type: ignore[return-value]


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
