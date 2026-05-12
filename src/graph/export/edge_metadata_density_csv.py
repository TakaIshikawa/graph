"""CSV export for per-relation edge metadata density."""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeEdge

_FIELDNAMES = [
    "relation",
    "edge_count",
    "edges_with_metadata",
    "metadata_coverage_percent",
    "distinct_metadata_keys",
    "average_keys_per_edge",
    "top_metadata_keys",
]
_WHITESPACE_RE = re.compile(r"\s+")


def export_edge_metadata_density_csv(
    edges: Iterable[KnowledgeEdge],
    path: str | Path | None = None,
    *,
    min_edges: int = 1,
) -> str | dict[str, Any]:
    """Return or write a deterministic per-relation edge metadata density CSV."""
    if not isinstance(min_edges, int) or isinstance(min_edges, bool) or min_edges < 1:
        raise ValueError("min_edges must be a positive integer")

    edge_list = list(edges)
    rows = _density_rows(edge_list, min_edges=min_edges)
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
        "min_edges": min_edges,
        "bytes_written": output_path.stat().st_size,
    }


def _density_rows(edges: list[KnowledgeEdge], *, min_edges: int) -> list[dict[str, str | int]]:
    groups: dict[str, list[KnowledgeEdge]] = defaultdict(list)
    for edge in edges:
        groups[_edge_relation(edge)].append(edge)

    rows: list[dict[str, str | int]] = []
    for relation in sorted(groups, key=_sort_key):
        relation_edges = groups[relation]
        if len(relation_edges) < min_edges:
            continue

        key_counts: Counter[str] = Counter()
        total_keys = 0
        edges_with_metadata = 0
        for edge in relation_edges:
            keys = _metadata_keys(edge)
            if keys:
                edges_with_metadata += 1
            total_keys += len(keys)
            key_counts.update(keys)

        edge_count = len(relation_edges)
        rows.append(
            {
                "relation": relation,
                "edge_count": edge_count,
                "edges_with_metadata": edges_with_metadata,
                "metadata_coverage_percent": _decimal(edges_with_metadata * 100 / edge_count),
                "distinct_metadata_keys": len(key_counts),
                "average_keys_per_edge": _decimal(total_keys / edge_count),
                "top_metadata_keys": _top_keys(key_counts),
            }
        )
    return rows


def _metadata_keys(edge: KnowledgeEdge) -> list[str]:
    metadata = edge.metadata if isinstance(edge.metadata, Mapping) else {}
    return [_inline_text(key) for key in metadata if _inline_text(key)]


def _top_keys(key_counts: Counter[str]) -> str:
    return "; ".join(
        f"{key} ({count})"
        for key, count in sorted(key_counts.items(), key=lambda item: (-item[1], _sort_key(item[0])))
    )


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _edge_relation(edge: KnowledgeEdge) -> str:
    return _field_value(edge.relation) or "Unknown"


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
