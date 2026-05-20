"""CSV export for edge metadata key coverage."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeEdge

_FIELDNAMES = ["relation", "source", "metadata_key", "edge_count", "edges_with_key", "coverage_percent"]
_WHITESPACE_RE = re.compile(r"\s+")


def export_edge_metadata_key_coverage_csv(
    edges: Iterable[KnowledgeEdge | Mapping[str, Any]],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write metadata key presence by relation and edge source."""
    edge_list = list(edges)
    rows = _coverage_rows(edge_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "edge_count": len(edge_list),
        "coverage_key_count": len(rows),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _coverage_rows(edges: list[KnowledgeEdge | Mapping[str, Any]]) -> list[dict[str, str | int]]:
    groups: dict[tuple[str, str], list[KnowledgeEdge | Mapping[str, Any]]] = defaultdict(list)
    for edge in edges:
        groups[(_edge_relation(edge), _edge_source(edge))].append(edge)

    rows: list[dict[str, str | int]] = []
    for (relation, source), group_edges in sorted(
        groups.items(), key=lambda item: (_sort_key(item[0][0]), _sort_key(item[0][1]))
    ):
        metadata_keys = sorted(
            {
                _inline_text(key)
                for edge in group_edges
                for key in _metadata(edge)
                if _inline_text(key)
            },
            key=_sort_key,
        )
        for metadata_key in metadata_keys:
            edges_with_key = sum(1 for edge in group_edges if metadata_key in _metadata(edge))
            rows.append(
                {
                    "relation": relation,
                    "source": source,
                    "metadata_key": metadata_key,
                    "edge_count": len(group_edges),
                    "edges_with_key": edges_with_key,
                    "coverage_percent": f"{edges_with_key * 100 / len(group_edges):.2f}",
                }
            )
    return rows


def _metadata(edge: KnowledgeEdge | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _get(edge, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _edge_relation(edge: KnowledgeEdge | Mapping[str, Any]) -> str:
    return _field_value(_get(edge, "relation")) or "Unknown"


def _edge_source(edge: KnowledgeEdge | Mapping[str, Any]) -> str:
    return _field_value(_get(edge, "source")) or "Unknown"


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


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
