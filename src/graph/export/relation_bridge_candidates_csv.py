"""CSV export for relation bridge candidate edges."""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeEdge

_FIELDNAMES = [
    "relation",
    "relation_type",
    "source_unit_id",
    "target_unit_id",
    "source_degree",
    "target_degree",
    "component_delta_if_removed",
    "shared_neighbor_count",
    "bridge_score",
]
_UNKNOWN = "Unknown"
_TYPE_KEYS = ("relation_type", "type", "edge_type")
_WHITESPACE_RE = re.compile(r"\s+")


def export_relation_bridge_candidates_csv(
    edges: Iterable[KnowledgeEdge | Mapping[str, Any]],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write bridge-candidate context for relation edges."""
    edge_list = list(edges)
    rows = _bridge_rows(edge_list)
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
        "bytes_written": output_path.stat().st_size,
    }


def _bridge_rows(edges: list[KnowledgeEdge | Mapping[str, Any]]) -> list[dict[str, str | int]]:
    adjacency, pair_counts = _graph(edges)
    rows: list[dict[str, str | int]] = []
    for edge in edges:
        source = _field_value(_get(edge, "from_unit_id"))
        target = _field_value(_get(edge, "to_unit_id"))
        if not source or not target:
            continue
        source_neighbors = adjacency.get(source, set())
        target_neighbors = adjacency.get(target, set())
        shared_neighbor_count = len((source_neighbors & target_neighbors) - {source, target})
        component_delta = _component_delta(adjacency, pair_counts, source, target)
        rows.append(
            {
                "relation": _field_value(_get(edge, "relation")) or _UNKNOWN,
                "relation_type": _relation_type(edge),
                "source_unit_id": source,
                "target_unit_id": target,
                "source_degree": len(source_neighbors),
                "target_degree": len(target_neighbors),
                "component_delta_if_removed": component_delta,
                "shared_neighbor_count": shared_neighbor_count,
                "bridge_score": _bridge_score(
                    component_delta=component_delta,
                    source_degree=len(source_neighbors),
                    target_degree=len(target_neighbors),
                    shared_neighbor_count=shared_neighbor_count,
                ),
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            -int(row["component_delta_if_removed"]),
            -float(str(row["bridge_score"])),
            _sort_key(row["relation"]),
            _sort_key(row["source_unit_id"]),
            _sort_key(row["target_unit_id"]),
        ),
    )


def _graph(
    edges: list[KnowledgeEdge | Mapping[str, Any]],
) -> tuple[dict[str, set[str]], Counter[tuple[str, str]]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    pair_counts: Counter[tuple[str, str]] = Counter()
    for edge in edges:
        source = _field_value(_get(edge, "from_unit_id"))
        target = _field_value(_get(edge, "to_unit_id"))
        if not source or not target or source == target:
            continue
        adjacency[source].add(target)
        adjacency[target].add(source)
        pair_counts[_pair_key(source, target)] += 1
    return adjacency, pair_counts


def _component_delta(
    adjacency: dict[str, set[str]],
    pair_counts: Counter[tuple[str, str]],
    source: str,
    target: str,
) -> int:
    if pair_counts[_pair_key(source, target)] > 1:
        return 0
    visited = _reachable(adjacency, source, removed_pair=(source, target))
    return 1 if target not in visited else 0


def _reachable(
    adjacency: dict[str, set[str]],
    start: str,
    *,
    removed_pair: tuple[str, str],
) -> set[str]:
    removed = _pair_key(*removed_pair)
    visited: set[str] = set()
    queue: deque[str] = deque([start])
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        for neighbor in sorted(adjacency.get(node, set()), key=_sort_key):
            if _pair_key(node, neighbor) == removed or neighbor in visited:
                continue
            queue.append(neighbor)
    return visited


def _bridge_score(
    *,
    component_delta: int,
    source_degree: int,
    target_degree: int,
    shared_neighbor_count: int,
) -> str:
    score = (component_delta * 10) + (1 / max(source_degree, 1)) + (1 / max(target_degree, 1))
    score -= shared_neighbor_count * 0.25
    return f"{max(score, 0):.2f}"


def _relation_type(edge: KnowledgeEdge | Mapping[str, Any]) -> str:
    metadata = _metadata(edge)
    for key in _TYPE_KEYS:
        value = _field_value(metadata.get(key))
        if value:
            return value
    return _UNKNOWN


def _metadata(edge: KnowledgeEdge | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _get(edge, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _pair_key(source: str, target: str) -> tuple[str, str]:
    return tuple(sorted((source, target), key=_sort_key))


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
