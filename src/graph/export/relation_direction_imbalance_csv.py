"""CSV export for directional imbalance within relation/source groups."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeEdge

_FIELDNAMES = ["relation", "source", "pair_count", "forward_only_count", "reverse_only_count", "bidirectional_count", "imbalance_percent", "sample_pairs"]
_WHITESPACE_RE = re.compile(r"\s+")


def export_relation_direction_imbalance_csv(
    edges: Iterable[KnowledgeEdge | Mapping[str, Any]],
    path: str | Path | None = None,
    *,
    min_pair_count: int = 1,
) -> str | dict[str, Any]:
    """Return or write direction coverage counts for unordered endpoint pairs."""
    min_pair_count = _positive_int(min_pair_count, "min_pair_count")
    edge_list = list(edges)
    rows = _imbalance_rows(edge_list, min_pair_count)
    text = _render_csv(rows)
    if path is None:
        return text
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {"path": str(output_path), "edge_count": len(edge_list), "rows_exported": len(rows), "min_pair_count": min_pair_count, "bytes_written": output_path.stat().st_size}


def _imbalance_rows(edges: list[KnowledgeEdge | Mapping[str, Any]], min_pair_count: int) -> list[dict[str, str | int]]:
    groups: dict[tuple[str, str], dict[tuple[str, str], set[str]]] = defaultdict(lambda: defaultdict(set))
    for edge in edges:
        source_id = _source_id(edge)
        target_id = _target_id(edge)
        if not source_id or not target_id or source_id == target_id:
            continue
        a, b = sorted((source_id, target_id), key=_sort_key)
        direction = "forward" if (a, b) == (source_id, target_id) else "reverse"
        groups[(_relation(edge), _source(edge))][(a, b)].add(direction)
    rows: list[dict[str, str | int]] = []
    for (relation, source), pairs in groups.items():
        if len(pairs) < min_pair_count:
            continue
        forward_only = sum(1 for directions in pairs.values() if directions == {"forward"})
        reverse_only = sum(1 for directions in pairs.values() if directions == {"reverse"})
        bidirectional = sum(1 for directions in pairs.values() if len(directions) == 2)
        imbalance = abs(forward_only - reverse_only) / len(pairs) * 100.0
        samples = [f"{a}->{b}" for a, b in sorted(pairs, key=lambda pair: (_sort_key(pair[0]), _sort_key(pair[1])))[:5]]
        rows.append({"relation": relation, "source": source, "pair_count": len(pairs), "forward_only_count": forward_only, "reverse_only_count": reverse_only, "bidirectional_count": bidirectional, "imbalance_percent": _decimal(imbalance), "sample_pairs": "; ".join(samples)})
    return sorted(rows, key=lambda row: (_sort_key(row["relation"]), _sort_key(row["source"])))


def _source_id(edge: KnowledgeEdge | Mapping[str, Any]) -> str:
    return _field_value(_first_value(edge, ("from_unit_id", "source_id", "from_id", "source")))


def _target_id(edge: KnowledgeEdge | Mapping[str, Any]) -> str:
    return _field_value(_first_value(edge, ("to_unit_id", "target_id", "to_id", "target")))


def _relation(edge: KnowledgeEdge | Mapping[str, Any]) -> str:
    return _field_value(_first_value(edge, ("relation", "relation_type", "type"))) or "Unknown"


def _source(edge: KnowledgeEdge | Mapping[str, Any]) -> str:
    return _field_value(_first_value(edge, ("source", "source_project", "provider"))) or "Unknown"


def _first_value(value: object, keys: tuple[str, ...], default: object = "") -> object:
    for key in keys:
        item = _get(value, key, None)
        if item is not None:
            return item
    metadata = _get(value, "metadata", {})
    if isinstance(metadata, Mapping):
        for key in keys:
            if key in metadata:
                return metadata[key]
    return default


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


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


def _decimal(value: float) -> str:
    return f"{value:.2f}"
