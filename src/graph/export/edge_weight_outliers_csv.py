"""CSV export for high and low edge weight outliers."""

from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeEdge

_FIELDNAMES = ["relation", "source", "edge_id", "from_unit_id", "to_unit_id", "weight", "group_mean", "group_stddev", "zscore", "direction"]
_WHITESPACE_RE = re.compile(r"\s+")


def export_edge_weight_outliers_csv(
    edges: Iterable[KnowledgeEdge | Mapping[str, Any]],
    path: str | Path | None = None,
    *,
    min_group_size: int = 3,
    zscore_threshold: float = 2.0,
) -> str | dict[str, Any]:
    """Return or write edge weight outliers within relation/source groups."""
    min_group_size = _positive_int(min_group_size, "min_group_size")
    zscore_threshold = _positive_float(zscore_threshold, "zscore_threshold")
    edge_list = list(edges)
    rows = _outlier_rows(edge_list, min_group_size, zscore_threshold)
    text = _render_csv(rows)
    if path is None:
        return text
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {"path": str(output_path), "edge_count": len(edge_list), "rows_exported": len(rows), "min_group_size": min_group_size, "zscore_threshold": zscore_threshold, "bytes_written": output_path.stat().st_size}


def _outlier_rows(edges: list[KnowledgeEdge | Mapping[str, Any]], min_group_size: int, zscore_threshold: float) -> list[dict[str, str]]:
    groups: dict[tuple[str, str], list[KnowledgeEdge | Mapping[str, Any]]] = defaultdict(list)
    for edge in edges:
        groups[(_relation(edge), _source(edge))].append(edge)
    rows: list[dict[str, str]] = []
    for (relation, source), group_edges in groups.items():
        if len(group_edges) < min_group_size:
            continue
        weights = [_weight(edge) for edge in group_edges]
        mean = sum(weights) / len(weights)
        variance = sum((weight - mean) ** 2 for weight in weights) / len(weights)
        stddev = math.sqrt(variance)
        if stddev == 0.0:
            continue
        for edge, weight in zip(group_edges, weights, strict=True):
            zscore = (weight - mean) / stddev
            if abs(zscore) < zscore_threshold:
                continue
            rows.append({"relation": relation, "source": source, "edge_id": _edge_id(edge), "from_unit_id": _source_id(edge), "to_unit_id": _target_id(edge), "weight": _decimal(weight), "group_mean": _decimal(mean), "group_stddev": _decimal(stddev), "zscore": _decimal(zscore), "direction": "high" if zscore > 0 else "low"})
    return sorted(rows, key=lambda row: (_sort_key(row["relation"]), _sort_key(row["source"]), _sort_key(row["edge_id"])))


def _source_id(edge: KnowledgeEdge | Mapping[str, Any]) -> str:
    return _field_value(_first_value(edge, ("from_unit_id", "source_id", "from_id", "source")))


def _target_id(edge: KnowledgeEdge | Mapping[str, Any]) -> str:
    return _field_value(_first_value(edge, ("to_unit_id", "target_id", "to_id", "target")))


def _edge_id(edge: KnowledgeEdge | Mapping[str, Any]) -> str:
    return _field_value(_get(edge, "id")) or f"{_source_id(edge)}->{_target_id(edge)}"


def _relation(edge: KnowledgeEdge | Mapping[str, Any]) -> str:
    return _field_value(_first_value(edge, ("relation", "relation_type", "type"))) or "Unknown"


def _source(edge: KnowledgeEdge | Mapping[str, Any]) -> str:
    return _field_value(_first_value(edge, ("source", "source_project", "provider"))) or "Unknown"


def _weight(edge: KnowledgeEdge | Mapping[str, Any]) -> float:
    try:
        return float(_get(edge, "weight", 1.0))
    except (TypeError, ValueError):
        return 1.0


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


def _positive_float(value: object, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive") from exc
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _render_csv(rows: list[dict[str, str]]) -> str:
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
