"""CSV export for edge weight distribution by relation/type."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeEdge

_FIELDNAMES = ["edge_type", "edge_count", "min_weight", "max_weight", "average_weight", "low_count", "medium_count", "high_count", "missing_weight_count"]
_WHITESPACE_RE = re.compile(r"\s+")


def export_edge_weight_distribution_csv(
    edges: Iterable[KnowledgeEdge | Mapping[str, Any]],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write edge weight distribution rows grouped by relation/type."""
    edge_list = list(edges)
    rows = _distribution_rows(edge_list)
    text = _render_csv(rows)
    if path is None:
        return text
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {"path": str(output_path), "edge_count": len(edge_list), "rows_exported": len(rows), "bytes_written": output_path.stat().st_size}


def _distribution_rows(edges: list[KnowledgeEdge | Mapping[str, Any]]) -> list[dict[str, str | int]]:
    groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"weights": [], "missing": 0, "low": 0, "medium": 0, "high": 0})
    for edge in edges:
        group = groups[_edge_type(edge)]
        weight = _weight(edge)
        if weight is None:
            group["missing"] += 1
            continue
        group["weights"].append(weight)
        if weight < 0.34:
            group["low"] += 1
        elif weight < 0.67:
            group["medium"] += 1
        else:
            group["high"] += 1
    rows: list[dict[str, str | int]] = []
    for edge_type, group in groups.items():
        weights = group["weights"]
        edge_count = len(weights) + group["missing"]
        rows.append(
            {
                "edge_type": edge_type,
                "edge_count": edge_count,
                "min_weight": _decimal(min(weights)) if weights else "",
                "max_weight": _decimal(max(weights)) if weights else "",
                "average_weight": _decimal(sum(weights) / len(weights)) if weights else "",
                "low_count": group["low"],
                "medium_count": group["medium"],
                "high_count": group["high"],
                "missing_weight_count": group["missing"],
            }
        )
    return sorted(rows, key=lambda row: _sort_key(row["edge_type"]))


def _edge_type(edge: KnowledgeEdge | Mapping[str, Any]) -> str:
    return _field_value(_first_value(edge, ("relation", "relation_type", "type"))) or "Unknown"


def _weight(edge: KnowledgeEdge | Mapping[str, Any]) -> float | None:
    if isinstance(edge, Mapping) and "weight" not in edge:
        return None
    value = _get(edge, "weight", None)
    if value is None or _field_value(value) == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


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
