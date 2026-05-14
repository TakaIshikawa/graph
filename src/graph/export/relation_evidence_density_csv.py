"""CSV export for aggregate relation evidence density."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeEdge

_FIELDNAMES = [
    "relation",
    "edge_count",
    "edges_with_source",
    "average_confidence",
    "average_weight",
    "average_metadata_keys",
    "evidence_density_score",
]
_WHITESPACE_RE = re.compile(r"\s+")


def export_relation_evidence_density_csv(
    edges: Iterable[KnowledgeEdge],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write aggregate relation evidence density as deterministic CSV."""
    edge_list = list(edges)
    rows = _density_rows(edge_list)
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
        "bytes_written": output_path.stat().st_size,
    }


def _density_rows(edges: list[KnowledgeEdge]) -> list[dict[str, str | int]]:
    groups: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "edge_count": 0,
            "edges_with_source": 0,
            "confidences": [],
            "weights": [],
            "metadata_key_counts": [],
            "density_signals": 0,
        }
    )

    for edge in edges:
        relation = _field_value(edge.relation) or "Unknown"
        group = groups[relation]
        group["edge_count"] += 1

        if _edge_source(edge):
            group["edges_with_source"] += 1
            group["density_signals"] += 1

        confidence = _edge_confidence(edge)
        if confidence is not None:
            group["confidences"].append(confidence)
            group["density_signals"] += 1

        weight = _number(getattr(edge, "weight", None))
        if weight is not None:
            group["weights"].append(weight)
            group["density_signals"] += 1

        metadata_key_count = _metadata_key_count(edge)
        group["metadata_key_counts"].append(metadata_key_count)
        if metadata_key_count > 0:
            group["density_signals"] += 1

    rows: list[dict[str, str | int]] = []
    for relation in sorted(groups, key=_sort_key):
        group = groups[relation]
        edge_count = group["edge_count"]
        rows.append(
            {
                "relation": relation,
                "edge_count": edge_count,
                "edges_with_source": group["edges_with_source"],
                "average_confidence": _average(group["confidences"]),
                "average_weight": _average(group["weights"]),
                "average_metadata_keys": _decimal(sum(group["metadata_key_counts"]) / edge_count),
                "evidence_density_score": _decimal(group["density_signals"] / (edge_count * 4)),
            }
        )
    return rows


def _edge_source(edge: KnowledgeEdge) -> str:
    return _field_value(getattr(edge, "source", None))


def _edge_confidence(edge: KnowledgeEdge) -> float | None:
    value = _number(getattr(edge, "confidence", None))
    if value is not None:
        return value
    metadata = edge.metadata if isinstance(edge.metadata, Mapping) else {}
    return _number(metadata.get("confidence"))


def _metadata_key_count(edge: KnowledgeEdge) -> int:
    metadata = edge.metadata if isinstance(edge.metadata, Mapping) else {}
    return len([key for key in metadata if _inline_text(key)])


def _number(value: object) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _average(values: list[float]) -> str:
    if not values:
        return ""
    return _decimal(sum(values) / len(values))


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
