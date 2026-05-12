"""CSV export for relation evidence gaps."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeEdge

_FIELDNAMES = [
    "edge_id",
    "relation",
    "from_unit_id",
    "to_unit_id",
    "source_count",
    "confidence_summary",
    "gap_reason",
]
_SOURCE_METADATA_KEYS = (
    "source_id",
    "source_url",
    "source_uri",
    "source_project",
    "source_entity_type",
    "citation",
    "citations",
)
_SOURCE_ITEM_KEYS = ("id", "source_id", "url", "uri", "source_project", "citation")
_WHITESPACE_RE = re.compile(r"\s+")


def export_relation_evidence_gap_csv(
    edges: Iterable[KnowledgeEdge],
    path: str | Path | None = None,
    *,
    low_threshold: float = 0.5,
) -> str | dict[str, Any]:
    """Return or write edge rows with missing, sparse, or low-confidence evidence."""
    _validate_low_threshold(low_threshold)

    edge_list = list(edges)
    rows = _gap_rows(edge_list, low_threshold=low_threshold)
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
        "low_threshold": low_threshold,
        "bytes_written": output_path.stat().st_size,
    }


def _gap_rows(
    edges: list[KnowledgeEdge],
    *,
    low_threshold: float,
) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for edge in edges:
        source_count = _source_count(edge)
        confidence = _edge_confidence(edge)
        gap_reason = _gap_reason(
            edge,
            source_count=source_count,
            confidence=confidence,
            low_threshold=low_threshold,
        )
        if gap_reason is None:
            continue
        rows.append(
            {
                "edge_id": _field_value(edge.id),
                "relation": _field_value(edge.relation) or "Unknown",
                "from_unit_id": _field_value(edge.from_unit_id),
                "to_unit_id": _field_value(edge.to_unit_id),
                "source_count": source_count,
                "confidence_summary": _decimal(confidence) if confidence is not None else "",
                "gap_reason": gap_reason,
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            _sort_key(row["gap_reason"]),
            _sort_key(row["relation"]),
            _sort_key(row["edge_id"]),
            _sort_key(row["from_unit_id"]),
            _sort_key(row["to_unit_id"]),
        ),
    )


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _gap_reason(
    edge: KnowledgeEdge,
    *,
    source_count: int,
    confidence: float | None,
    low_threshold: float,
) -> str | None:
    if source_count == 0:
        return "no_sources"
    if confidence is not None and confidence < low_threshold:
        return "low_confidence"
    if not _has_source_metadata(edge):
        return "missing_source_metadata"
    return None


def _source_count(edge: KnowledgeEdge) -> int:
    metadata = _metadata(edge)
    sources = metadata.get("sources")
    if isinstance(sources, list):
        return len([source for source in sources if _has_source_item(source)])
    if isinstance(sources, tuple):
        return len([source for source in sources if _has_source_item(source)])
    source = _field_value(getattr(edge, "source", None))
    return 1 if source else 0


def _has_source_metadata(edge: KnowledgeEdge) -> bool:
    metadata = _metadata(edge)
    if any(_field_value(metadata.get(key)) for key in _SOURCE_METADATA_KEYS):
        return True
    sources = metadata.get("sources")
    if isinstance(sources, list | tuple):
        return any(_has_source_item(source) for source in sources)
    return False


def _has_source_item(source: object) -> bool:
    if isinstance(source, Mapping):
        return any(_field_value(source.get(key)) for key in _SOURCE_ITEM_KEYS)
    return bool(_field_value(source))


def _edge_confidence(edge: KnowledgeEdge) -> float | None:
    value = getattr(edge, "confidence", None)
    if value is not None:
        return _confidence_value(value)
    metadata = _metadata(edge)
    return _confidence_value(metadata.get("confidence"))


def _metadata(edge: KnowledgeEdge) -> Mapping[str, Any]:
    return edge.metadata if isinstance(edge.metadata, Mapping) else {}


def _validate_low_threshold(low_threshold: float) -> None:
    if not _is_number(low_threshold) or not 0 <= low_threshold <= 1:
        raise ValueError("low_threshold must be a number between 0 and 1")


def _confidence_value(value: object) -> float | None:
    if not _is_number(value):
        return None
    return float(value)


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


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
