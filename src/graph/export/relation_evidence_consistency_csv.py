"""CSV export for relation evidence consistency checks."""

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
    "relation_type",
    "source_bucket",
    "edge_count",
    "evidence_count",
    "missing_evidence_count",
    "conflicting_source_count",
    "conflicting_date_count",
    "unit_pairs",
    "sample_edge_ids",
]
_UNKNOWN = "Unknown"
_SOURCE_KEYS = ("source", "source_id", "source_url", "url", "uri", "citation")
_DATE_KEYS = ("date", "evidence_date", "source_date", "observed_date", "published_date")
_TYPE_KEYS = ("relation_type", "type", "edge_type")
_SOURCE_BUCKET_KEYS = ("source_project", "source_id", "source_url", "source")
_WHITESPACE_RE = re.compile(r"\s+")


def export_relation_evidence_consistency_csv(
    edges: Iterable[KnowledgeEdge | Mapping[str, Any]],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write relation groups with inconsistent evidence metadata."""
    edge_list = list(edges)
    rows = _consistency_rows(edge_list)
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


def _consistency_rows(edges: list[KnowledgeEdge | Mapping[str, Any]]) -> list[dict[str, str | int]]:
    groups: dict[tuple[str, str, str], list[KnowledgeEdge | Mapping[str, Any]]] = defaultdict(list)
    for edge in edges:
        groups[(_relation(edge), _relation_type(edge), _source_bucket(edge))].append(edge)

    rows: list[dict[str, str | int]] = []
    for (relation, relation_type, source_bucket), group_edges in groups.items():
        evidence_values: list[Mapping[str, Any]] = []
        missing_evidence_count = 0
        for edge in group_edges:
            evidence = _edge_evidence(edge)
            if evidence:
                evidence_values.extend(evidence)
            else:
                missing_evidence_count += 1

        source_values = {_evidence_source(evidence) for evidence in evidence_values if _evidence_source(evidence)}
        date_values = {_evidence_date(evidence) for evidence in evidence_values if _evidence_date(evidence)}
        rows.append(
            {
                "relation": relation,
                "relation_type": relation_type,
                "source_bucket": source_bucket,
                "edge_count": len(group_edges),
                "evidence_count": len(evidence_values),
                "missing_evidence_count": missing_evidence_count,
                "conflicting_source_count": len(source_values) if len(source_values) > 1 else 0,
                "conflicting_date_count": len(date_values) if len(date_values) > 1 else 0,
                "unit_pairs": _joined_unique(
                    f"{_field_value(_get(edge, 'from_unit_id'))}->{_field_value(_get(edge, 'to_unit_id'))}"
                    for edge in group_edges
                    if _field_value(_get(edge, "from_unit_id")) or _field_value(_get(edge, "to_unit_id"))
                ),
                "sample_edge_ids": _joined_unique(_field_value(_get(edge, "id")) for edge in group_edges),
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            _sort_key(row["relation"]),
            _sort_key(row["relation_type"]),
            _sort_key(row["source_bucket"]),
            _sort_key(row["unit_pairs"]),
        ),
    )


def _relation(edge: KnowledgeEdge | Mapping[str, Any]) -> str:
    return _field_value(_get(edge, "relation")) or _UNKNOWN


def _relation_type(edge: KnowledgeEdge | Mapping[str, Any]) -> str:
    metadata = _metadata(edge)
    for key in _TYPE_KEYS:
        value = _field_value(metadata.get(key))
        if value:
            return value
    return _UNKNOWN


def _source_bucket(edge: KnowledgeEdge | Mapping[str, Any]) -> str:
    metadata = _metadata(edge)
    for key in _SOURCE_BUCKET_KEYS:
        value = _field_value(metadata.get(key))
        if value:
            return value
    return _field_value(_get(edge, "source")) or _UNKNOWN


def _edge_evidence(edge: KnowledgeEdge | Mapping[str, Any]) -> list[Mapping[str, Any]]:
    metadata = _metadata(edge)
    evidence = metadata.get("evidence", metadata.get("evidences", metadata.get("sources")))
    if isinstance(evidence, Mapping):
        return [evidence]
    if isinstance(evidence, list | tuple):
        return [item for item in evidence if isinstance(item, Mapping) and _has_evidence_value(item)]
    if any(_field_value(metadata.get(key)) for key in (*_SOURCE_KEYS, *_DATE_KEYS)):
        return [metadata]
    return []


def _has_evidence_value(value: Mapping[str, Any]) -> bool:
    return any(_field_value(value.get(key)) for key in (*_SOURCE_KEYS, *_DATE_KEYS))


def _evidence_source(evidence: Mapping[str, Any]) -> str:
    for key in _SOURCE_KEYS:
        value = _field_value(evidence.get(key))
        if value:
            return value
    return ""


def _evidence_date(evidence: Mapping[str, Any]) -> str:
    for key in _DATE_KEYS:
        value = _field_value(evidence.get(key))
        if value:
            return value
    return ""


def _metadata(edge: KnowledgeEdge | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _get(edge, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _joined_unique(values: Iterable[object]) -> str:
    return "; ".join(sorted({_field_value(value) for value in values if _field_value(value)}, key=_sort_key))


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
