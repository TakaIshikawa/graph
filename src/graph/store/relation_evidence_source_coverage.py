"""Report source coverage for relation evidence."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def relation_evidence_source_coverage(
    edges: Iterable[Any],
    *,
    units: Iterable[Any] | None = None,
) -> list[dict[str, Any]]:
    unit_sources = {_string(_get(unit, "id")): _string(_get(unit, "source_id")) for unit in units or []}
    rows = []
    for edge in edges:
        metadata = _metadata(edge)
        evidence = _evidence_items(metadata)
        sources = sorted(
            {
                source
                for item in evidence
                if (source := _evidence_source(item, unit_sources)) is not None
            }
        )
        count = len(evidence)
        distinct = len(sources)
        if count == 0 or distinct == 0:
            status = "none"
        elif distinct == 1:
            status = "single_source"
        else:
            status = "multi_source"
        rows.append(
            {
                "relation_id": _string(_get(edge, "id")),
                "relation": _string(_get(edge, "relation")),
                "evidence_count": count,
                "distinct_source_count": distinct,
                "sources": sources,
                "coverage_status": status,
            }
        )
    return sorted(rows, key=lambda row: row["relation_id"] or "")


def _evidence_items(metadata: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = metadata.get("evidence") or metadata.get("evidence_items") or []
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, Mapping)]
    source = metadata.get("evidence_source_id") or metadata.get("source_id")
    return [{"source_id": source}] if source else []


def _evidence_source(item: Mapping[str, Any], unit_sources: dict[str | None, str | None]) -> str | None:
    direct = item.get("source_id") or item.get("source")
    if isinstance(direct, Mapping):
        direct = direct.get("id") or direct.get("source_id")
    if direct not in (None, ""):
        return str(direct)
    unit_id = item.get("unit_id")
    return unit_sources.get(str(unit_id)) if unit_id not in (None, "") else None


def _metadata(item: Any) -> Mapping[str, Any]:
    value = _get(item, "metadata")
    return value if isinstance(value, Mapping) else {}


def _get(item: Any, key: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(key)
    return getattr(item, key, None)


def _string(value: Any) -> str | None:
    return None if value is None else str(value)
