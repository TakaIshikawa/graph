"""Summarize common unit metadata completeness by source and entity type."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

_FIELDS = ("title", "created_at", "updated_at", "url", "tags", "language", "attachments")


def summarize_unit_metadata_completeness(units: Iterable[Any]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[Any]] = defaultdict(list)
    total_units = 0
    for unit in units:
        total_units += 1
        grouped[(_source(unit), _entity_type(unit))].append(unit)
    rows = [_row(source, entity_type, grouped[(source, entity_type)]) for source, entity_type in sorted(grouped, key=lambda key: (_sort_key(key[0]), _sort_key(key[1])))]
    return {"total_units": total_units, "fields": list(_FIELDS), "rows": rows}


def _row(source: str, entity_type: str, units: list[Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"source": source, "entity_type": entity_type, "unit_count": len(units)}
    for field in _FIELDS:
        present = sum(1 for unit in units if _present(_value(unit, field)))
        row[f"{field}_present_count"] = present
        row[f"{field}_coverage_ratio"] = f"{(present / len(units)):.2f}" if units else "0.00"
    return row


def _value(unit: Any, key: str) -> Any:
    value = _get(unit, key)
    return value if value not in (None, "") else _metadata(unit).get(key)


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, list | tuple | set):
        return bool(value)
    return True


def _source(unit: Any) -> str:
    meta = _metadata(unit)
    return _text(_get(unit, "source_project") or meta.get("source") or meta.get("source_project")) or "unknown"


def _entity_type(unit: Any) -> str:
    meta = _metadata(unit)
    return _text(_get(unit, "source_entity_type") or _get(unit, "entity_type") or meta.get("entity_type")) or "unknown"


def _metadata(unit: Any) -> Mapping[str, Any]:
    value = _get(unit, "metadata")
    return value if isinstance(value, Mapping) else {}


def _get(item: Any, key: str) -> Any:
    return item.get(key) if isinstance(item, Mapping) else getattr(item, key, None)


def _text(value: Any) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def _sort_key(value: Any) -> tuple[str, str]:
    text = _text(value)
    return (text.casefold(), text)
