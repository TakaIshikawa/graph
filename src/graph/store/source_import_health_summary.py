"""Import health summary by source."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any


def source_import_health_summary(units: Iterable[Mapping[str, Any] | object]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any] | object]] = defaultdict(list)
    for unit in units:
        grouped[_text(_get(unit, "source_project")) or "unknown"].append(unit)

    rows: list[dict[str, Any]] = []
    for source in sorted(grouped, key=_sort_key):
        source_units = grouped[source]
        source_ids = [_text(_get(unit, "source_id")) for unit in source_units if _text(_get(unit, "source_id"))]
        id_counts = Counter(source_ids)
        latest = [_parse_datetime(_get(unit, "ingested_at")) for unit in source_units]
        latest = [value for value in latest if value is not None]
        rows.append(
            {
                "source_project": source,
                "unit_count": len(source_units),
                "latest_ingested_at": max(latest).isoformat() if latest else "",
                "missing_source_id_count": sum(1 for unit in source_units if not _text(_get(unit, "source_id"))),
                "duplicate_source_id_count": sum(count for count in id_counts.values() if count > 1),
                "missing_content_count": sum(1 for unit in source_units if not _text(_get(unit, "content"))),
                "error_flag_count": sum(1 for unit in source_units if _has_error_flag(unit)),
            }
        )
    return rows


def _has_error_flag(unit: Mapping[str, Any] | object) -> bool:
    metadata = _metadata(unit)
    for container in (unit, metadata):
        for key in ("error", "import_error", "failed"):
            value = _get(container, key) if container is unit else container.get(key)
            if _truthy(value):
                return True
        status = _get(container, "status") if container is unit else container.get("status")
        if _text(status).casefold() == "error":
            return True
    return False


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).casefold() in {"1", "true", "yes", "y", "failed", "error"}


def _metadata(unit: Mapping[str, Any] | object) -> Mapping[str, Any]:
    value = _get(unit, "metadata")
    return value if isinstance(value, Mapping) else {}


def _parse_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = _text(value)
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _get(value: Mapping[str, Any] | object, key: str) -> object:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _text(value: object) -> str:
    return "" if value is None else str(getattr(value, "value", value)).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _text(value)
    return (text.casefold(), text)
