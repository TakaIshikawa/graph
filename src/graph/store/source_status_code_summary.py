"""Summarize source status code metadata for knowledge units."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

DEFAULT_SAMPLE_LIMIT = 5
_STATUS_KEYS = ("source_status", "status_code", "http_status", "response_status")
_SOURCE_KEYS = ("source", "source_project")


def source_status_code_summary(
    units: Iterable[Any],
    *,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
) -> list[dict[str, Any]]:
    """Return stable status-class counts grouped by source.

    Units may be mappings or KnowledgeUnit-like objects. Status fields are read
    from the unit first, then from ``metadata``. Missing or unparsable statuses
    are grouped under ``unknown``.
    """

    if sample_limit < 0:
        raise ValueError("sample_limit must be non-negative")

    groups: dict[tuple[str, str], dict[str, Any]] = {}
    source_totals: dict[str, int] = defaultdict(int)
    source_error_totals: dict[str, int] = defaultdict(int)

    for unit in units:
        metadata = _metadata(unit)
        source = _source(unit, metadata)
        status = _status_value(unit, metadata)
        status_class = _status_class(status)
        status_code = _status_code(status)
        unit_id = _text(_get(unit, "id") or metadata.get("id"))
        key = (source, status_class)

        group = groups.setdefault(
            key,
            {
                "source": source,
                "status_class": status_class,
                "count": 0,
                "status_codes": set(),
                "sample_unit_ids": [],
            },
        )
        group["count"] += 1
        if status_code is not None:
            group["status_codes"].add(status_code)
        if unit_id and len(group["sample_unit_ids"]) < sample_limit:
            group["sample_unit_ids"].append(unit_id)

        source_totals[source] += 1
        if status_class in {"4xx", "5xx"}:
            source_error_totals[source] += 1

    rows: list[dict[str, Any]] = []
    for key in sorted(
        groups, key=lambda item: (_sort_key(item[0]), _status_sort_key(item[1]))
    ):
        group = groups[key]
        source = group["source"]
        rows.append(
            {
                "source": source,
                "status_class": group["status_class"],
                "count": group["count"],
                "status_codes": sorted(group["status_codes"]),
                "sample_unit_ids": group["sample_unit_ids"],
                "error_share": (
                    source_error_totals[source] / source_totals[source]
                    if source_totals[source]
                    else 0.0
                ),
            }
        )
    return rows


def _status_value(unit: Any, metadata: Mapping[str, Any]) -> Any:
    for key in _STATUS_KEYS:
        value = _get(unit, key)
        if value not in (None, ""):
            return value
        value = metadata.get(key)
        if value not in (None, ""):
            return value
    return None


def _status_code(value: Any) -> int | None:
    try:
        status = int(value)
    except (TypeError, ValueError):
        return None
    return status if status > 0 else None


def _status_class(value: Any) -> str:
    status = _status_code(value)
    if status is None:
        return "unknown"
    status_class = status // 100
    return f"{status_class}xx" if status_class in {2, 3, 4, 5} else "unknown"


def _source(unit: Any, metadata: Mapping[str, Any]) -> str:
    for key in _SOURCE_KEYS:
        value = _get(unit, key)
        if value not in (None, ""):
            return _text(value) or "unknown"
        value = metadata.get(key)
        if value not in (None, ""):
            return _text(value) or "unknown"
    return "unknown"


def _metadata(unit: Any) -> Mapping[str, Any]:
    value = _get(unit, "metadata")
    return value if isinstance(value, Mapping) else {}


def _get(item: Any, key: str) -> Any:
    return item.get(key) if isinstance(item, Mapping) else getattr(item, key, None)


def _text(value: Any) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def _sort_key(value: str) -> tuple[str, str]:
    return (value.casefold(), value)


def _status_sort_key(value: str) -> tuple[int, str]:
    order = {"2xx": 0, "3xx": 1, "4xx": 2, "5xx": 3, "unknown": 4}
    return (order.get(value, 5), value)
