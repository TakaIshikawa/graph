"""Summarize store unit reference resolution coverage."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

_REFERENCE_KEYS = ("references", "citations", "links")
_TARGET_KEYS = ("target", "target_id", "unit_id", "id", "url", "href", "doi")
_URL_KEYS = ("url", "href")
_STATUS_KEYS = ("status", "status_code", "http_status")
_UNIT_ID_KEYS = ("id", "unit_id")


def summarize_unit_reference_resolution(units: Iterable[Any]) -> dict[str, Any]:
    """Aggregate reference resolution and broken URL counts per unit."""

    rows: list[dict[str, Any]] = []
    total_units = reference_count = resolved_count = unresolved_count = broken_url_count = 0

    for unit in units:
        total_units += 1
        unit_reference_count = unit_resolved_count = unit_unresolved_count = unit_broken_url_count = 0
        for reference in _references(unit):
            unit_reference_count += 1
            if _is_broken_url(reference):
                unit_broken_url_count += 1
            if _is_unresolved(reference):
                unit_unresolved_count += 1
            else:
                unit_resolved_count += 1

        reference_count += unit_reference_count
        resolved_count += unit_resolved_count
        unresolved_count += unit_unresolved_count
        broken_url_count += unit_broken_url_count
        rows.append(
            {
                "unit_id": _unit_id(unit),
                "reference_count": unit_reference_count,
                "resolved_reference_count": unit_resolved_count,
                "unresolved_reference_count": unit_unresolved_count,
                "broken_url_count": unit_broken_url_count,
                "resolution_ratio": round(unit_resolved_count / unit_reference_count, 2)
                if unit_reference_count
                else 0.0,
            }
        )

    return {
        "total_units": total_units,
        "reference_count": reference_count,
        "resolved_reference_count": resolved_count,
        "unresolved_reference_count": unresolved_count,
        "broken_url_count": broken_url_count,
        "units": sorted(rows, key=lambda item: item["unit_id"]),
    }


def _references(unit: Any) -> list[Any]:
    metadata = _metadata(unit)
    for key in _REFERENCE_KEYS:
        value = _get(unit, key)
        if isinstance(value, list):
            return [item for item in value if item not in (None, "")]
        value = metadata.get(key)
        if isinstance(value, list):
            return [item for item in value if item not in (None, "")]
    return []


def _is_unresolved(reference: Any) -> bool:
    if isinstance(reference, str):
        return not bool(reference.strip())
    if not isinstance(reference, Mapping):
        return True
    resolved = reference.get("resolved")
    if isinstance(resolved, bool) and not resolved:
        return True
    if _status_code(reference) is not None and _status_code(reference) >= 400:
        return True
    return _target(reference) is None


def _is_broken_url(reference: Any) -> bool:
    if not isinstance(reference, Mapping):
        return False
    status_code = _status_code(reference)
    return bool(_url(reference)) and status_code is not None and status_code >= 400


def _target(reference: Mapping[str, Any]) -> Any:
    for key in _TARGET_KEYS:
        value = reference.get(key)
        if value not in (None, ""):
            return value
    return None


def _url(reference: Mapping[str, Any]) -> str | None:
    for key in _URL_KEYS:
        value = reference.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _status_code(reference: Mapping[str, Any]) -> int | None:
    value = _first(reference, _STATUS_KEYS)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"resolved", "ok", "success"}:
            return 200
        if normalized in {"unresolved", "broken", "missing", "failed", "error"}:
            return 500
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _metadata(item: Any) -> Mapping[str, Any]:
    value = _get(item, "metadata")
    return value if isinstance(value, Mapping) else {}


def _unit_id(item: Any) -> str:
    for key in _UNIT_ID_KEYS:
        value = _get(item, key)
        if value not in (None, ""):
            return str(value)
    return ""


def _get(item: Any, key: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(key)
    return getattr(item, key, None)


def _first(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None
