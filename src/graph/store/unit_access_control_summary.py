"""Summarize unit access-control signals by source."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

_ACCESS_KEYS = ("visibility", "access", "privacy", "sharing")
_SENSITIVITY_KEYS = ("sensitivity", "sensitive", "contains_pii")
_PUBLIC_VALUES = {"public", "open", "anyone", "anyone_with_link", "shared_public", "published"}
_PRIVATE_VALUES = {"private", "personal", "restricted_private", "only_me", "owner_only"}
_RESTRICTED_VALUES = {"restricted", "limited", "internal", "confidential", "team", "domain", "invite_only"}
_SENSITIVE_VALUES = {"sensitive", "high", "pii", "contains_pii", "confidential", "restricted", "true", "yes", "1"}


def summarize_unit_access_control(units: Iterable[Any]) -> dict[str, Any]:
    """Group units by source and count privacy, access, and sensitivity metadata."""

    grouped: dict[str, list[Any]] = defaultdict(list)
    total_units = 0
    for unit in units:
        total_units += 1
        grouped[_source(unit)].append(unit)

    rows = [_row(source, grouped[source]) for source in sorted(grouped, key=_sort_key)]
    return {"total_units": total_units, "rows": rows, "source_summaries": rows}


def _row(source: str, units: list[Any]) -> dict[str, Any]:
    buckets = [_access_bucket(unit) for unit in units]
    return {
        "source": source,
        "unit_count": len(units),
        "public_count": buckets.count("public"),
        "private_count": buckets.count("private"),
        "restricted_count": buckets.count("restricted"),
        "missing_access_count": buckets.count("missing"),
        "sensitive_count": sum(1 for unit in units if _is_sensitive(unit)),
    }


def _access_bucket(unit: Any) -> str:
    value = _first(_metadata(unit), _ACCESS_KEYS)
    if value in (None, ""):
        return "missing"
    normalized = _normalize(value)
    if normalized in _PUBLIC_VALUES or normalized.startswith("public"):
        return "public"
    if normalized in _PRIVATE_VALUES or normalized.startswith("private"):
        return "private"
    if normalized in _RESTRICTED_VALUES:
        return "restricted"
    return "restricted"


def _is_sensitive(unit: Any) -> bool:
    metadata = _metadata(unit)
    for key in _SENSITIVITY_KEYS:
        value = metadata.get(key)
        if isinstance(value, bool):
            if value:
                return True
            continue
        if value in (None, ""):
            continue
        if _normalize(value) in _SENSITIVE_VALUES:
            return True
    return False


def _source(unit: Any) -> str:
    meta = _metadata(unit)
    return _text(_get(unit, "source_project") or meta.get("source") or meta.get("source_project")) or "unknown"


def _metadata(unit: Any) -> Mapping[str, Any]:
    value = _get(unit, "metadata")
    return value if isinstance(value, Mapping) else {}


def _get(item: Any, key: str) -> Any:
    return item.get(key) if isinstance(item, Mapping) else getattr(item, key, None)


def _first(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def _text(value: Any) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def _normalize(value: Any) -> str:
    return _text(value).replace("-", "_").casefold()


def _sort_key(value: Any) -> tuple[str, str]:
    text = _text(value)
    return (text.casefold(), text)
