"""Summarize metadata key namespaces for knowledge units."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

DEFAULT_SAMPLE_LIMIT = 5
_DELIMITERS = (".", ":", "_")
UNSCOPED_NAMESPACE = "unscoped"


def unit_metadata_namespace_summary(
    units: Iterable[Any],
    *,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
) -> list[dict[str, Any]]:
    """Return namespace usage rows for unit metadata keys."""

    if sample_limit < 0:
        raise ValueError("sample_limit must be non-negative")

    total_units = 0
    groups: dict[str, dict[str, Any]] = {}
    for unit in units:
        total_units += 1
        metadata = _metadata(unit)
        unit_id = _text(_get(unit, "id") or metadata.get("id"))
        unit_namespaces: set[str] = set()

        for key in metadata:
            if not isinstance(key, str) or not key:
                continue
            namespace = _namespace(key)
            group = groups.setdefault(
                namespace,
                {
                    "namespace": namespace,
                    "keys": set(),
                    "unit_ids": set(),
                    "sample_unit_ids": [],
                },
            )
            group["keys"].add(key)
            unit_namespaces.add(namespace)

        for namespace in unit_namespaces:
            group = groups[namespace]
            group["unit_ids"].add(unit_id)
            if unit_id and len(group["sample_unit_ids"]) < sample_limit:
                group["sample_unit_ids"].append(unit_id)

    rows: list[dict[str, Any]] = []
    for namespace in sorted(groups, key=_sort_key):
        group = groups[namespace]
        unit_count = len(group["unit_ids"])
        rows.append(
            {
                "namespace": namespace,
                "key_count": len(group["keys"]),
                "unit_count": unit_count,
                "keys": sorted(group["keys"], key=_sort_key),
                "sample_unit_ids": group["sample_unit_ids"],
                "coverage_share": unit_count / total_units if total_units else 0.0,
            }
        )
    return rows


def _namespace(key: str) -> str:
    positions = [key.find(delimiter) for delimiter in _DELIMITERS if delimiter in key]
    if not positions:
        return UNSCOPED_NAMESPACE
    prefix = key[: min(positions)].strip()
    return prefix or UNSCOPED_NAMESPACE


def _metadata(unit: Any) -> Mapping[str, Any]:
    value = _get(unit, "metadata")
    return value if isinstance(value, Mapping) else {}


def _get(item: Any, key: str) -> Any:
    return item.get(key) if isinstance(item, Mapping) else getattr(item, key, None)


def _text(value: Any) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def _sort_key(value: str) -> tuple[str, str]:
    return (value.casefold(), value)
