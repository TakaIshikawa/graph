"""Timestamp consistency summary for store units."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

CREATED_KEYS = ("created_at", "created", "created_time", "date_created")
UPDATED_KEYS = ("updated_at", "updated", "modified_at", "last_modified")
IMPORTED_KEYS = ("imported_at", "ingested_at", "imported", "ingest_time")
ISSUES = (
    "missing_created",
    "missing_updated",
    "updated_before_created",
    "imported_before_created",
    "invalid_timestamp",
)


def summarize_unit_timestamp_consistency(
    units: Iterable[Any], *, sample_limit: int = 5
) -> dict[str, Any]:
    """Return counts and example ids for suspicious unit timestamp ordering."""

    issue_examples = {issue: [] for issue in ISSUES}
    issue_counts = {issue: 0 for issue in ISSUES}
    total_units = 0

    for index, unit in enumerate(units):
        total_units += 1
        unit_id = _unit_id(unit, index)
        created_raw = _first_value(unit, CREATED_KEYS)
        updated_raw = _first_value(unit, UPDATED_KEYS)
        imported_raw = _first_value(unit, IMPORTED_KEYS)

        created = _parse_timestamp(created_raw)
        updated = _parse_timestamp(updated_raw)
        imported = _parse_timestamp(imported_raw)

        invalid = any(
            raw not in (None, "") and parsed is None
            for raw, parsed in ((created_raw, created), (updated_raw, updated), (imported_raw, imported))
        )
        if invalid:
            _add_issue("invalid_timestamp", unit_id, issue_counts, issue_examples, sample_limit)
        if created_raw in (None, ""):
            _add_issue("missing_created", unit_id, issue_counts, issue_examples, sample_limit)
        if updated_raw in (None, ""):
            _add_issue("missing_updated", unit_id, issue_counts, issue_examples, sample_limit)
        if created is not None and updated is not None and updated < created:
            _add_issue("updated_before_created", unit_id, issue_counts, issue_examples, sample_limit)
        if created is not None and imported is not None and imported < created:
            _add_issue("imported_before_created", unit_id, issue_counts, issue_examples, sample_limit)

    return {
        "total_units": total_units,
        "issue_counts": issue_counts,
        "example_unit_ids": issue_examples,
        **{f"{issue}_count": issue_counts[issue] for issue in ISSUES},
    }


def _add_issue(issue: str, unit_id: str, counts: dict[str, int], examples: dict[str, list[str]], limit: int) -> None:
    counts[issue] += 1
    if len(examples[issue]) < limit:
        examples[issue].append(unit_id)


def _parse_timestamp(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _first_value(unit: Any, keys: tuple[str, ...]) -> Any:
    meta = _metadata(unit)
    for key in keys:
        value = _get(unit, key)
        if value not in (None, ""):
            return value
        value = meta.get(key)
        if value not in (None, ""):
            return value
    return None


def _metadata(unit: Any) -> Mapping[str, Any]:
    value = _get(unit, "metadata")
    return value if isinstance(value, Mapping) else {}


def _unit_id(unit: Any, index: int) -> str:
    return _text(_get(unit, "id") or _metadata(unit).get("id")) or str(index)


def _get(item: Any, key: str) -> Any:
    return item.get(key) if isinstance(item, Mapping) else getattr(item, key, None)


def _text(value: Any) -> str:
    return " ".join(str(value).split()) if value is not None else ""
