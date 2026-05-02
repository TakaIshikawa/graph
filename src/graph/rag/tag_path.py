"""Plan deterministic reading paths from unit tag continuity."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from graph.rag.reading_queue import UNREAD_STATUSES

_MISSING = object()


@dataclass(frozen=True)
class _PathUnit:
    unit_id: str
    title: str
    tags: tuple[str, ...]
    tag_keys: frozenset[str]
    unread: bool


def _validate_limit(limit: int | None) -> int | None:
    if limit is None:
        return None
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        raise ValueError("max_units must be a non-negative integer")
    return limit


def _value(item: Any, key: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, _MISSING)
    return getattr(item, key, _MISSING)


def _result_value(result: Any, key: str) -> Any:
    value = _value(result, key)
    if value is not _MISSING and value is not None:
        return value

    unit = _value(result, "unit")
    if unit is _MISSING or unit is None:
        return value
    nested_value = _value(unit, key)
    if nested_value is not _MISSING:
        return nested_value
    return value


def _metadata_value(result: Any, key: str) -> Any:
    metadata = _result_value(result, "metadata")
    if isinstance(metadata, Mapping):
        return metadata.get(key, _MISSING)
    return _MISSING


def _text_value(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    text = " ".join(str(value).strip().split())
    return text or None


def _tag_key(tag: str) -> str:
    return tag.casefold()


def _tag_sort_key(tag: str) -> tuple[str, str]:
    return (tag.casefold(), tag)


def _tag_values(value: Any) -> list[str]:
    if value is _MISSING or value is None:
        return []
    if isinstance(value, str):
        if "," in value:
            return [_tag for part in value.split(",") if (_tag := _text_value(part))]
        tag = _text_value(value)
        return [] if tag is None else [tag]
    if isinstance(value, Mapping):
        for key in ("tag", "name", "label", "value"):
            tag = _text_value(value.get(key))
            if tag is not None:
                return [tag]
        return []
    if isinstance(value, Iterable) and not isinstance(value, bytes):
        tags: list[str] = []
        for item in value:
            tags.extend(_tag_values(item))
        return tags

    tag = _text_value(value)
    return [] if tag is None else [tag]


def _tags_for_unit(unit: Any) -> tuple[str, ...]:
    tags_by_key: dict[str, str] = {}
    raw_values = [_result_value(unit, "tags")]
    if not _tag_values(raw_values[0]):
        raw_values = [_metadata_value(unit, "tags"), _metadata_value(unit, "tag")]

    for raw_value in raw_values:
        for tag in _tag_values(raw_value):
            tags_by_key.setdefault(_tag_key(tag), tag)
    return tuple(sorted(tags_by_key.values(), key=_tag_sort_key))


def _read_status(unit: Any) -> str | None:
    value = _metadata_value(unit, "read_status")
    if value is _MISSING:
        value = _result_value(unit, "read_status")
    text = _text_value(value)
    if text is None:
        return None
    return text.casefold().replace(" ", "_")


def _is_unread(unit: Any) -> bool:
    status = _read_status(unit)
    return status is None or status in UNREAD_STATUSES


def _path_unit(unit: Any) -> _PathUnit:
    unit_id = _text_value(_result_value(unit, "id")) or _text_value(
        _result_value(unit, "unit_id")
    )
    title = _text_value(_result_value(unit, "title"))
    tags = _tags_for_unit(unit)
    return _PathUnit(
        unit_id=unit_id or "",
        title=title or unit_id or "",
        tags=tags,
        tag_keys=frozenset(_tag_key(tag) for tag in tags),
        unread=_is_unread(unit),
    )


def _matched_tags(unit: _PathUnit, context_keys: set[str]) -> list[str]:
    return [tag for tag in unit.tags if _tag_key(tag) in context_keys]


def _reason(
    *,
    matched: list[str],
    context_keys: set[str],
    previous_unit_id: str | None,
    unread: bool,
    prefer_unread: bool,
) -> str:
    if matched and previous_unit_id is None:
        return "start_tag_match"
    if matched:
        return "tag_continuity"
    if not context_keys:
        return "initial_fallback"
    if prefer_unread and unread:
        return "unread_fallback"
    return "fallback"


def _payload(
    unit: _PathUnit,
    *,
    matched: list[str],
    reason: str,
    previous_unit_id: str | None,
) -> dict[str, Any]:
    return {
        "id": unit.unit_id,
        "unit_id": unit.unit_id,
        "title": unit.title,
        "tags": list(unit.tags),
        "matched_tags": matched,
        "transition_reason": reason,
        "previous_unit_id": previous_unit_id,
    }


def plan_tag_reading_path(
    units: Iterable[Any],
    *,
    start_tags: Iterable[Any] | None = None,
    max_units: int | None = None,
    prefer_unread: bool = False,
) -> dict[str, Any]:
    """Build a deterministic reading path that follows shared tags.

    The planner greedily selects the next unit with the strongest tag overlap
    with the current context. If no candidate shares tags with that context,
    it falls back to unread status when requested, then title and unit id.
    """
    max_units_value = _validate_limit(max_units)
    start_tag_labels = _tag_values(list(start_tags or []))
    start_tag_keys = {_tag_key(tag) for tag in start_tag_labels}
    candidates = sorted(
        (_path_unit(unit) for unit in units),
        key=lambda unit: (unit.title.casefold(), unit.unit_id),
    )

    remaining = list(candidates)
    context_keys = set(start_tag_keys)
    previous_unit_id: str | None = None
    planned: list[dict[str, Any]] = []

    while remaining:
        if max_units_value is not None and len(planned) >= max_units_value:
            break

        def sort_key(unit: _PathUnit) -> tuple[int, int, int, int, str, str]:
            overlap = len(unit.tag_keys & context_keys)
            start_overlap = len(unit.tag_keys & start_tag_keys)
            return (
                -overlap,
                -start_overlap,
                0 if prefer_unread and unit.unread else 1,
                0 if unit.tag_keys else 1,
                unit.title.casefold(),
                unit.unit_id,
            )

        next_unit = min(remaining, key=sort_key)
        remaining.remove(next_unit)

        matched = _matched_tags(next_unit, context_keys)
        reason = _reason(
            matched=matched,
            context_keys=context_keys,
            previous_unit_id=previous_unit_id,
            unread=next_unit.unread,
            prefer_unread=prefer_unread,
        )
        planned.append(
            _payload(
                next_unit,
                matched=matched,
                reason=reason,
                previous_unit_id=previous_unit_id,
            )
        )
        context_keys.update(next_unit.tag_keys)
        previous_unit_id = next_unit.unit_id

    return {
        "units": planned,
        "stats": {
            "total_units": len(candidates),
            "candidate_units": len(candidates),
            "planned_units": len(planned),
            "omitted_units": len(candidates) - len(planned),
            "start_tags": sorted(set(start_tag_labels), key=_tag_sort_key),
            "start_tag_keys": sorted(start_tag_keys),
            "max_units": max_units_value,
            "prefer_unread": prefer_unread,
        },
    }
