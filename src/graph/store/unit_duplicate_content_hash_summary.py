"""Summarize duplicate normalized unit content hashes."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any


def summarize_unit_duplicate_content_hashes(units: Iterable[Any]) -> dict[str, Any]:
    groups: dict[str, list[Any]] = defaultdict(list)
    total_units = 0
    for unit in units:
        total_units += 1
        content = _normalize(_get(unit, "content"))
        if not content:
            continue
        groups[hashlib.sha256(content.encode("utf-8")).hexdigest()].append(unit)
    duplicate_groups = []
    for content_hash, members in groups.items():
        if len(members) <= 1:
            continue
        ordered = sorted(members, key=lambda unit: _sort_key(_unit_id(unit)))
        duplicate_groups.append({"content_hash": content_hash, "unit_ids": [_unit_id(unit) for unit in ordered], "count": len(ordered), "title_samples": [_title(unit) for unit in ordered[:3] if _title(unit)]})
    duplicate_groups.sort(key=lambda row: (-row["count"], row["unit_ids"]))
    return {"total_units": total_units, "duplicate_group_count": len(duplicate_groups), "duplicate_groups": duplicate_groups}


def _get(item: Any, key: str) -> Any:
    return item.get(key) if isinstance(item, Mapping) else getattr(item, key, None)


def _text(value: Any) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def _normalize(value: Any) -> str:
    return _text(value).casefold()


def _unit_id(unit: Any) -> str:
    return _text(_get(unit, "id") or _get(unit, "unit_id"))


def _title(unit: Any) -> str:
    return _text(_get(unit, "title"))


def _sort_key(value: Any) -> tuple[str, str]:
    text = _text(value)
    return (text.casefold(), text)
