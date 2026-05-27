"""Summarize units with empty or metadata-only content."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

_FRONTMATTER_RE = re.compile(r"\A\s*---\s*\n.*?\n---\s*\Z", re.DOTALL)


def summarize_unit_empty_content(units: Iterable[Any]) -> dict[str, Any]:
    total = 0
    empty: list[str] = []
    whitespace: list[str] = []
    metadata_only: list[str] = []
    non_empty = 0
    for unit in units:
        total += 1
        unit_id = _unit_id(unit)
        raw = _get(unit, "content")
        content = "" if raw is None else str(raw)
        if content == "":
            empty.append(unit_id)
        elif content.strip() == "":
            whitespace.append(unit_id)
        elif _FRONTMATTER_RE.match(content):
            metadata_only.append(unit_id)
        else:
            non_empty += 1
    return {
        "total_units": total,
        "empty_content_units": len(empty),
        "whitespace_only_units": len(whitespace),
        "metadata_only_units": len(metadata_only),
        "non_empty_units": non_empty,
        "examples": {
            "empty_content_unit_ids": sorted(empty, key=_sort_key)[:5],
            "whitespace_only_unit_ids": sorted(whitespace, key=_sort_key)[:5],
            "metadata_only_unit_ids": sorted(metadata_only, key=_sort_key)[:5],
        },
    }


def _get(item: Any, key: str) -> Any:
    return item.get(key) if isinstance(item, Mapping) else getattr(item, key, None)


def _unit_id(unit: Any) -> str:
    return _text(_get(unit, "id") or _get(unit, "unit_id"))


def _text(value: Any) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def _sort_key(value: Any) -> tuple[str, str]:
    text = _text(value)
    return (text.casefold(), text)
