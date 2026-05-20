"""CSV export for keyword indexes derived from unit titles."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["keyword", "unit_count", "sources", "tags", "unit_titles"]
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
_WHITESPACE_RE = re.compile(r"\s+")
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def export_unit_title_keyword_index_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | None = None,
    *,
    min_length: int = 3,
) -> str | dict[str, Any]:
    """Return or write a lightweight keyword index from unit titles."""
    if not isinstance(min_length, int) or isinstance(min_length, bool) or min_length < 1:
        raise ValueError("min_length must be a positive integer")

    unit_list = list(units)
    rows = _keyword_rows(unit_list, min_length=min_length)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "rows_exported": len(rows),
        "min_length": min_length,
        "bytes_written": output_path.stat().st_size,
    }


def _keyword_rows(
    units: list[KnowledgeUnit | Mapping[str, Any]],
    *,
    min_length: int,
) -> list[dict[str, str | int]]:
    index: dict[str, dict[str, KnowledgeUnit | Mapping[str, Any]]] = defaultdict(dict)
    for unit in units:
        unit_id = _unit_id(unit)
        for keyword in _title_keywords(_get(unit, "title"), min_length=min_length):
            if unit_id:
                index[keyword][unit_id] = unit

    rows: list[dict[str, str | int]] = []
    for keyword, units_by_id in index.items():
        indexed_units = list(units_by_id.values())
        rows.append(
            {
                "keyword": keyword,
                "unit_count": len(units_by_id),
                "sources": _joined_unique(_field_value(_get(unit, "source_project")) or "Unknown" for unit in indexed_units),
                "tags": _joined_unique(tag for unit in indexed_units for tag in _unit_tags(unit)),
                "unit_titles": _joined_unique(_field_value(_get(unit, "title")) for unit in indexed_units),
            }
        )
    return sorted(rows, key=lambda row: (-int(row["unit_count"]), _sort_key(row["keyword"])))


def _title_keywords(title: object, *, min_length: int) -> list[str]:
    keywords: set[str] = set()
    for match in _TOKEN_RE.findall(_field_value(title).casefold()):
        if len(match) >= min_length and match not in _STOP_WORDS:
            keywords.add(match)
    return sorted(keywords, key=_sort_key)


def _unit_tags(unit: KnowledgeUnit | Mapping[str, Any]) -> list[str]:
    tags = _get(unit, "tags", [])
    if isinstance(tags, str):
        return [tags] if _field_value(tags) else []
    if isinstance(tags, Iterable):
        return [_field_value(tag) for tag in tags if _field_value(tag)]
    return []


def _unit_id(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    return _field_value(_get(unit, "id")) or _field_value(_get(unit, "source_id"))


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _joined_unique(values: Iterable[object]) -> str:
    return "; ".join(sorted({_field_value(value) for value in values if _field_value(value)}, key=_sort_key))


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
