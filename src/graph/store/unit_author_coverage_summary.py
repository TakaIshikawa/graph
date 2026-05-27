"""Author coverage summary for store units."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

AUTHOR_KEYS = ("author", "authors", "creator", "created_by", "owner")


def summarize_unit_author_coverage(units: Iterable[Mapping[str, Any] | object]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any] | object]] = defaultdict(list)
    total_units = 0
    for unit in units:
        total_units += 1
        grouped[_source(unit)].append(unit)

    rows: list[dict[str, Any]] = []
    for source in sorted(grouped, key=_sort_key):
        source_units = grouped[source]
        author_counts: Counter[str] = Counter()
        authored_count = 0
        multi_author_count = 0
        for unit in source_units:
            authors = _unit_authors(unit)
            if authors:
                authored_count += 1
                author_counts.update(authors)
            if len(authors) > 1:
                multi_author_count += 1
        rows.append(
            {
                "source": source,
                "source_project": source,
                "unit_count": len(source_units),
                "authored_count": authored_count,
                "missing_author_count": len(source_units) - authored_count,
                "multi_author_count": multi_author_count,
                "top_authors": _top_counts(author_counts),
            }
        )
    return {"total_units": total_units, "rows": rows, "source_summaries": rows}


def _unit_authors(unit: Mapping[str, Any] | object) -> list[str]:
    values: list[str] = []
    metadata = _metadata(unit)
    for key in AUTHOR_KEYS:
        values.extend(_author_values(_get(unit, key)))
        values.extend(_author_values(metadata.get(key)))
    seen: set[str] = set()
    authors: list[str] = []
    for value in values:
        normalized = _normalize_author(value)
        if normalized and normalized.casefold() not in seen:
            seen.add(normalized.casefold())
            authors.append(normalized)
    return authors


def _author_values(value: object) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [_text(item) for item in value]
    text = _text(value)
    if not text:
        return []
    if "," in text or ";" in text:
        return [_text(part) for part in text.replace(";", ",").split(",")]
    return [text]


def _normalize_author(value: object) -> str:
    return " ".join(_text(value).split())


def _top_counts(counts: Counter[str]) -> list[dict[str, Any]]:
    return [{"author": author, "count": count} for author, count in sorted(counts.items(), key=lambda item: (-item[1], item[0].casefold(), item[0]))]


def _source(unit: Mapping[str, Any] | object) -> str:
    metadata = _metadata(unit)
    return _text(_get(unit, "source_project")) or _text(_get(unit, "source")) or _text(metadata.get("source_project")) or _text(metadata.get("source")) or "unknown"


def _metadata(unit: Mapping[str, Any] | object) -> Mapping[str, Any]:
    value = _get(unit, "metadata")
    return value if isinstance(value, Mapping) else {}


def _get(value: Mapping[str, Any] | object, key: str) -> object:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _text(value: object) -> str:
    return "" if value is None else str(getattr(value, "value", value)).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _text(value)
    return (text.casefold(), text)
