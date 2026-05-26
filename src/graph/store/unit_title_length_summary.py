"""Summarize unit title lengths by source."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any


def summarize_unit_title_lengths(units: Iterable[Any]) -> dict[str, Any]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    total_units = 0
    for unit in units:
        total_units += 1
        grouped[_source(unit)].append(unit)
    rows = [_row(source, grouped[source]) for source in sorted(grouped, key=_sort_key)]
    return {"total_units": total_units, "rows": rows, "source_summaries": rows}


def _row(source: str, units: list[Any]) -> dict[str, Any]:
    titled = [(_unit_id(unit), _title(unit), len(_title(unit))) for unit in units if _title(unit)]
    lengths = [length for _unit_id, _title, length in titled]
    shortest = sorted(titled, key=lambda item: (item[2], _sort_key(item[0])))[0][0] if titled else ""
    longest = sorted(titled, key=lambda item: (-item[2], _sort_key(item[0])))[0][0] if titled else ""
    return {
        "source": source,
        "unit_count": len(units),
        "missing_title_count": sum(1 for unit in units if not _title(unit)),
        "min_title_length": min(lengths, default=0),
        "max_title_length": max(lengths, default=0),
        "average_title_length": f"{(sum(lengths) / len(lengths)):.2f}" if lengths else "0.00",
        "shortest_title_unit_id": shortest,
        "longest_title_unit_id": longest,
    }


def _source(unit: Any) -> str:
    meta = _metadata(unit)
    return _text(_get(unit, "source_project") or meta.get("source") or meta.get("source_project")) or "unknown"


def _title(unit: Any) -> str:
    return _text(_get(unit, "title") or _metadata(unit).get("title"))


def _unit_id(unit: Any) -> str:
    return _text(_get(unit, "id") or _get(unit, "unit_id"))


def _metadata(unit: Any) -> Mapping[str, Any]:
    value = _get(unit, "metadata")
    return value if isinstance(value, Mapping) else {}


def _get(item: Any, key: str) -> Any:
    return item.get(key) if isinstance(item, Mapping) else getattr(item, key, None)


def _text(value: Any) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def _sort_key(value: Any) -> tuple[str, str]:
    text = _text(value)
    return (text.casefold(), text)
