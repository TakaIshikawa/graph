"""Metadata cardinality summary for store units."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any


def summarize_unit_metadata_cardinality(units: Iterable[Any], *, sample_limit: int = 5) -> dict[str, Any]:
    total = 0
    seen: dict[str, int] = defaultdict(int)
    blanks: dict[str, int] = defaultdict(int)
    values: dict[str, Counter[str]] = defaultdict(Counter)
    for unit in units:
        total += 1
        for key, raw in _metadata(unit).items():
            seen[key] += 1
            flattened = _flatten(raw)
            nonblank = [_text(value) for value in flattened if _text(value)]
            if not nonblank:
                blanks[key] += 1
            for value in nonblank:
                values[key][value] += 1
    rows = []
    for key in sorted(seen, key=_sort_key):
        counter = values[key]
        rows.append(
            {
                "key": key,
                "unit_count": seen[key],
                "distinct_value_count": len(counter),
                "blank_value_count": blanks[key],
                "repeated_value_count": sum(count for count in counter.values() if count > 1),
                "frequent_values": [
                    {"value": value, "count": count}
                    for value, count in sorted(counter.items(), key=lambda item: (-item[1], _sort_key(item[0])))[:sample_limit]
                ],
            }
        )
    return {"total_units": total, "rows": rows}


def _metadata(unit: Any) -> Mapping[str, Any]:
    value = _get(unit, "metadata")
    return value if isinstance(value, Mapping) else {}


def _flatten(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list | tuple | set):
        return [item for child in value for item in _flatten(child)]
    return [value]


def _get(item: Any, key: str) -> Any:
    return item.get(key) if isinstance(item, Mapping) else getattr(item, key, None)


def _text(value: Any) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def _sort_key(value: Any) -> tuple[str, str]:
    text = _text(value)
    return (text.casefold(), text)
