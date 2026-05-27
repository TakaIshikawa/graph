"""Annotation density summary for store units."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from numbers import Number
from typing import Any

ANNOTATION_KEYS = ("annotations", "comments", "highlights", "notes", "marginalia")


def summarize_unit_annotation_density(units: Iterable[Mapping[str, Any] | object]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any] | object]] = defaultdict(list)
    for unit in units:
        grouped[_source(unit)].append(unit)

    rows: list[dict[str, Any]] = []
    for source in sorted(grouped, key=_sort_key):
        totals = {key: 0 for key in ANNOTATION_KEYS}
        per_unit: list[int] = []
        for unit in grouped[source]:
            counts = {key: _annotation_count(_field(unit, key)) for key in ANNOTATION_KEYS}
            per_unit.append(sum(counts.values()))
            for key, count in counts.items():
                totals[key] += count
        unit_count = len(grouped[source])
        total_annotations = sum(per_unit)
        rows.append(
            {
                "source": source,
                "source_project": source,
                "unit_count": unit_count,
                "annotated_count": sum(1 for count in per_unit if count > 0),
                "comment_count": totals["comments"],
                "highlight_count": totals["highlights"],
                "note_count": totals["notes"],
                "annotation_count": totals["annotations"],
                "marginalia_count": totals["marginalia"],
                "total_annotations": total_annotations,
                "max_annotations": max(per_unit) if per_unit else 0,
                "average_annotations": round(total_annotations / unit_count, 2) if unit_count else 0.0,
            }
        )
    return {"rows": rows, "source_summaries": rows, "total_units": sum(row["unit_count"] for row in rows)}


def _annotation_count(value: object) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, Number):
        return max(0, int(value))
    if isinstance(value, (list, tuple, set)):
        return len(value)
    if isinstance(value, Mapping):
        return len(value)
    return 1 if _text(value) else 0


def _field(unit: Mapping[str, Any] | object, key: str) -> object:
    value = _get(unit, key)
    if value not in (None, ""):
        return value
    return _metadata(unit).get(key)


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
