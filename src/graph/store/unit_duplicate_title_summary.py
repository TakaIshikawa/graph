"""Summarize duplicate unit titles."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get, inline_text, sort_key, unit_id


def summarize_unit_duplicate_titles(units: Iterable[Any]) -> dict[str, Any]:
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    total = 0
    for index, unit in enumerate(units):
        total += 1
        title = inline_text(get(unit, "title"))
        if title:
            groups[title.casefold()].append((unit_id(unit) or str(index), title))
    duplicates = []
    for normalized in sorted(groups, key=sort_key):
        values = sorted(groups[normalized], key=lambda item: sort_key(item[0]))
        if len(values) > 1:
            sample_titles = sorted({title for _, title in values}, key=sort_key)
            duplicates.append(
                {
                    "normalized_title": normalized,
                    "duplicate_count": len(values),
                    "unit_ids": [uid for uid, _ in values],
                    "sample_titles": sample_titles,
                }
            )
    return {"total_units": total, "duplicate_group_count": len(duplicates), "duplicate_groups": duplicates}
