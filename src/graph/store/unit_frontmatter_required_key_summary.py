"""Summarize required frontmatter metadata keys on units."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, metadata, sort_key, unit_id


def summarize_unit_frontmatter_required_keys(units: Iterable[Any], required_keys: Iterable[str]) -> dict[str, Any]:
    required = [field_value(key) for key in required_keys if field_value(key)]
    total = complete = 0
    missing_counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    for index, unit in enumerate(units):
        total += 1
        keys = {field_value(key) for key in metadata(unit)}
        missing = [key for key in required if key not in keys]
        if not missing:
            complete += 1
            continue
        uid = unit_id(unit) or str(index)
        missing_counts.update(missing)
        for key in missing:
            if len(examples[key]) < 5:
                examples[key].append(uid)
    return {
        "total_units": total,
        "complete_units": complete,
        "incomplete_units": total - complete,
        "missing_key_counts": [{"key": key, "count": missing_counts[key]} for key in sorted(missing_counts, key=sort_key)],
        "examples_by_missing_key": {key: examples[key] for key in sorted(examples, key=sort_key)},
    }
