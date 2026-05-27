"""Summarize source-label to target-label pairs on relations."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key

_SOURCE_LABEL_KEYS = ("source_label", "from_label")
_TARGET_LABEL_KEYS = ("target_label", "to_label")


def summarize_relation_label_pairs(relations: Iterable[Any]) -> dict[str, Any]:
    counts: Counter[tuple[str, str]] = Counter()
    total = 0
    for relation in relations:
        total += 1
        meta = metadata(relation)
        source = _first(relation, meta, _SOURCE_LABEL_KEYS) or "unknown"
        target = _first(relation, meta, _TARGET_LABEL_KEYS) or "unknown"
        counts[(source.casefold(), target.casefold())] += 1
    rows = [
        {"source_label": source, "target_label": target, "count": count}
        for (source, target), count in sorted(counts.items(), key=lambda item: (sort_key(item[0][0]), sort_key(item[0][1])))
    ]
    return {"total_relations": total, "pair_counts": rows}


def _first(item: Any, meta: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = field_value(get(item, key)) or field_value(meta.get(key))
        if value:
            return value
    return ""
