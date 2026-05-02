"""Build deterministic tag co-occurrence summaries for graph units."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from itertools import combinations
from typing import Any

from graph.types.models import KnowledgeUnit


def _validate_min_count(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("min_count must be a positive integer")
    return value


def _validate_limit(value: int | None) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("limit must be a non-negative integer or None")
    return value


def _tag_key(raw_tag: str) -> str:
    return " ".join(raw_tag.strip().casefold().split())


def _display_key(tag: str) -> tuple[str, str]:
    return (tag.casefold(), tag)


def _choose_display_label(labels: Counter[str]) -> str:
    return sorted(labels, key=lambda label: (-labels[label], _display_key(label)))[0]


def build_tag_cooccurrence_matrix(
    units: Iterable[KnowledgeUnit],
    *,
    min_count: int = 1,
    limit: int | None = None,
) -> dict[str, Any]:
    """Return tag counts and unordered tag-pair counts across units.

    Tag identity is case-insensitive. Counts are unit counts, so repeated case
    variants of the same tag within a unit do not inflate tag or pair totals.
    """
    min_count_value = _validate_min_count(min_count)
    limit_value = _validate_limit(limit)
    unit_list = list(units)

    labels_by_key: dict[str, Counter[str]] = defaultdict(Counter)
    unit_ids_by_key: dict[str, set[str]] = defaultdict(set)
    pair_unit_ids: dict[tuple[str, str], set[str]] = defaultdict(set)

    for unit in unit_list:
        unit_tag_keys: set[str] = set()
        for raw_tag in unit.tags:
            if not isinstance(raw_tag, str):
                continue
            label = " ".join(raw_tag.strip().split())
            key = _tag_key(label)
            if not key:
                continue
            labels_by_key[key][label] += 1
            unit_tag_keys.add(key)

        for key in unit_tag_keys:
            unit_ids_by_key[key].add(unit.id)

        for left, right in combinations(sorted(unit_tag_keys), 2):
            pair_unit_ids[(left, right)].add(unit.id)

    display_by_key = {
        key: _choose_display_label(labels) for key, labels in labels_by_key.items()
    }

    tags = [
        {
            "tag": display_by_key[key],
            "key": key,
            "count": len(unit_ids),
            "unit_ids": sorted(unit_ids),
        }
        for key, unit_ids in unit_ids_by_key.items()
    ]
    tags.sort(key=lambda item: (-item["count"], _display_key(item["tag"])))

    pairs = [
        {
            "source": display_by_key[left],
            "target": display_by_key[right],
            "source_key": left,
            "target_key": right,
            "count": len(unit_ids),
            "unit_ids": sorted(unit_ids),
        }
        for (left, right), unit_ids in pair_unit_ids.items()
        if len(unit_ids) >= min_count_value
    ]
    pairs.sort(
        key=lambda item: (
            -item["count"],
            _display_key(item["source"]),
            _display_key(item["target"]),
        )
    )
    if limit_value is not None:
        pairs = pairs[:limit_value]

    return {
        "tags": tags,
        "pairs": pairs,
        "stats": {
            "unit_count": len(unit_list),
            "tag_count": len(tags),
            "pair_count": len(pairs),
            "min_count": min_count_value,
            "limit": limit_value,
        },
    }
