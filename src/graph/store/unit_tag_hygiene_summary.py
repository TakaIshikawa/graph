"""Tag hygiene summary for store units."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from string import punctuation
from typing import Any


def summarize_unit_tag_hygiene(units: Iterable[Any], *, sample_limit: int = 5) -> dict[str, Any]:
    groups: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {"unit_ids": set(), "tags": set()})
    total = 0
    for index, unit in enumerate(units):
        total += 1
        unit_id = _unit_id(unit, index)
        normalized_counts: dict[str, int] = defaultdict(int)
        for tag in _raw_tags(unit):
            normalized = _normalize(tag)
            normalized_counts[normalized] += 1
            for issue in _issues(tag, normalized):
                group = groups[(issue, normalized)]
                group["unit_ids"].add(unit_id)
                group["tags"].add(tag)
        for normalized, count in normalized_counts.items():
            if normalized and count > 1:
                group = groups[("duplicate_normalized_tag", normalized)]
                group["unit_ids"].add(unit_id)
                group["tags"].add(normalized)
    rows = []
    for issue, normalized in sorted(groups, key=lambda key: (_sort_key(key[0]), _sort_key(key[1]))):
        group = groups[(issue, normalized)]
        unit_ids = sorted(group["unit_ids"], key=_sort_key)
        tag = sorted(group["tags"], key=_sort_key)[0]
        rows.append(
            {
                "issue_type": issue,
                "tag": tag,
                "normalized_tag": normalized,
                "unit_count": len(unit_ids),
                "example_unit_ids": unit_ids[:sample_limit],
            }
        )
    return {"total_units": total, "rows": rows}


def _issues(tag: str, normalized: str) -> list[str]:
    issues = []
    if not normalized:
        issues.append("empty_tag")
        return issues
    if tag != tag.strip():
        issues.append("surrounding_whitespace")
    if "  " in tag.strip() or "\t" in tag or "\n" in tag:
        issues.append("repeated_whitespace")
    if any(char.isupper() for char in tag):
        issues.append("uppercase_variant")
    punct_count = sum(1 for char in tag if char in punctuation)
    if punct_count >= max(2, len(tag.strip()) // 2):
        issues.append("punctuation_heavy")
    return issues


def _raw_tags(unit: Any) -> list[str]:
    raw = _get(unit, "tags")
    if raw is None:
        raw = _metadata(unit).get("tags")
    if isinstance(raw, str):
        values = raw.split(",")
    elif isinstance(raw, list | tuple | set):
        values = list(raw)
    else:
        values = []
    return ["" if value is None else str(value) for value in values]


def _normalize(tag: str) -> str:
    return " ".join(tag.strip().casefold().split())


def _metadata(unit: Any) -> Mapping[str, Any]:
    value = _get(unit, "metadata")
    return value if isinstance(value, Mapping) else {}


def _unit_id(unit: Any, index: int) -> str:
    return _text(_get(unit, "id") or _metadata(unit).get("id")) or str(index)


def _get(item: Any, key: str) -> Any:
    return item.get(key) if isinstance(item, Mapping) else getattr(item, key, None)


def _text(value: Any) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def _sort_key(value: Any) -> tuple[str, str]:
    text = _text(value)
    return (text.casefold(), text)
