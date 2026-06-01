"""Summarize aliases declared in unit YAML frontmatter."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

import yaml

from graph.export._report_csv import field_value, get, sort_key, unit_id


def summarize_unit_frontmatter_aliases(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total = units_with = alias_count = 0
    counts: Counter[str] = Counter()
    display: dict[str, str] = {}
    samples: list[dict[str, str]] = []
    for index, unit in enumerate(units):
        total += 1
        unit_aliases: list[tuple[str, str]] = []
        for key, value in _frontmatter(str(get(unit, "content") or "")).items():
            if field_value(key).casefold() not in {"alias", "aliases"}:
                continue
            for alias in _aliases(value):
                unit_aliases.append((field_value(key), alias))
                normalized = alias.casefold()
                counts[normalized] += 1
                display.setdefault(normalized, alias)
                alias_count += 1
                if len(samples) < limit:
                    samples.append({"unit_id": unit_id(unit) or str(index), "key": field_value(key), "alias": alias})
        if unit_aliases:
            units_with += 1
    duplicates = [display[key] for key, count in counts.items() if count > 1]
    duplicates.sort(key=sort_key)
    return {
        "total_units": total,
        "units_with_aliases": units_with,
        "alias_count": alias_count,
        "duplicate_aliases": duplicates,
        "samples": samples,
    }


def _frontmatter(content: str) -> dict[str, Any]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    block: list[str] = []
    for line in lines[1:]:
        if line.strip() == "---":
            try:
                parsed = yaml.safe_load("\n".join(block))
            except yaml.YAMLError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        block.append(line)
    return {}


def _aliases(value: Any) -> list[str]:
    values = value if isinstance(value, list | tuple | set) else [value]
    return [alias for alias in (field_value(item) for item in values) if alias]
