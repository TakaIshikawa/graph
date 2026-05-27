"""Summarize YAML frontmatter block scalar usage."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get, sort_key, unit_id

_BLOCK_RE = re.compile(r"^([A-Za-z0-9_-]+)\s*:\s*([|>])([+-]?)\s*(?:#.*)?$")


def summarize_unit_yaml_block_scalars(units: Iterable[Any], *, sample_limit: int = 5) -> dict[str, Any]:
    total_units = units_with_block_scalars = block_scalar_count = 0
    styles: Counter[str] = Counter()
    chomping: Counter[str] = Counter()
    keys: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "literal_count": 0, "folded_count": 0, "chomping_counts": Counter(), "samples": []})
    for index, unit in enumerate(units):
        total_units += 1
        uid = unit_id(unit) or str(index)
        fields = _fields(str(get(unit, "content") or ""))
        if fields:
            units_with_block_scalars += 1
        for key, style, chomp in fields:
            block_scalar_count += 1
            style_name = "literal" if style == "|" else "folded"
            chomp_name = chomp or "clip"
            styles[style_name] += 1
            chomping[chomp_name] += 1
            row = keys[key]
            row["count"] += 1
            row[f"{style_name}_count"] += 1
            row["chomping_counts"][chomp_name] += 1
            if len(row["samples"]) < sample_limit:
                row["samples"].append({"unit_id": uid, "indicator": f"{style}{chomp}"})
    return {
        "total_units": total_units,
        "units_with_block_scalars": units_with_block_scalars,
        "block_scalar_count": block_scalar_count,
        "style_counts": _counter_rows(styles, "style"),
        "chomping_counts": _counter_rows(chomping, "chomping"),
        "keys": [
            {
                "key": key,
                "count": keys[key]["count"],
                "literal_count": keys[key]["literal_count"],
                "folded_count": keys[key]["folded_count"],
                "chomping_counts": _counter_rows(keys[key]["chomping_counts"], "chomping"),
                "samples": keys[key]["samples"],
            }
            for key in sorted(keys, key=sort_key)
        ],
    }


def _fields(content: str) -> list[tuple[str, str, str]]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return []
    rows: list[tuple[str, str, str]] = []
    for line in lines[1:]:
        if line.strip() == "---":
            break
        match = _BLOCK_RE.match(line.strip())
        if match:
            rows.append(match.groups())
    return rows


def _counter_rows(counter: Counter[str], key: str) -> list[dict[str, Any]]:
    return [{key: name, "count": counter[name]} for name in sorted(counter, key=sort_key)]
