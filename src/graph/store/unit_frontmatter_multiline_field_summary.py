"""Summarize block scalar fields in leading YAML frontmatter."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get, sort_key, unit_id

_FIELD_RE = re.compile(r"^\s*([A-Za-z0-9_-]+)\s*:\s*([|>])([+-]?)\s*(?:#.*)?$")


def summarize_unit_frontmatter_multiline_fields(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    total = with_frontmatter = with_fields = field_total = 0
    styles: Counter[str] = Counter()
    chomping: Counter[str] = Counter()
    keys: Counter[str] = Counter()
    examples: list[dict[str, str | int]] = []
    for index, unit in enumerate(units):
        total += 1
        uid = unit_id(unit) or str(index)
        lines = _frontmatter_lines(str(get(unit, "content") or ""))
        if lines is None:
            continue
        with_frontmatter += 1
        unit_fields = 0
        for line_no, line in lines:
            match = _FIELD_RE.match(line)
            if not match:
                continue
            key, marker, chomp = match.groups()
            style = "literal" if marker == "|" else "folded"
            mode = {"+": "keep", "-": "strip", "": "clip"}[chomp]
            unit_fields += 1
            field_total += 1
            styles[style] += 1
            chomping[mode] += 1
            keys[key] += 1
            if len(examples) < sample_limit:
                examples.append({"unit_id": uid, "key": key, "style": style, "chomping": mode, "line": line_no})
        if unit_fields:
            with_fields += 1
    return {"total_units": total, "units_with_frontmatter": with_frontmatter, "units_with_multiline_fields": with_fields, "total_multiline_fields": field_total, "style_counts": dict(sorted(styles.items())), "chomping_counts": dict(sorted(chomping.items())), "key_counts": {key: keys[key] for key in sorted(keys, key=sort_key)}, "examples": examples}


def _frontmatter_lines(content: str) -> list[tuple[int, str]] | None:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    rows = []
    for index, line in enumerate(lines[1:], start=2):
        if line.strip() == "---":
            return rows
        rows.append((index, line))
    return None
