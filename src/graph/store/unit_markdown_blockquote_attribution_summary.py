"""Summarize Markdown blockquote attributions."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, sort_key, unit_id

_ATTR_RE = re.compile(r"(?:^|\n)\s*(?:--|—|cite:|source:)\s*(?P<name>.+?)\s*$", re.IGNORECASE)


def summarize_unit_markdown_blockquote_attributions(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    total = attributed = 0
    counts: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    for unit in units:
        for start, end, text in _blocks(str(get(unit, "content") or "")):
            total += 1
            match = _ATTR_RE.search(text)
            name = field_value(match.group("name")) if match else ""
            if name:
                attributed += 1
                counts[name] += 1
            if len(samples) < sample_limit:
                samples.append({"unit_id": unit_id(unit), "line_range": f"{start}-{end}", "preview": field_value(text)[:120], "attribution": name})
    return {"total_blockquotes": total, "attributed_count": attributed, "unattributed_count": total - attributed, "attribution_counts": [{"attribution": key, "count": counts[key]} for key in sorted(counts, key=sort_key)], "samples": samples}


def _blocks(content: str) -> list[tuple[int, int, str]]:
    rows: list[tuple[int, int, str]] = []
    current: list[str] = []
    start = 0
    last = 0
    for line_number, line in enumerate(content.splitlines(), start=1):
        if line.lstrip().startswith(">"):
            if not current:
                start = line_number
            current.append(line.lstrip()[1:].strip())
            last = line_number
        elif current:
            rows.append((start, last, "\n".join(current)))
            current = []
    if current:
        rows.append((start, last, "\n".join(current)))
    return rows
