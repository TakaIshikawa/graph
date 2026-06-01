"""Summarize Markdown links with trailing attribute blocks."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_LINK_ATTR_RE = re.compile(r"\[[^\]\n]+\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)\{([^}\n]+)\}")
_ATTR_KEY_RE = re.compile(r"(?:^|\s)([A-Za-z_:][A-Za-z0-9_.:-]*)(?:\s*=|\s|$)")


def summarize_unit_markdown_link_attributes(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total = count = 0
    units_with: set[str] = set()
    attr_counts: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    for unit in units:
        total += 1
        uid = unit_id(unit)
        for line_number, line in _lines_without_fences(str(get(unit, "content") or "")):
            for match in _LINK_ATTR_RE.finditer(line):
                attrs = _attribute_keys(match.group(2))
                count += 1
                units_with.add(uid)
                attr_counts.update(attrs)
                if len(samples) < limit:
                    samples.append({"unit_id": uid, "line_number": line_number, "url": match.group(1), "attributes": attrs})
    return {
        "total_units": total,
        "link_attribute_count": count,
        "units_with_link_attributes": len(units_with),
        "attribute_counts": {key: attr_counts[key] for key in sorted(attr_counts, key=sort_key)},
        "samples": samples,
    }


def _attribute_keys(block: str) -> list[str]:
    keys = {match.group(1).casefold() for match in _ATTR_KEY_RE.finditer(block) if match.group(1)}
    return sorted(keys, key=sort_key)


def _lines_without_fences(content: str) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            rows.append((line_number, line))
    return rows
