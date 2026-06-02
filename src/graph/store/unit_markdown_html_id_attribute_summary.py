"""Summarize inline HTML id attributes in Markdown content."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_TAG_RE = re.compile(r"<([A-Za-z][A-Za-z0-9:-]*)(?:\s[^<>]*)?/?>")
_ID_RE = re.compile(r"""(?:^|\s)id\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""", re.IGNORECASE)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def summarize_unit_markdown_html_id_attributes(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total = attr_count = 0
    counts: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for unit in units:
        total += 1
        uid = unit_id(unit)
        unit_count = 0
        for line_number, tag_name, raw_id in _ids(str(get(unit, "content") or "")):
            attr_count += 1
            unit_count += 1
            normalized = field_value(raw_id)
            counts[normalized] += 1
            if len(samples) < limit:
                samples.append({"unit_id": uid, "line_number": line_number, "tag_name": tag_name, "id": raw_id})
        if unit_count:
            rows.append({"unit_id": uid, "id_attribute_count": unit_count})
    duplicates = {key: counts[key] for key in sorted(counts, key=sort_key) if counts[key] > 1}
    rows.sort(key=lambda row: sort_key(row["unit_id"]))
    samples.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["id"])))
    return {
        "total_units": total,
        "id_attribute_count": attr_count,
        "unique_id_count": len(counts),
        "duplicate_id_count": len(duplicates),
        "duplicate_ids": duplicates,
        "samples": samples[:limit],
        "units": rows,
    }


def _ids(content: str) -> list[tuple[int, str, str]]:
    rows: list[tuple[int, str, str]] = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        clean_line = _COMMENT_RE.sub("", line)
        for tag_match in _TAG_RE.finditer(clean_line):
            id_match = _ID_RE.search(tag_match.group(0))
            if id_match:
                rows.append((line_number, tag_match.group(1).casefold(), next(group for group in id_match.groups() if group is not None)))
    return rows
