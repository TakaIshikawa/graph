"""Summarize HTML data attributes in unit content."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_TAG_RE = re.compile(r"<([A-Za-z][A-Za-z0-9:-]*)([^<>]*)>")
_ATTR_RE = re.compile(r"""\s(data-[A-Za-z0-9_.:-]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?""")


def summarize_unit_html_data_attributes(units: Iterable[Any], sample_limit: int = 10) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total = units_with = count = 0
    attrs: Counter[str] = Counter()
    tags: Counter[str] = Counter()
    samples: list[dict[str, str | int]] = []
    for unit in units:
        total += 1
        uid = unit_id(unit)
        found = False
        for line_number, line in _content_lines(str(get(unit, "content") or "")):
            for tag_match in _TAG_RE.finditer(line):
                tag, attr_text = tag_match.groups()
                for attr_match in _ATTR_RE.finditer(attr_text):
                    attr = attr_match.group(1)
                    value = next((item for item in attr_match.groups()[1:] if item is not None), "")
                    found = True
                    count += 1
                    attrs[attr] += 1
                    tags[tag] += 1
                    if len(samples) < limit:
                        samples.append({"unit_id": uid, "line_number": line_number, "tag": tag, "attribute": attr, "value": value})
        if found:
            units_with += 1
    return {"total_units": total, "units_with_html_data_attributes": units_with, "html_data_attribute_count": count, "attribute_counts": dict(sorted(attrs.items())), "tag_counts": dict(sorted(tags.items())), "data_attribute_samples": samples}


def _content_lines(content: str) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            rows.append((line_number, line))
    return rows
