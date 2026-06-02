"""Summarize inline HTML style attributes in Markdown content."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_TAG_RE = re.compile(r"<([A-Za-z][A-Za-z0-9:-]*)(?:\s[^<>]*)?/?>")
_STYLE_RE = re.compile(r"""(?:^|\s)style\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""", re.IGNORECASE)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def summarize_unit_markdown_html_style_attributes(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total = units_with = attr_count = property_count = 0
    properties: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for unit in units:
        total += 1
        uid = unit_id(unit)
        unit_attrs = unit_props = 0
        for line_number, tag, style in _styles(str(get(unit, "content") or "")):
            names = _property_names(style)
            unit_attrs += 1
            unit_props += len(names)
            properties.update(names)
            if len(samples) < limit:
                samples.append({"unit_id": uid, "line_number": line_number, "tag_name": tag, "style_text": style})
        if unit_attrs:
            units_with += 1
            attr_count += unit_attrs
            property_count += unit_props
            rows.append({"unit_id": uid, "style_attribute_count": unit_attrs, "property_count": unit_props})
    rows.sort(key=lambda row: sort_key(row["unit_id"]))
    top = [{"property": key, "count": properties[key]} for key in sorted(properties, key=lambda key: (-properties[key], sort_key(key)))]
    return {
        "total_units": total,
        "units_with_style_attributes": units_with,
        "style_attribute_count": attr_count,
        "property_count": property_count,
        "top_properties": top[:limit],
        "samples": samples[:limit],
        "units": rows,
    }


def _styles(content: str) -> list[tuple[int, str, str]]:
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
            style_match = _STYLE_RE.search(tag_match.group(0))
            if style_match:
                rows.append((line_number, tag_match.group(1).casefold(), field_value(next(group for group in style_match.groups() if group is not None))))
    return rows


def _property_names(style: str) -> list[str]:
    names = []
    for part in style.split(";"):
        if ":" in part:
            name = field_value(part.split(":", 1)[0]).casefold()
            if name:
                names.append(name)
    return names
