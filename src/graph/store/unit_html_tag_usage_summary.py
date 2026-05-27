"""Summarize inline HTML tags in unit content."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_TAG_RE = re.compile(r"<\s*(/)?\s*([A-Za-z][A-Za-z0-9:-]*)([^<>]*?)(/)?\s*>")
_COMMENT_RE = re.compile(r"<!--.*?-->")


def summarize_unit_html_tag_usage(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    total = units_with = tag_total = self_closing = closing = attr = 0
    counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, str | int]]] = defaultdict(list)
    for index, unit in enumerate(units):
        total += 1
        uid = unit_id(unit) or str(index)
        found = False
        for line_no, line in _content_lines(unit):
            line = _COMMENT_RE.sub("", line)
            for match in _TAG_RE.finditer(line):
                slash, name, tail, endslash = match.groups()
                tag = name.casefold()
                found = True
                tag_total += 1
                counts[tag] += 1
                closing += 1 if slash else 0
                self_closing += 1 if endslash or tail.strip().endswith("/") else 0
                attr += 1 if (not slash and tail.strip().strip("/")) else 0
                if len(examples[tag]) < sample_limit:
                    examples[tag].append({"unit_id": uid, "line": line_no, "tag": tag, "snippet": match.group(0)})
        if found:
            units_with += 1
    top = [{"tag": tag, "count": counts[tag], "examples": examples[tag]} for tag in sorted(counts, key=lambda k: (-counts[k], sort_key(k)))]
    return {"total_units": total, "units_with_html_tags": units_with, "total_tags": tag_total, "unique_tag_names": len(counts), "self_closing_tag_count": self_closing, "closing_tag_count": closing, "attribute_tag_count": attr, "top_tags": top, "examples": [item for tag in sorted(examples, key=sort_key) for item in examples[tag]][:sample_limit]}


def _content_lines(unit: Any) -> list[tuple[int, str]]:
    in_fence = False
    rows = []
    for line_no, line in enumerate(str(get(unit, "content") or "").splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            rows.append((line_no, line))
    return rows
