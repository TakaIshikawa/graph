"""Summarize Markdown inline and block math spans."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, sort_key, unit_id

_BLOCK_RE = re.compile(r"(?<!\\)\$\$(.+?)(?<!\\)\$\$", re.DOTALL)
_INLINE_RE = re.compile(r"(?<!\\)(?<!\$)\$([^\s$](?:[^$\n]*?[^\s$])?)\$(?!\$)")


def summarize_unit_markdown_math_spans(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    total = inline = block = 0
    units_with: set[str] = set()
    by_unit: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    for unit in units:
        uid = unit_id(unit)
        for kind, expr, start, end in _spans(str(get(unit, "content") or "")):
            total += 1
            inline += kind == "inline"
            block += kind == "block"
            units_with.add(uid)
            by_unit[uid] += 1
            if len(samples) < sample_limit:
                samples.append({"unit_id": uid, "line_range": f"{start}-{end}", "kind": kind, "expression": field_value(expr)[:120]})
    return {"total_math_spans": total, "units_with_math": len(units_with), "inline_count": inline, "block_count": block, "expression_samples": samples, "units_by_count": [{"unit_id": key, "count": by_unit[key]} for key in sorted(by_unit, key=lambda k: (-by_unit[k], sort_key(k)))]}


def _spans(content: str) -> list[tuple[str, str, int, int]]:
    rows: list[tuple[str, str, int, int]] = []
    for match in _BLOCK_RE.finditer(content):
        start = content[: match.start()].count("\n") + 1
        end = start + match.group(0).count("\n")
        rows.append(("block", match.group(1), start, end))
    without_blocks = _BLOCK_RE.sub("", content)
    for line_number, line in enumerate(without_blocks.splitlines(), start=1):
        for match in _INLINE_RE.finditer(line):
            rows.append(("inline", match.group(1), line_number, line_number))
    return rows
