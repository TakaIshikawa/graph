"""Summarize Markdown math usage across units."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})(?P<info>\w+)?")
_INLINE_CODE_RE = re.compile(r"`+[^`\n]*`+")
_DISPLAY_RE = re.compile(r"(?<!\\)\$\$(.+?)(?<!\\)\$\$", re.DOTALL)
_INLINE_RE = re.compile(r"(?<!\\)(?<!\$)\$([^\s$](?:[^$\n]*?[^\s$])?)\$(?!\$)")


def summarize_unit_markdown_math(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    total_units = total = inline = block = 0
    units_with: set[str] = set()
    exprs: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for unit in units:
        total_units += 1
        seen = False
        for kind, text, line in _spans(unit):
            if kind == "inline" and _currency(text):
                continue
            total += 1; seen = True; exprs[field_value(text)[:80]] += 1
            inline += kind == "inline"; block += kind == "block"
            if len(examples) < sample_limit:
                examples.append({"unit_id": unit_id(unit), "title": field_value(get(unit, "title")), "line": line, "kind": kind, "preview": field_value(text)[:80]})
        if seen:
            units_with.add(unit_id(unit))
    return {"total_units": total_units, "total_expression_count": total, "units_containing_math": len(units_with), "inline_expression_count": inline, "block_expression_count": block, "top_expressions": [{"expression": expr, "count": exprs[expr]} for expr in sorted(exprs, key=lambda e: (-exprs[e], e))[:sample_limit]], "examples": examples}


def _spans(unit: Any) -> list[tuple[str, str, int]]:
    rows=[]; in_fence=False; math_fence=False; body=[]; start=0
    for line_no,line in enumerate(str(get(unit,"content") or "").splitlines(),1):
        fence=_FENCE_RE.match(line)
        if fence:
            if in_fence and math_fence:
                rows.append(("block", "\n".join(body), start))
            in_fence=not in_fence; math_fence=bool(in_fence and fence.group('info') and fence.group('info').casefold() in {'math','tex','latex'}); body=[]; start=line_no
            continue
        if in_fence:
            if math_fence: body.append(line)
            continue
        clean=_INLINE_CODE_RE.sub('', line)
        rows += [("block", m.group(1), line_no) for m in _DISPLAY_RE.finditer(clean)]
        clean=_DISPLAY_RE.sub('', clean)
        rows += [("inline", m.group(1), line_no) for m in _INLINE_RE.finditer(clean)]
    return rows


def _currency(text: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:[,.]\d{2})?", text.strip()))
