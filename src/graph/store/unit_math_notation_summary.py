"""Summarize Markdown and LaTeX math notation in units."""

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
_BRACKET_RE = re.compile(r"(?<!\\)\\\[(.+?)(?<!\\)\\\]|(?<!\\)\\\((.+?)(?<!\\)\\\)", re.DOTALL)


def summarize_unit_math_notation(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    total_units=0; counts=Counter(); examples=[]; unclosed_examples=[]
    for unit in units:
        total_units += 1
        unit_counts, unit_examples, unit_unclosed_examples = _counts(unit, sample_limit)
        counts.update(unit_counts); examples.extend(unit_examples[:max(0, sample_limit-len(examples))])
        unclosed_examples.extend(unit_unclosed_examples[:max(0, sample_limit-len(unclosed_examples))])
    return {'total_units': total_units, 'total_math_spans': sum(counts[k] for k in ('inline_dollar','display_dollar','bracket_math','fenced_math')), 'delimiter_counts': [{'delimiter': k, 'count': counts[k]} for k in sorted((k for k in counts if k != 'unclosed'), key=sort_key)], 'unclosed_delimiter_count': counts['unclosed'], 'examples': examples, 'unclosed_examples': unclosed_examples}


def _counts(unit: Any, limit: int) -> tuple[Counter[str], list[dict[str, Any]], list[dict[str, Any]]]:
    counts=Counter(); examples=[]; unclosed_examples=[]; in_fence=False; math_fence=False; body=[]; start=0
    for line_no,line in enumerate(str(get(unit,'content') or '').splitlines(),1):
        fence=_FENCE_RE.match(line)
        if fence:
            if in_fence and math_fence:
                counts['fenced_math'] += 1; _example(examples, unit, start, 'fenced_math', '\n'.join(body), limit)
            in_fence=not in_fence; math_fence=bool(in_fence and fence.group('info') and fence.group('info').casefold() in {'math','tex','latex'}); body=[]; start=line_no
            continue
        if in_fence:
            if math_fence: body.append(line)
            continue
        clean=_INLINE_CODE_RE.sub('', line)
        for regex,key in ((_DISPLAY_RE,'display_dollar'), (_INLINE_RE,'inline_dollar'), (_BRACKET_RE,'bracket_math')):
            for m in regex.finditer(clean):
                text=next(g for g in m.groups() if g is not None)
                if key != 'inline_dollar' or not re.fullmatch(r'\d+(?:[,.]\d{2})?', text.strip()):
                    counts[key] += 1; _example(examples, unit, line_no, key, text, limit)
        if clean.count('$') % 2 == 1 or clean.count(r'\[') != clean.count(r'\]') or clean.count(r'\(') != clean.count(r'\)'):
            counts['unclosed'] += 1
            if len(unclosed_examples) < limit:
                unclosed_examples.append({'unit_id': unit_id(unit), 'line': line_no, 'preview': field_value(clean)[:80]})
    return counts, examples, unclosed_examples


def _example(examples: list[dict[str, Any]], unit: Any, line: int, delimiter: str, text: str, limit: int) -> None:
    if len(examples) < limit: examples.append({'unit_id': unit_id(unit), 'line': line, 'delimiter': delimiter, 'preview': field_value(text)[:80]})
