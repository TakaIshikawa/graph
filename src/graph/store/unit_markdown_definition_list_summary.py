"""Summarize Markdown definition-list usage across units."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_DEF_RE = re.compile(r"^\s*:\s+(.+)$")


def summarize_unit_markdown_definition_lists(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    total_units=blocks=terms=defs=multi=loose=0; examples=[]
    for unit in units:
        parsed=_blocks(unit); blocks += len(parsed)
        for block in parsed:
            terms += len(block['terms']); defs += sum(len(v) for v in block['terms'].values()); multi += sum(1 for v in block['terms'].values() if len(v) > 1); loose += block['loose']
            if len(examples) < sample_limit: examples.append({'unit_id': unit_id(unit), 'line': block['line'], 'term': next(iter(block['terms']))})
        total_units += 1
    return {'total_units': total_units, 'definition_list_blocks': blocks, 'term_count': terms, 'definition_count': defs, 'multi_definition_term_count': multi, 'loose_spacing_variant_count': loose, 'examples': examples}


def _blocks(unit: Any) -> list[dict[str, Any]]:
    lines=[]; in_fence=False
    for no,line in enumerate(str(get(unit,'content') or '').splitlines(),1):
        if _FENCE_RE.match(line): in_fence=not in_fence; continue
        if not in_fence: lines.append((no,line))
    out=[]; i=0
    while i < len(lines)-1:
        no,term=lines[i]
        terms=[]
        while i < len(lines):
            term_text=field_value(lines[i][1])
            if not term_text or lines[i][1].startswith(' ') or '://' in term_text or term_text.startswith(':'):
                break
            terms.append((lines[i][0], term_text)); i += 1
        if not terms:
            i += 1; continue
        j=i; blanks=0
        while j < len(lines) and not lines[j][1].strip(): blanks += 1; j += 1
        if j < len(lines) and _DEF_RE.match(lines[j][1]):
            defs=[]; loose_count=1 if blanks else 0
            while j < len(lines) and (m := _DEF_RE.match(lines[j][1])):
                defs.append(field_value(m.group(1))); j += 1
            out.append({'line': no, 'terms': {term_text: list(defs) for _, term_text in terms}, 'loose': loose_count}); i=j; continue
    return out
