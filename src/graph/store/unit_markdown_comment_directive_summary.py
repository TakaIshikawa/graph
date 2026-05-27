"""Summarize Markdown HTML comment directives."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, sort_key, unit_id

_COMMENT_RE = re.compile(r"<!--(.*?)-->", re.DOTALL)
_LABEL_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_-]*)(?=\s*:|\s|$)")
_DIRECTIVES = {'todo','fixme','note','review','graph','graph-id','graph_id','graph-key','graph_key'}


def summarize_unit_markdown_comment_directives(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    total_units=hidden=plain=0; units_with=set(); labels=Counter(); examples=[]
    for unit in units:
        total_units += 1
        for body,line in _comments(unit):
            hidden += 1; label=_label(body)
            if label:
                labels[label] += 1; units_with.add(unit_id(unit))
                if len(examples) < sample_limit: examples.append({'unit_id': unit_id(unit), 'line': line, 'label': label, 'preview': field_value(body)[:80]})
            else:
                plain += 1
    return {'total_units': total_units, 'hidden_comment_blocks': hidden, 'plain_comment_blocks': plain, 'units_with_directives': len(units_with), 'directive_labels': [{'label': l, 'count': labels[l]} for l in sorted(labels, key=sort_key)], 'examples': examples}


def _comments(unit: Any) -> list[tuple[str,int]]:
    text=str(get(unit,'content') or '')
    starts=[0]
    for line in text.splitlines(True): starts.append(starts[-1]+len(line))
    out=[]
    for m in _COMMENT_RE.finditer(text):
        line=sum(1 for pos in starts if pos <= m.start())
        out.append((m.group(1), line))
    return out


def _label(body: str) -> str:
    m=_LABEL_RE.match(body); label=m.group(1).casefold().replace('_','-') if m else ''
    return label if label in _DIRECTIVES else ''
