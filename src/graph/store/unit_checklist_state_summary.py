"""Summarize Markdown checklist item states across units."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_TASK_RE = re.compile(r"^(?P<indent>[ \t]*)(?:[-+*]|\d+[.)])\s+\[(?P<marker>[^\]\n]*)\]\s*(?P<text>.*)$")


def summarize_unit_checklist_states(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    total_units=total_items=nested=0; markers=Counter(); states=Counter(); examples=defaultdict(list)
    for unit in units:
        total_units += 1
        for item in _items(unit):
            total_items += 1; markers[item['marker']] += 1; states[item['state']] += 1; nested += item['depth'] > 0
            if len(examples[item['state']]) < sample_limit: examples[item['state']].append({'unit_id': unit_id(unit), 'line': item['line'], 'item_text': item['text']})
    return {'total_units': total_units, 'total_items': total_items, 'state_marker_counts': [{'marker': k, 'count': markers[k]} for k in sorted(markers, key=sort_key)], 'normalized_state_counts': [{'state': k, 'count': states[k]} for k in sorted(states, key=sort_key)], 'nested_item_count': nested, 'examples_by_state': {k: examples[k] for k in sorted(examples, key=sort_key)}}


def _items(unit: Any) -> list[dict[str, Any]]:
    out=[]; in_fence=False
    for line_no,line in enumerate(str(get(unit,'content') or '').splitlines(),1):
        if _FENCE_RE.match(line): in_fence=not in_fence; continue
        if in_fence: continue
        m=_TASK_RE.match(line)
        if m:
            marker=m.group('marker'); out.append({'marker': marker, 'state': _state(marker), 'text': field_value(m.group('text')), 'line': line_no, 'depth': len(m.group('indent').replace('\t','    '))//2})
    return out


def _state(marker: str) -> str:
    if marker == ' ' or marker == '': return 'open'
    if marker.casefold() == 'x': return 'done'
    return 'custom'
