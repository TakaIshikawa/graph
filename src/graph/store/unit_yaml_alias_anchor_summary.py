"""Summarize YAML anchors and aliases in frontmatter sections."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get, sort_key, unit_id

_ANCHOR_RE = re.compile(r"(?<!\w)&([A-Za-z0-9_-]+)")
_ALIAS_RE = re.compile(r"(?<!\w)\*([A-Za-z0-9_-]+)")


def summarize_unit_yaml_alias_anchors(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    total_units=anchor_total=alias_total=0; anchors=Counter(); aliases=Counter(); unresolved=Counter(); examples=[]
    for unit in units:
        total_units += 1; text=_frontmatter(unit)
        if not text: continue
        unit_anchors=_ANCHOR_RE.findall(text); unit_aliases=_ALIAS_RE.findall(text)
        anchor_total += len(unit_anchors); alias_total += len(unit_aliases); anchors.update(unit_anchors); aliases.update(unit_aliases)
        for alias in unit_aliases:
            if alias not in unit_anchors: unresolved[alias] += 1
        if (unit_anchors or unit_aliases) and len(examples) < sample_limit: examples.append({'unit_id': unit_id(unit), 'anchors': sorted(set(unit_anchors), key=sort_key), 'aliases': sorted(set(unit_aliases), key=sort_key)})
    return {'total_units': total_units, 'anchor_count': anchor_total, 'alias_count': alias_total, 'reused_anchor_names': [{'name': n, 'count': anchors[n]} for n in sorted(anchors, key=sort_key) if anchors[n] > 1], 'unresolved_aliases': [{'name': n, 'count': unresolved[n]} for n in sorted(unresolved, key=sort_key)], 'examples': examples}


def _frontmatter(unit: Any) -> str:
    lines=str(get(unit,'content') or '').splitlines()
    if not lines or lines[0].strip() != '---': return ''
    body=[]
    for line in lines[1:]:
        if line.strip() in {'---','...'}: return '\n'.join(body)
        body.append(line)
    return ''
