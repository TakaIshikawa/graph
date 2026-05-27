"""Summarize DOI-like identifiers in unit content and string metadata."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, unit_id

_DOI_RE = re.compile(r"(?i)(?:doi:\s*|https?://(?:dx\.)?doi\.org/)?\b(10\.\d{4,9}/[^\s<>\]\)\"']+)")
_TRAIL = ".,;:"


def summarize_unit_doi_hints(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    total_units=total_matches=0; units_with=set(); values=Counter(); sources=Counter(); examples=defaultdict(list)
    for unit in units:
        total_units += 1
        for source,text in _strings(unit):
            for doi in _dois(text):
                total_matches += 1; units_with.add(unit_id(unit)); values[doi] += 1; sources[source] += 1
                if len(examples[doi]) < sample_limit: examples[doi].append({'unit_id': unit_id(unit), 'source': source})
    return {'total_units': total_units, 'total_matches': total_matches, 'units_with_matches': len(units_with), 'source_field_counts': [{'source': s, 'count': sources[s]} for s in sorted(sources, key=sort_key)], 'doi_values': [{'doi': d, 'count': values[d], 'examples': examples[d]} for d in sorted(values, key=lambda d: (-values[d], d))]}


def _strings(unit: Any) -> list[tuple[str,str]]:
    vals=[('content', field_value(get(unit,'content')))]
    for k,v in metadata(unit).items():
        if isinstance(v,str): vals.append((f'metadata.{k}', v))
    items=unit.items() if isinstance(unit, Mapping) else ((k,getattr(unit,k)) for k in ('doi','url','source','title') if hasattr(unit,k))
    for k,v in items:
        if k not in {'content','metadata'} and isinstance(v,str): vals.append((str(k), v))
    return vals


def _dois(text: str) -> list[str]:
    return [m.group(1).rstrip(_TRAIL).casefold() for m in _DOI_RE.finditer(text)]
