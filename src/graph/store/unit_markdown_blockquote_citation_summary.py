"""Summarize citation-like attribution lines inside Markdown blockquotes."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, sort_key, unit_id

_CITATION_RE = re.compile(r"^\s*(?:(?P<dash>--|-)\s+(?P<dash_text>.+)|(?P<emdash>—)\s*(?P<emdash_text>.+)|(?P<source>source:)\s*(?P<source_text>.+))$", re.IGNORECASE)


def summarize_unit_markdown_blockquote_citations(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total = units_with = citation_count = 0
    styles: Counter[str] = Counter()
    samples: list[dict[str, str | int]] = []
    rows: list[dict[str, Any]] = []
    for unit in units:
        total += 1
        uid = unit_id(unit)
        unit_count = 0
        unit_styles: Counter[str] = Counter()
        for line_number, text in _blockquote_lines(str(get(unit, "content") or "")):
            parsed = _citation(text)
            if not parsed:
                continue
            style, citation_text = parsed
            citation_count += 1
            unit_count += 1
            styles[style] += 1
            unit_styles[style] += 1
            if len(samples) < limit:
                samples.append({"unit_id": uid, "line_number": line_number, "style": style, "citation_text": citation_text})
        if unit_count:
            units_with += 1
            rows.append({"unit_id": uid, "citation_count": unit_count, "styles": {key: unit_styles[key] for key in sorted(unit_styles, key=sort_key)}})
    rows.sort(key=lambda row: sort_key(row["unit_id"]))
    return {
        "total_units": total,
        "units_with_citations": units_with,
        "citation_count": citation_count,
        "styles": {key: styles[key] for key in sorted(styles, key=sort_key)},
        "samples": samples[:limit],
        "units": rows,
    }


def _blockquote_lines(content: str) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith(">"):
            rows.append((line_number, stripped[1:].strip()))
    return rows


def _citation(text: str) -> tuple[str, str] | None:
    match = _CITATION_RE.match(text)
    if not match:
        return None
    if match.group("source"):
        return ("source", field_value(match.group("source_text")))
    if match.group("emdash"):
        return ("em_dash", field_value(match.group("emdash_text")))
    return ("dash", field_value(match.group("dash_text")))
