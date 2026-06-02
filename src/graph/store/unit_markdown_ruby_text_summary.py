"""Summarize HTML ruby annotation usage in Markdown content."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, sort_key, unit_id

_RUBY_RE = re.compile(r"<ruby\b[^>]*>(?P<body>.*?)</ruby\s*>", re.IGNORECASE | re.DOTALL)
_RT_RE = re.compile(r"<rt\b[^>]*>(?P<text>.*?)</rt\s*>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")


def summarize_unit_markdown_ruby_text(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total = units_with = ruby_count = annotation_count = 0
    samples: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []
    for unit in units:
        total += 1
        uid = unit_id(unit)
        unit_ruby = unit_annotations = 0
        for base, annotations in _rubies(str(get(unit, "content") or "")):
            unit_ruby += 1
            unit_annotations += len(annotations)
            if len(samples) < limit:
                samples.append({"unit_id": uid, "base_text": base, "annotation_text": " | ".join(annotations)})
        if unit_ruby:
            units_with += 1
            ruby_count += unit_ruby
            annotation_count += unit_annotations
            rows.append({"unit_id": uid, "ruby_count": unit_ruby, "annotation_count": unit_annotations})
    samples.sort(key=lambda row: (sort_key(row["unit_id"]), sort_key(row["base_text"]), sort_key(row["annotation_text"])))
    rows.sort(key=lambda row: sort_key(row["unit_id"]))
    return {"total_units": total, "units_with_ruby": units_with, "ruby_count": ruby_count, "annotation_count": annotation_count, "samples": samples[:limit], "units": rows}


def _rubies(content: str) -> list[tuple[str, list[str]]]:
    rows: list[tuple[str, list[str]]] = []
    for match in _RUBY_RE.finditer(content):
        body = match.group("body")
        annotations = [field_value(rt.group("text")) for rt in _RT_RE.finditer(body)]
        annotations = [item for item in annotations if item]
        if not annotations:
            continue
        base = field_value(_TAG_RE.sub(" ", _RT_RE.sub(" ", body)))
        if base:
            rows.append((base, annotations))
    return rows
