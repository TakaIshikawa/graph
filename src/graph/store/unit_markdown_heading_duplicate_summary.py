"""Summarize duplicate markdown headings within units."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get, metadata, sort_key, unit_id

_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$", re.MULTILINE)
_SPACE_RE = re.compile(r"\s+")


def summarize_unit_markdown_heading_duplicates(units: Iterable[Any]) -> dict[str, Any]:
    total_units = units_with_duplicate_headings = duplicate_heading_count = 0
    rows: list[dict[str, Any]] = []
    for index, unit in enumerate(units):
        total_units += 1
        headings = [_normalize(match.group(2)) for match in _HEADING_RE.finditer(str(get(unit, "content") or metadata(unit).get("content") or ""))]
        duplicates = [{"heading": heading, "count": count} for heading, count in Counter(headings).items() if count > 1]
        duplicates.sort(key=lambda row: (-row["count"], sort_key(row["heading"])))
        if duplicates:
            units_with_duplicate_headings += 1
            duplicate_heading_count += len(duplicates)
            rows.append({"unit_id": unit_id(unit) or str(index), "duplicates": duplicates})
    return {"total_units": total_units, "units_with_duplicate_headings": units_with_duplicate_headings, "duplicate_heading_count": duplicate_heading_count, "units": sorted(rows, key=lambda row: sort_key(row["unit_id"]))}


def _normalize(text: str) -> str:
    return _SPACE_RE.sub(" ", text.strip().casefold())
