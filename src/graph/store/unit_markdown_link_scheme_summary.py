"""Summarize destination schemes for inline markdown links."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlparse

from graph.export._report_csv import get, metadata, sort_key, unit_id

_LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]*)\)")


def summarize_unit_markdown_link_schemes(units: Iterable[Any]) -> dict[str, Any]:
    total_units = link_count = 0
    schemes: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for index, unit in enumerate(units):
        total_units += 1
        counts: Counter[str] = Counter()
        for match in _LINK_RE.finditer(str(get(unit, "content") or metadata(unit).get("content") or "")):
            scheme = _scheme(match.group(1).strip())
            counts[scheme] += 1
            schemes[scheme] += 1
            link_count += 1
        if counts:
            rows.append({"unit_id": unit_id(unit) or str(index), "scheme_counts": _rows(counts, "scheme")})
    return {"total_units": total_units, "link_count": link_count, "scheme_counts": _rows(schemes, "scheme"), "units": sorted(rows, key=lambda row: sort_key(row["unit_id"]))}


def _scheme(target: str) -> str:
    if not target:
        return "unknown"
    if target.startswith("#"):
        return "anchor"
    parsed = urlparse(target)
    if parsed.scheme:
        return parsed.scheme.casefold()
    return "relative"


def _rows(counter: Counter[str], key: str) -> list[dict[str, Any]]:
    return [{key: name, "count": counter[name]} for name in sorted(counter, key=sort_key)]
