"""Summarize Markdown table-of-contents entries in units."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, sort_key, unit_id
from graph.export.unit_markdown_toc_entry_csv import _TOC_RE

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(?P<text>.+?)\s*$")
_CUSTOM_ID_RE = re.compile(r"\{#([^}\s]+)}")


def summarize_unit_markdown_toc(units: Iterable[Any], sample_limit: int = 10) -> dict[str, Any]:
    total_units = units_with = toc_count = max_depth = 0
    unresolved: list[dict[str, str | int]] = []
    duplicate: list[dict[str, str | int]] = []
    for unit in units:
        total_units += 1
        content = str(get(unit, "content") or "")
        targets = _targets(content)
        entries = []
        for line_number, line in enumerate(content.splitlines(), start=1):
            match = _TOC_RE.match(line)
            if not match:
                continue
            fragment = field_value(match.group("fragment"))
            depth = len(match.group("indent").replace("\t", "    ")) // 2 + 1
            max_depth = max(max_depth, depth)
            row = {"unit_id": unit_id(unit), "fragment": fragment, "line_number": line_number}
            entries.append(row)
            if fragment.casefold() not in targets:
                unresolved.append(row)
        if entries:
            units_with += 1
            toc_count += len(entries)
            counts = Counter(row["fragment"].casefold() for row in entries)
            duplicate.extend(row for row in entries if counts[row["fragment"].casefold()] > 1)
    unresolved.sort(key=lambda row: (sort_key(row["unit_id"]), sort_key(row["fragment"]), int(row["line_number"])))
    duplicate.sort(key=lambda row: (sort_key(row["unit_id"]), sort_key(row["fragment"]), int(row["line_number"])))
    return {
        "total_units": total_units,
        "units_with_toc_entries": units_with,
        "toc_entry_count": toc_count,
        "max_toc_depth": max_depth,
        "unresolved_toc_fragment_samples": unresolved[:sample_limit],
        "duplicate_toc_fragment_samples": duplicate[:sample_limit],
    }


def _targets(content: str) -> set[str]:
    targets: set[str] = set()
    for line in content.splitlines():
        custom_ids = _CUSTOM_ID_RE.findall(line)
        targets.update(value.casefold() for value in custom_ids)
        match = _HEADING_RE.match(line)
        if match:
            text = _CUSTOM_ID_RE.sub("", match.group("text")).strip()
            slug = re.sub(r"[^A-Za-z0-9\s-]", "", text).strip().lower()
            slug = re.sub(r"\s+", "-", slug)
            if slug:
                targets.add(slug)
    return targets
