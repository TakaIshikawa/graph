"""Summarize numeric prefixes in ATX Markdown headings."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import get, metadata, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(?P<title>.+?)\s*#*\s*$")
_NUMBER_RE = re.compile(r"^(?P<number>\d+(?:\.\d+)*)(?:\.|\s+)")


def summarize_unit_markdown_heading_numbering(units: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    """Summarize numbered ATX headings and simple numbering issues."""
    unit_list = list(units)
    total_headings = numbered_headings = 0
    units_with: set[str] = set()
    depth_counts: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    repeated: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for index, unit in enumerate(unit_list):
        uid = unit_id(unit) or str(index)
        seen: set[str] = set()
        expected_top = 1
        for line_number, title in _headings(_content(unit)):
            total_headings += 1
            match = _NUMBER_RE.match(title)
            if not match:
                continue
            number = match.group("number")
            numbered_headings += 1
            units_with.add(uid)
            parts = number.split(".")
            depth_counts[str(len(parts))] += 1
            samples.append({"unit_id": uid, "line_number": line_number, "number": number, "heading": title})
            if number in seen:
                repeated.append({"unit_id": uid, "line_number": line_number, "number": number, "heading": title})
            seen.add(number)
            if len(parts) == 1:
                value = int(parts[0])
                if value > expected_top:
                    skipped.append({"unit_id": uid, "line_number": line_number, "expected": str(expected_top), "actual": number, "heading": title})
                expected_top = max(expected_top, value + 1)
    key = lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row.get("number", row.get("actual", ""))))
    return {
        "total_units": len(unit_list),
        "total_headings": total_headings,
        "numbered_headings": numbered_headings,
        "units_with_numbered_headings": len(units_with),
        "numbering_depth_counts": dict(sorted(depth_counts.items(), key=lambda item: sort_key(item[0]))),
        "repeated_number_samples": sorted(repeated, key=key)[:sample_limit],
        "skipped_sequence_samples": sorted(skipped, key=key)[:sample_limit],
        "samples": sorted(samples, key=key)[:sample_limit],
    }


def _content(unit: Mapping[str, Any] | object) -> str:
    return str(get(unit, "content") or metadata(unit).get("content") or "")


def _headings(content: str) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = _HEADING_RE.match(line)
        if match:
            rows.append((line_number, match.group("title").strip()))
    return rows
