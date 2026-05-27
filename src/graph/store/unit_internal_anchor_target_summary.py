"""Summarize same-document markdown anchor target resolution."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get, metadata, sort_key, unit_id

_LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(#([^)]+)\)")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)
_CUSTOM_ID_RE = re.compile(r"\{#([A-Za-z0-9_.:-]+)\}")
_HTML_ID_RE = re.compile(r"\bid=[\"']([^\"']+)[\"']", re.IGNORECASE)
_NON_SLUG_RE = re.compile(r"[^a-z0-9 -]+")
_SPACE_RE = re.compile(r"\s+")


def summarize_unit_internal_anchor_targets(units: Iterable[Any], *, sample_limit: int = 5) -> dict[str, Any]:
    total_units = resolved_count = unresolved_count = duplicate_target_count = 0
    missing: list[dict[str, str]] = []
    duplicates: list[dict[str, Any]] = []
    for index, unit in enumerate(units):
        total_units += 1
        uid = unit_id(unit) or str(index)
        content = str(get(unit, "content") or metadata(unit).get("content") or "")
        targets = _targets(content)
        counts = Counter(targets)
        for target, count in sorted(counts.items(), key=lambda item: sort_key(item[0])):
            if count > 1:
                duplicate_target_count += 1
                duplicates.append({"unit_id": uid, "target": target, "count": count})
        available = set(targets)
        for fragment in _links(content):
            if fragment in available:
                resolved_count += 1
            else:
                unresolved_count += 1
                if len(missing) < sample_limit:
                    missing.append({"unit_id": uid, "fragment": fragment})
    return {
        "total_units": total_units,
        "resolved_count": resolved_count,
        "unresolved_count": unresolved_count,
        "duplicate_target_count": duplicate_target_count,
        "missing_fragment_samples": sorted(missing, key=lambda row: (sort_key(row["unit_id"]), sort_key(row["fragment"]))),
        "duplicate_target_samples": sorted(duplicates, key=lambda row: (sort_key(row["unit_id"]), sort_key(row["target"])))[:sample_limit],
    }


def _targets(content: str) -> list[str]:
    explicit = _CUSTOM_ID_RE.findall(content)
    html = _HTML_ID_RE.findall(content)
    headings = []
    for match in _HEADING_RE.finditer(content):
        if _CUSTOM_ID_RE.search(match.group(1)):
            continue
        headings.append(_slug(match.group(1)))
    return [target for target in [*headings, *explicit, *html] if target]


def _links(content: str) -> list[str]:
    return [match.group(1).strip() for match in _LINK_RE.finditer(content)]


def _slug(text: str) -> str:
    cleaned = _NON_SLUG_RE.sub("", text.casefold())
    return _SPACE_RE.sub("-", cleaned.strip())
