"""Summarize broken internal Markdown links between units."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import unquote, urlparse

from graph.export._report_csv import field_value, get, metadata, sort_key, unit_id

_LINK_RE = re.compile(r"(?<!!)\[[^\]\n]+\]\(([^)\n]+)\)")
_WIKILINK_RE = re.compile(r"(?<!!)\[\[([^\[\]\n]+)\]\]")
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_NON_SLUG_RE = re.compile(r"[^a-z0-9 -]+")
_SPACE_RE = re.compile(r"\s+")


def summarize_unit_broken_internal_links(units: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    """Report internal Markdown link targets that do not resolve to available units."""
    unit_list = list(units)
    available = _available_targets(unit_list)
    missing_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    examples: list[dict[str, str | int]] = []
    total_links = 0
    for index, unit in enumerate(unit_list):
        uid = unit_id(unit) or str(index)
        for line_number, target in _links(_content(unit)):
            normalized = _normalize_target(target)
            if not normalized:
                continue
            total_links += 1
            if normalized in available:
                continue
            missing_counts[normalized] += 1
            source_counts[uid] += 1
            if len(examples) < sample_limit:
                examples.append({"unit_id": uid, "line_number": line_number, "target": normalized})
    return {
        "total_units": len(unit_list),
        "internal_link_count": total_links,
        "broken_link_count": sum(missing_counts.values()),
        "missing_targets": [{"target": target, "count": missing_counts[target]} for target in sorted(missing_counts, key=sort_key)],
        "source_unit_counts": [{"unit_id": uid, "count": source_counts[uid]} for uid in sorted(source_counts, key=sort_key)],
        "examples": sorted(examples, key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["target"]))),
    }


def _available_targets(units: list[Mapping[str, Any] | object]) -> set[str]:
    targets: set[str] = set()
    for unit in units:
        data = metadata(unit)
        for value in (unit_id(unit), get(unit, "slug"), data.get("slug"), get(unit, "title"), data.get("title")):
            text = field_value(value)
            if text:
                targets.add(text)
                targets.add(_slug(text))
    return targets


def _content(unit: Mapping[str, Any] | object) -> str:
    return str(get(unit, "content") or metadata(unit).get("content") or "")


def _links(content: str) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        rows.extend((line_number, match.group(1)) for match in _LINK_RE.finditer(line))
        rows.extend((line_number, match.group(1)) for match in _WIKILINK_RE.finditer(line))
    return rows


def _normalize_target(target: str) -> str:
    target = unquote(field_value(target).strip("<>").split("|", 1)[0].split("#", 1)[0])
    parsed = urlparse(target)
    if parsed.scheme or target.startswith(("#", "/")):
        return ""
    return target


def _slug(text: str) -> str:
    return _SPACE_RE.sub("-", _NON_SLUG_RE.sub("", text.casefold()).strip())
