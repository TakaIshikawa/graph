"""Summarize Markdown links with URL fragments."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlparse

from graph.export._report_csv import get, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)")


def summarize_unit_markdown_link_fragments(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    total_units = total_links = internal = external = 0
    fragments: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    for unit in units:
        total_units += 1
        uid = unit_id(unit)
        for line_number, target in _targets(str(get(unit, "content") or "")):
            parsed = urlparse(target)
            if not parsed.fragment:
                continue
            total_links += 1
            fragments[parsed.fragment] += 1
            if parsed.scheme or parsed.netloc:
                external += 1
            else:
                internal += 1
            if len(samples) < sample_limit:
                samples.append({"unit_id": uid, "line": line_number, "target": target, "fragment": parsed.fragment})
    top = [{"fragment": name, "count": fragments[name]} for name in sorted(fragments, key=lambda item: (-fragments[item], sort_key(item)))[:sample_limit]]
    return {"total_units": total_units, "total_fragment_links": total_links, "internal_fragment_count": internal, "external_fragment_count": external, "top_fragments": top, "sample_unit_references": samples}


def _targets(content: str) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        rows.extend((line_number, match.group(1)) for match in _LINK_RE.finditer(line))
    return rows
