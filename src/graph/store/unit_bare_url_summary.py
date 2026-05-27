"""Summarize bare HTTP URLs in unit content."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlparse

from graph.export._report_csv import get, unit_id

_URL_RE = re.compile(r"https?://[^\s<>\]\"']+", re.IGNORECASE)
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")


def summarize_unit_bare_urls(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    total = units_with = count = 0
    schemes: Counter[str] = Counter()
    domains: Counter[str] = Counter()
    examples: list[dict[str, str | int]] = []
    for index, unit in enumerate(units):
        total += 1
        uid = unit_id(unit) or str(index)
        found = False
        for line_no, line in _content_lines(unit):
            for match in _URL_RE.finditer(line):
                url = match.group(0).rstrip(".,;:!?)]}")
                clean_end = match.start() + len(url)
                if _is_markdown_url(line, match.start(), clean_end):
                    continue
                parsed = urlparse(url)
                domain = (parsed.hostname or "").casefold()
                scheme = parsed.scheme.casefold()
                if not domain or scheme not in {"http", "https"}:
                    continue
                found = True
                count += 1
                schemes[scheme] += 1
                domains[domain] += 1
                if len(examples) < sample_limit:
                    examples.append({"unit_id": uid, "line": line_no, "url": url, "domain": domain})
        if found:
            units_with += 1
    return {
        "total_units": total,
        "units_with_bare_urls": units_with,
        "total_bare_urls": count,
        "scheme_counts": dict(sorted(schemes.items())),
        "domain_counts": dict(sorted(domains.items())),
        "examples": examples,
    }


def _content_lines(unit: Any) -> list[tuple[int, str]]:
    in_fence = False
    rows = []
    for line_no, line in enumerate(str(get(unit, "content") or "").splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            rows.append((line_no, line))
    return rows


def _is_markdown_url(line: str, start: int, end: int) -> bool:
    before = line[:start]
    after = line[end:]
    return (before.endswith("(") and after.startswith(")")) or (before.endswith("<") and after.startswith(">"))
