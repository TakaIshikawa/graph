"""Summarize CommonMark angle autolinks in unit content."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlparse

from graph.export._report_csv import get, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_ANGLE_RE = re.compile(r"<([^<>\s]+)>")
_EMAIL_RE = re.compile(r"^[^@\s<>]+@[^@\s<>]+\.[^@\s<>]+$")


def summarize_unit_markdown_autolinks(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    total = units_with = autolinks = urls = emails = 0
    schemes: Counter[str] = Counter()
    domains: Counter[str] = Counter()
    examples: list[dict[str, str | int]] = []
    for index, unit in enumerate(units):
        total += 1
        uid = unit_id(unit) or str(index)
        found = False
        for line_no, line in _content_lines(unit):
            for match in _ANGLE_RE.finditer(line):
                target = match.group(1)
                kind = _kind(target)
                if not kind:
                    continue
                found = True
                autolinks += 1
                if kind == "email":
                    emails += 1
                    parsed = urlparse(target)
                    if parsed.scheme:
                        schemes[parsed.scheme.casefold()] += 1
                    domain = target.rsplit("@", 1)[1].casefold()
                else:
                    urls += 1
                    parsed = urlparse(target)
                    schemes[parsed.scheme.casefold()] += 1
                    domain = (parsed.hostname or "").casefold()
                if domain:
                    domains[domain] += 1
                if len(examples) < sample_limit:
                    examples.append({"unit_id": uid, "line": line_no, "target": target, "kind": kind})
        if found:
            units_with += 1
    return {"total_units": total, "units_with_autolinks": units_with, "total_autolinks": autolinks, "url_autolink_count": urls, "email_autolink_count": emails, "scheme_counts": dict(sorted(schemes.items())), "domain_counts": dict(sorted(domains.items())), "examples": examples}


def _kind(target: str) -> str:
    parsed = urlparse(target)
    if parsed.scheme and (parsed.netloc or parsed.scheme.casefold() == "mailto"):
        return "email" if parsed.scheme.casefold() == "mailto" else "url"
    return "email" if _EMAIL_RE.match(target) else ""


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
