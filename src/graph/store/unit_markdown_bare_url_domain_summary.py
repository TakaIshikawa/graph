"""Summarize bare URL domains in Markdown unit content."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlparse

from graph.export._report_csv import field_value, get, metadata, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_URL_RE = re.compile(r"https?://[^\s<>\]]+", re.IGNORECASE)
_AUTOLINK_RE = re.compile(r"<https?://[^<>\s]+>", re.IGNORECASE)
_INLINE_LINK_RE = re.compile(r"!?\[[^\]]*\]\([^)]*\)")
_TRAILING_PUNCTUATION = ".,!?;:"


def summarize_unit_markdown_bare_url_domains(units: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    """Summarize bare http/https URLs by normalized domain."""
    unit_list = list(units)
    bare_url_count = 0
    domains: Counter[str] = Counter()
    affected: set[str] = set()
    examples: list[dict[str, str | int]] = []
    for index, unit in enumerate(unit_list):
        uid = unit_id(unit) or str(index)
        for line_number, url, domain in _bare_urls(_content(unit)):
            bare_url_count += 1
            domains[domain] += 1
            affected.add(uid)
            examples.append({"unit_id": uid, "line_number": line_number, "domain": domain, "url": url})
    examples.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["domain"]), sort_key(row["url"])))
    return {
        "total_units": len(unit_list),
        "bare_url_count": bare_url_count,
        "domain_counts": dict(sorted(domains.items(), key=lambda item: sort_key(item[0]))),
        "affected_units": sorted(affected, key=sort_key),
        "examples": examples[:sample_limit],
    }


def _content(unit: Mapping[str, Any] | object) -> str:
    return str(get(unit, "content") or metadata(unit).get("content") or "")


def _bare_urls(content: str) -> list[tuple[int, str, str]]:
    rows: list[tuple[int, str, str]] = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        masked = _mask_markdown_urls(line)
        for match in _URL_RE.finditer(masked):
            url = _trim_url(match.group(0))
            domain = _domain(url)
            if domain:
                rows.append((line_number, field_value(url), domain))
    return rows


def _mask_markdown_urls(line: str) -> str:
    masked = line
    for pattern in (_INLINE_LINK_RE, _AUTOLINK_RE):
        masked = pattern.sub(lambda match: " " * (match.end() - match.start()), masked)
    return masked


def _trim_url(url: str) -> str:
    while url and url[-1] in _TRAILING_PUNCTUATION:
        url = url[:-1]
    while url.endswith(")") and url.count("(") < url.count(")"):
        url = url[:-1]
    return url


def _domain(url: str) -> str:
    parsed = urlparse(url)
    return (parsed.hostname or "").casefold()
