"""Summarize external link domains found in unit content."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlparse

from graph.export._report_csv import get, sort_key, unit_id

_MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_BARE_URL_RE = re.compile(r"https?://[^\s<>()\[\]\"']+", re.IGNORECASE)


def summarize_unit_external_link_domains(units: Iterable[Mapping[str, Any] | object], *, sample_limit: int = 5) -> dict[str, Any]:
    """Group external http(s) links in unit content by normalized domain."""

    limit = max(0, sample_limit)
    unit_list = list(units)
    domain_counts: Counter[str] = Counter()
    linked_unit_ids: set[str] = set()
    samples: list[dict[str, str]] = []

    for index, unit in enumerate(unit_list):
        uid = unit_id(unit) or str(index)
        unit_linked = False
        for url, domain in _content_links(str(get(unit, "content") or "")):
            domain_counts[domain] += 1
            unit_linked = True
            if len(samples) < limit:
                samples.append({"unit_id": uid, "url": url, "domain": domain})
        if unit_linked:
            linked_unit_ids.add(uid)

    return {
        "unit_count": len(unit_list),
        "linked_unit_count": len(linked_unit_ids),
        "external_link_count": sum(domain_counts.values()),
        "domain_counts": {domain: domain_counts[domain] for domain in sorted(domain_counts, key=lambda value: (-domain_counts[value], sort_key(value)))},
        "samples": samples,
    }


def _content_links(content: str) -> list[tuple[str, str]]:
    links: list[tuple[str, str]] = []
    bare_content = content
    for match in _MARKDOWN_LINK_RE.finditer(content):
        url = _clean(match.group(1))
        domain = _domain(url)
        if domain:
            links.append((url, domain))
        bare_content = bare_content.replace(match.group(0), " ")
    for match in _BARE_URL_RE.finditer(bare_content):
        url = _clean(match.group(0))
        domain = _domain(url)
        if domain:
            links.append((url, domain))
    return links


def _clean(url: str) -> str:
    return url.rstrip(".,;:!?\"]}'")


def _domain(url: str) -> str:
    try:
        parsed = urlparse(url)
    except ValueError:
        return ""
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
        return ""
    hostname = parsed.hostname.casefold()
    return hostname[4:] if hostname.startswith("www.") else hostname
