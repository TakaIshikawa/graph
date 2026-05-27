"""Summarize external URL hostnames found on units."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlparse

from graph.export._report_csv import field_value, flatten_values, get, metadata, sort_key, unit_id

_URL_RE = re.compile(r"https?://[^\s<>()\[\]\"']+", re.IGNORECASE)


def summarize_unit_external_url_domains(units: Iterable[Any], *, sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total = 0
    url_counts: Counter[str] = Counter()
    unit_ids_by_host: dict[str, set[str]] = defaultdict(set)
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict[str, str]]] = defaultdict(list)

    for index, unit in enumerate(units):
        total += 1
        uid = unit_id(unit) or str(index)
        for source, url, host in _unit_urls(unit):
            url_counts[host] += 1
            unit_ids_by_host[host].add(uid)
            source_counts[host][source] += 1
            if len(examples[host]) < limit:
                examples[host].append({"unit_id": uid, "url": url})

    rows = []
    for host in sorted(url_counts, key=lambda item: (-url_counts[item], sort_key(item))):
        rows.append(
            {
                "hostname": host,
                "unit_count": len(unit_ids_by_host[host]),
                "url_count": url_counts[host],
                "source_counts": [
                    {"source": source, "count": source_counts[host][source]}
                    for source in sorted(source_counts[host], key=sort_key)
                ],
                "examples": examples[host],
            }
        )
    return {"total_units": total, "domains": rows}


def _unit_urls(unit: Any) -> list[tuple[str, str, str]]:
    urls: list[tuple[str, str, str]] = []
    for url in _extract_urls("" if get(unit, "content") is None else str(get(unit, "content"))):
        host = _hostname(url)
        if host:
            urls.append(("content", url, host))
    for value in flatten_values(metadata(unit)):
        for url in _extract_urls(field_value(value)):
            host = _hostname(url)
            if host:
                urls.append(("metadata", url, host))
    return urls


def _extract_urls(text: str) -> list[str]:
    return [_clean(match.group(0)) for match in _URL_RE.finditer(text)]


def _clean(url: str) -> str:
    return url.rstrip(".,;:!?)\"]}'")


def _hostname(url: str) -> str:
    try:
        parsed = urlparse(url)
    except ValueError:
        return ""
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
        return ""
    return parsed.hostname.casefold()
