"""Summarize domains from source URL values."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlparse

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id


def summarize_source_url_domains(sources: Iterable[Any], *, sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total_sources = missing_url_count = invalid_url_count = schemeless_url_count = 0
    domain_counts: Counter[str] = Counter()
    scheme_counts: Counter[str] = Counter()
    samples: dict[str, list[dict[str, str]]] = defaultdict(list)
    for index, source in enumerate(sources):
        total_sources += 1
        url = _url(source)
        if not url:
            missing_url_count += 1
            continue
        parsed = urlparse(url)
        if not parsed.scheme:
            schemeless_url_count += 1
            parsed = urlparse(f"https://{url}")
        if not parsed.hostname or any(char.isspace() for char in parsed.hostname):
            invalid_url_count += 1
            continue
        domain = parsed.hostname.casefold()
        scheme = parsed.scheme.casefold() or "(none)"
        domain_counts[domain] += 1
        scheme_counts[scheme] += 1
        if len(samples[domain]) < limit:
            samples[domain].append({"source_id": source_id(source) or str(index), "title": _title(source)})
    rows = [
        {"domain": domain, "count": domain_counts[domain], "samples": samples[domain]}
        for domain in sorted(domain_counts, key=lambda item: (-domain_counts[item], sort_key(item)))
    ]
    return {
        "total_sources": total_sources,
        "domains": rows,
        "schemes": [{"scheme": scheme, "count": count} for scheme, count in sorted(scheme_counts.items(), key=lambda item: (-item[1], sort_key(item[0])))],
        "missing_url_count": missing_url_count,
        "invalid_url_count": invalid_url_count,
        "schemeless_url_count": schemeless_url_count,
    }


def _url(source: Any) -> str:
    meta = metadata(source)
    return field_value(get(source, "url") or get(source, "source_url") or meta.get("url") or meta.get("source_url"))


def _title(source: Any) -> str:
    return field_value(get(source, "title") or metadata(source).get("title"))
