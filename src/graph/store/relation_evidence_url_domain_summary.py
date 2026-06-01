"""Summarize domains used by relation evidence URLs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlparse

from graph.export._report_csv import field_value, get, metadata, sort_key

_EVIDENCE_KEYS = ("evidence", "evidence_urls", "supporting_evidence")
_URL_KEYS = ("url", "source_url", "href", "link")
_ID_KEYS = ("id", "relation_id", "edge_id")
_FALLBACK_KEYS = ("source", "source_id", "target", "target_id", "type", "relation_type", "predicate")


def summarize_relation_evidence_url_domains(relations: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    domain_counts: Counter[str] = Counter()
    relations_with_evidence_urls = 0
    relations_without_evidence_urls = 0
    samples: list[dict[str, Any]] = []

    for relation in relations:
        urls = _urls(relation)
        if urls:
            relations_with_evidence_urls += 1
        else:
            relations_without_evidence_urls += 1
            continue
        domains = sorted({_domain(url) for url in urls if _domain(url)}, key=sort_key)
        domain_counts.update(domains)
        if domains and len(samples) < limit:
            samples.append({"relation_id": _relation_id(relation), "domains": domains})

    return {
        "domain_counts": {key: domain_counts[key] for key in sorted(domain_counts, key=sort_key)},
        "relations_with_evidence_urls": relations_with_evidence_urls,
        "relations_without_evidence_urls": relations_without_evidence_urls,
        "external_domain_count": len(domain_counts),
        "samples": sorted(samples, key=lambda row: sort_key(row["relation_id"])),
    }


def _urls(relation: Any) -> list[str]:
    meta = metadata(relation)
    values: list[Any] = []
    for key in _EVIDENCE_KEYS + _URL_KEYS:
        raw = get(relation, key)
        if raw not in (None, ""):
            values.extend(_as_list(raw))
        raw = meta.get(key)
        if raw not in (None, ""):
            values.extend(_as_list(raw))
    urls: list[str] = []
    for value in values:
        if isinstance(value, Mapping):
            for key in _URL_KEYS:
                text = field_value(value.get(key))
                if text:
                    urls.append(text)
        else:
            text = field_value(value)
            if text:
                urls.append(text)
    return urls


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list | tuple | set) else [value]


def _domain(url: str) -> str:
    parsed = urlparse(url if "://" in url else f"https://{url}")
    host = (parsed.hostname or "").casefold()
    return host[4:] if host.startswith("www.") else host


def _relation_id(relation: Any) -> str:
    meta = metadata(relation)
    for key in _ID_KEYS:
        value = field_value(get(relation, key)) or field_value(meta.get(key))
        if value:
            return value
    parts = [field_value(get(relation, key)) or field_value(meta.get(key)) for key in _FALLBACK_KEYS]
    return "|".join(part for part in parts if part)
