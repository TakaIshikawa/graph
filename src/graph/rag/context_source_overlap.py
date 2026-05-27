"""Analyze source overlap across packed RAG context."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from graph.rag._analysis_utils import domain_for, result_id, string, value

_URL_KEYS = ("url", "source_url", "canonical_url", "external_url", "link", "permalink", "uri")
_TRACKING_PREFIXES = ("utm_",)
_TRACKING_KEYS = {"fbclid", "gclid", "mc_cid", "mc_eid", "ref", "referrer"}


def analyze_context_source_overlap(context_items: Iterable[Any]) -> dict[str, Any]:
    items = list(context_items or [])
    url_groups: dict[str, list[str]] = defaultdict(list)
    title_groups: dict[str, list[str]] = defaultdict(list)
    domain_counts: Counter[str] = Counter()

    for index, item in enumerate(items):
        item_id = result_id(item, index)
        url = _normalized_url(item)
        title = _normalized_title(item)
        domain = domain_for(item)
        if url:
            url_groups[url].append(item_id)
        if title:
            title_groups[title].append(item_id)
        if domain:
            domain_counts[domain] += 1

    overlap_groups = []
    for key, ids in sorted(url_groups.items()):
        if len(ids) > 1:
            overlap_groups.append({"type": "url", "value": key, "item_ids": sorted(ids), "count": len(ids)})
    for key, ids in sorted(title_groups.items()):
        if len(ids) > 1:
            overlap_groups.append({"type": "title", "value": key, "item_ids": sorted(ids), "count": len(ids)})

    repeated_ids = {item_id for group in overlap_groups for item_id in group["item_ids"]}
    redundancy_ratio = round(len(repeated_ids) / len(items), 4) if items else 0.0
    risk_level = "high" if redundancy_ratio >= 0.5 else "medium" if redundancy_ratio > 0 else "low"

    return {
        "overlap_groups": overlap_groups,
        "repeated_domain_counts": [{"domain": domain, "count": count} for domain, count in sorted(domain_counts.items()) if count > 1],
        "redundancy_ratio": redundancy_ratio,
        "risk_level": risk_level,
    }


def _normalized_url(item: Any) -> str:
    for key in _URL_KEYS:
        text = string(value(item, key))
        if not text:
            continue
        parsed = urlparse(text if "://" in text else f"https://{text}")
        query = urlencode(
            [(key_, val) for key_, val in parse_qsl(parsed.query, keep_blank_values=True) if not key_.startswith(_TRACKING_PREFIXES) and key_ not in _TRACKING_KEYS]
        )
        host = parsed.netloc.casefold()
        if host.startswith("www."):
            host = host[4:]
        path = parsed.path.rstrip("/")
        return urlunparse((parsed.scheme.casefold() or "https", host, path, "", query, ""))
    return ""


def _normalized_title(item: Any) -> str:
    return (string(value(item, "title")) or "").casefold()
