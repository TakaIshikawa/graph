"""Summarize canonical URL hints in sources."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from html import unescape
from typing import Any
from urllib.parse import urlparse

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_KEYS = ("canonical_url", "canonical", "rel_canonical")
_CONTENT_KEYS = ("content", "html", "body", "text")
_LINK_RE = re.compile(r"<link\b(?P<attrs>[^>]*?)>", re.IGNORECASE | re.DOTALL)
_ATTR_RE = re.compile(r"""([:\w-]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+))""", re.IGNORECASE)


def summarize_source_canonical_urls(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows = [_row(source, index) for index, source in enumerate(source_list)]
    present = [row for row in rows if row["canonical_url"]]
    counts = Counter(row["canonical_url"] for row in present)
    samples = [
        {"source_id": row["source_id"], "canonical_url": row["canonical_url"], "source_url": row["source_url"]}
        for row in sorted(present, key=lambda row: sort_key(row["source_id"]))[: max(0, sample_limit)]
    ]
    return {
        "total_sources": len(source_list),
        "sources_with_canonical_url": len(present),
        "matching_source_url_count": sum(1 for row in present if row["canonical_url"].casefold() == row["source_url"].casefold() and row["source_url"]),
        "external_canonical_count": sum(1 for row in present if _host(row["canonical_url"]) and _host(row["source_url"]) and _host(row["canonical_url"]) != _host(row["source_url"])),
        "missing_canonical_url_count": len(source_list) - len(present),
        "canonical_domain_counts": dict(sorted(Counter(_host(row["canonical_url"]) for row in present if _host(row["canonical_url"])).items())),
        "duplicate_canonical_url_count": sum(1 for count in counts.values() if count > 1),
        "samples": samples,
    }


def _row(source: Mapping[str, Any] | object, index: int) -> dict[str, str]:
    return {
        "source_id": source_id(source) or str(index),
        "source_url": _source_url(source),
        "canonical_url": _lookup_value(source),
    }


def _lookup_value(source: Mapping[str, Any] | object) -> str:
    data = metadata(source)
    for container_name, container in (("source", source), ("metadata", data)):
        for key in _KEYS:
            value = field_value(get(container, key) if container_name == "source" else container.get(key))
            if value:
                return value
    for container in (source, data):
        for key in _CONTENT_KEYS:
            value = _extract_from_html(field_value(get(container, key) if container is source else container.get(key)))
            if value:
                return value
    return ""


def _extract_from_html(html: str) -> str:
    for match in _LINK_RE.finditer(html):
        attrs = {name.casefold(): unescape(value) for name, value in _attrs(match.group("attrs")).items()}
        rels = {part.casefold() for part in field_value(attrs.get("rel")).split()}
        if "canonical" in rels and field_value(attrs.get("href")):
            return field_value(attrs.get("href"))
    return ""


def _attrs(text: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for match in _ATTR_RE.finditer(text):
        attrs[match.group(1)] = next(group for group in match.groups()[1:] if group is not None)
    return attrs


def _source_url(source: Mapping[str, Any] | object) -> str:
    data = metadata(source)
    return field_value(get(source, "url") or get(source, "source_url") or data.get("url") or data.get("source_url"))


def _host(url: str) -> str:
    return urlparse(field_value(url)).hostname.casefold() if urlparse(field_value(url)).hostname else ""
