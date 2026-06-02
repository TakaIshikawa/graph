"""Summarize favicon hints in sources."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_KEYS = {"favicon_url": "icon", "icon_url": "icon", "apple_touch_icon": "apple-touch-icon"}
_CONTENT_KEYS = ("content", "html", "body", "text")
_LINK_RE = re.compile(r"<link\b(?P<attrs>[^>]*?)>", re.IGNORECASE | re.DOTALL)
_ATTR_RE = re.compile(r"""([:\w-]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+))""", re.IGNORECASE)
_RELATIONS = {"icon", "shortcut icon", "apple-touch-icon"}


def summarize_source_favicon_hints(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows = [_row(source, index) for index, source in enumerate(source_list)]
    present = [row for row in rows if row["icon_url"]]
    samples = [
        {"source_id": row["source_id"], "relation": row["relation"], "icon_url": row["icon_url"]}
        for row in sorted(present, key=lambda row: sort_key(row["source_id"]))[: max(0, sample_limit)]
    ]
    return {
        "total_sources": len(source_list),
        "sources_with_favicon_hint": len(present),
        "icon_relation_counts": dict(sorted(Counter(row["relation"] for row in present).items())),
        "external_icon_count": sum(1 for row in present if _host(row["icon_url"]) and _host(row["source_url"]) and _host(row["icon_url"]) != _host(row["source_url"])),
        "missing_favicon_hint_count": len(source_list) - len(present),
        "icon_extension_counts": dict(sorted(Counter(_extension(row["icon_url"]) for row in present if _extension(row["icon_url"])).items())),
        "samples": samples,
    }


def _row(source: Mapping[str, Any] | object, index: int) -> dict[str, str]:
    relation, icon_url = _lookup_value(source)
    return {"source_id": source_id(source) or str(index), "source_url": _source_url(source), "relation": relation, "icon_url": icon_url}


def _lookup_value(source: Mapping[str, Any] | object) -> tuple[str, str]:
    data = metadata(source)
    for container_name, container in (("source", source), ("metadata", data)):
        for key, relation in _KEYS.items():
            value = field_value(get(container, key) if container_name == "source" else container.get(key))
            if value:
                return relation, value
    for container in (source, data):
        for key in _CONTENT_KEYS:
            value = _extract_from_html(field_value(get(container, key) if container is source else container.get(key)))
            if value[1]:
                return value
    return "", ""


def _extract_from_html(html: str) -> tuple[str, str]:
    for match in _LINK_RE.finditer(html):
        attrs = {name.casefold(): unescape(value) for name, value in _attrs(match.group("attrs")).items()}
        rel = " ".join(field_value(attrs.get("rel")).casefold().split())
        if rel in _RELATIONS and field_value(attrs.get("href")):
            return rel, field_value(attrs.get("href"))
    return "", ""


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


def _extension(url: str) -> str:
    return Path(urlparse(field_value(url)).path).suffix.casefold().lstrip(".")
