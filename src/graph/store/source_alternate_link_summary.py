"""Summarize alternate links in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from html.parser import HTMLParser
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_LINK_KEYS = ("alternate_links", "alternates", "hreflang_links")
_HTML_KEYS = ("html", "content", "text", "snippet", "description")


def summarize_source_alternate_links(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows = [_row(source, index) for index, source in enumerate(source_list)]
    present = [row for row in rows if row["alternates"]]
    limit = max(0, sample_limit)
    samples = [
        {"source_id": row["source_id"], "alternates": row["alternates"][:limit]}
        for row in sorted(present, key=lambda row: sort_key(row["source_id"]))[:limit]
    ]
    return {
        "total_sources": len(source_list),
        "sources_with_alternate_links": len(present),
        "total_alternate_links": sum(len(row["alternates"]) for row in present),
        "hreflang_counts": dict(sorted(Counter(link["hreflang"] for row in present for link in row["alternates"] if link["hreflang"]).items())),
        "media_type_counts": dict(sorted(Counter(link["type"] for row in present for link in row["alternates"] if link["type"]).items())),
        "missing_alternate_link_count": len(source_list) - len(present),
        "samples": samples,
    }


def _row(source: Mapping[str, Any] | object, index: int) -> dict[str, Any]:
    return {"source_id": source_id(source) or str(index), "alternates": _alternate_links(source)}


def _alternate_links(source: Mapping[str, Any] | object) -> list[dict[str, str]]:
    links: list[dict[str, str]] = []
    data = metadata(source)
    for key in _LINK_KEYS:
        links.extend(_links_from_value(get(source, key)))
        links.extend(_links_from_value(data.get(key)))
    for html in _html_values(source, data):
        links.extend(_parse_html_links(html))
    return _dedupe_links(links)


def _links_from_value(value: object) -> list[dict[str, str]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        if any(key in value for key in ("href", "url", "hreflang", "lang", "type", "media_type")):
            return [_normalize_link(value)]
        return [
            _normalize_link({"hreflang": lang, "href": href})
            for lang, href in value.items()
            if field_value(href)
        ]
    if isinstance(value, list | tuple | set):
        return [link for item in value for link in _links_from_value(item)]
    text = field_value(value)
    return [{"href": text, "hreflang": "", "type": ""}] if text else []


def _normalize_link(value: Mapping[str, Any]) -> dict[str, str]:
    return {
        "href": field_value(value.get("href") or value.get("url")),
        "hreflang": _normalize_hreflang(value.get("hreflang") or value.get("lang") or value.get("language")),
        "type": _normalize_media_type(value.get("type") or value.get("media_type") or value.get("media")),
    }


def _html_values(source: Mapping[str, Any] | object, data: Mapping[str, Any]) -> list[str]:
    values = []
    for container in (source, data):
        for key in _HTML_KEYS:
            value = field_value(get(container, key) if container is source else container.get(key))
            if value:
                values.append(value)
    return values


def _parse_html_links(html: str) -> list[dict[str, str]]:
    parser = _AlternateLinkParser()
    parser.feed(html)
    return parser.links


class _AlternateLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() != "link":
            return
        attr_map = {name.casefold(): field_value(value) for name, value in attrs}
        rels = {part.casefold() for part in attr_map.get("rel", "").split()}
        if "alternate" not in rels:
            return
        link = _normalize_link(attr_map)
        if link["href"]:
            self.links.append(link)


def _dedupe_links(links: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str]] = set()
    result = []
    for link in links:
        if not link["href"]:
            continue
        key = (link["href"], link["hreflang"], link["type"])
        if key not in seen:
            seen.add(key)
            result.append(link)
    return sorted(result, key=lambda link: (sort_key(link["href"]), sort_key(link["hreflang"]), sort_key(link["type"])))


def _normalize_hreflang(value: object) -> str:
    return field_value(value).casefold().replace("_", "-")


def _normalize_media_type(value: object) -> str:
    return field_value(value).casefold()
