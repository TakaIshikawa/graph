"""Summarize meta robots directives in sources."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from html import unescape
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_KEYS = ("meta_robots", "robots_meta", "html_meta_robots")
_CONTENT_KEYS = ("content", "html", "body", "text")
_META_RE = re.compile(r"<meta\b(?P<attrs>[^>]*?)>", re.IGNORECASE | re.DOTALL)
_ATTR_RE = re.compile(r"""([:\w-]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+))""", re.IGNORECASE)


def summarize_source_meta_robots(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows = [_row(source, index) for index, source in enumerate(source_list)]
    present = [row for row in rows if row["value"]]
    directive_counts = Counter(directive for row in present for directive in row["directives"])
    samples = [
        {"source_id": row["source_id"], "directives": row["directives"], "value": row["value"]}
        for row in sorted(present, key=lambda row: sort_key(row["source_id"]))[: max(0, sample_limit)]
    ]
    return {
        "total_sources": len(source_list),
        "sources_with_meta_robots": len(present),
        "directive_counts": dict(sorted(directive_counts.items())),
        "noindex_count": sum(1 for row in present if "noindex" in row["directives"]),
        "nofollow_count": sum(1 for row in present if "nofollow" in row["directives"]),
        "noarchive_count": sum(1 for row in present if "noarchive" in row["directives"]),
        "missing_meta_robots_count": len(source_list) - len(present),
        "samples": samples,
    }


def _row(source: Mapping[str, Any] | object, index: int) -> dict[str, Any]:
    value = _lookup_value(source)
    return {"source_id": source_id(source) or str(index), "value": value, "directives": _directives(value)}


def _directives(value: object) -> list[str]:
    directives: list[str] = []
    for part in re.split(r"[,;]", field_value(value)):
        directive = part.strip().casefold()
        if directive:
            directives.append(directive)
    return directives


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
    for match in _META_RE.finditer(html):
        attrs = {name.casefold(): unescape(value) for name, value in _attrs(match.group("attrs")).items()}
        if attrs.get("name", "").casefold() == "robots" and field_value(attrs.get("content")):
            return field_value(attrs.get("content"))
    return ""


def _attrs(text: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for match in _ATTR_RE.finditer(text):
        attrs[match.group(1)] = next(group for group in match.groups()[1:] if group is not None)
    return attrs
