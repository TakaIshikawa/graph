"""Summarize Open Graph metadata in sources."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_COMMON = ("og:title", "og:type", "og:image", "og:url")
_META_RE = re.compile(r"<meta\b(?P<attrs>[^>]*)>", re.IGNORECASE)
_ATTR_RE = re.compile(r"""\s([A-Za-z_:][-A-Za-z0-9_:.]*)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?""")


def summarize_source_open_graph_meta(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    limit = max(0, sample_limit)
    property_counts: Counter[str] = Counter()
    common_missing: Counter[str] = Counter()
    samples: list[dict[str, str]] = []
    sources_with = 0

    for index, source in enumerate(source_list):
        sid = source_id(source) or str(index)
        properties = _open_graph_properties(source)
        if not properties:
            for prop in _COMMON:
                common_missing[prop] += 1
            continue
        sources_with += 1
        seen = {prop for prop, _content in properties}
        property_counts.update(seen)
        for prop in _COMMON:
            if prop not in seen:
                common_missing[prop] += 1
        for prop, content in properties:
            if len(samples) < limit:
                samples.append({"source_id": sid, "property": prop, "content": content})

    return {
        "total_sources": len(source_list),
        "sources_with_open_graph": sources_with,
        "property_counts": {key: property_counts[key] for key in sorted(property_counts, key=sort_key)},
        "common_property_missing_counts": {key: common_missing[key] for key in _COMMON},
        "samples": samples,
    }


def _open_graph_properties(source: Mapping[str, Any] | object) -> list[tuple[str, str]]:
    data = metadata(source)
    rows: list[tuple[str, str]] = []
    for container in (data.get("open_graph"), data.get("og"), get(source, "open_graph")):
        if isinstance(container, Mapping):
            for key, value in container.items():
                prop = field_value(key).casefold()
                if prop and not prop.startswith("og:"):
                    prop = f"og:{prop}"
                if prop.startswith("og:"):
                    rows.append((prop, field_value(value)))
    for match in _META_RE.finditer(field_value(get(source, "html") or data.get("html") or data.get("content"))):
        attrs = {m.group(1).casefold(): field_value(next((part for part in m.groups()[1:] if part is not None), "")) for m in _ATTR_RE.finditer(match.group("attrs"))}
        prop = field_value(attrs.get("property") or attrs.get("name")).casefold()
        if prop.startswith("og:"):
            rows.append((prop, field_value(attrs.get("content"))))
    return rows
