"""Summarize resource hint Link headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id
from graph.store.source_preload_hint_summary import _parse_link, _split_links

_RELATIONS = {"dns-prefetch", "preconnect", "prefetch", "prerender", "modulepreload"}


def summarize_source_resource_hints(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    relation_counts: Counter[str] = Counter()
    cross_origin = 0
    samples: list[dict[str, str]] = []
    limit = max(0, sample_limit)
    for index, source in enumerate(source_list):
        sid = source_id(source) or str(index)
        for value in _lookup_headers(source, "link"):
            for link in _split_links(value):
                attrs = _parse_link(link)
                for rel in attrs.get("rel", "").casefold().split():
                    if rel not in _RELATIONS:
                        continue
                    relation_counts[rel] += 1
                    cross_origin += int("crossorigin" in attrs)
                    if len(samples) < limit:
                        samples.append({"source_id": sid, "relation": rel, "url": attrs.get("url", "")})
    samples.sort(key=lambda row: (sort_key(row["source_id"]), sort_key(row["relation"]), sort_key(row["url"])))
    return {
        "total_sources": len(source_list),
        "relation_counts": {key: relation_counts[key] for key in sorted(relation_counts, key=sort_key)},
        "cross_origin_count": cross_origin,
        "samples": samples[:limit],
    }


def _lookup_headers(source: Mapping[str, Any] | object, header: str) -> list[str]:
    values: list[str] = []
    data = metadata(source)
    for container in (source, data, get(source, "headers"), get(source, "response_headers"), data.get("headers"), data.get("response_headers")):
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == header:
                    _append(values, value)
    return values


def _append(values: list[str], raw: Any) -> None:
    if isinstance(raw, list | tuple | set):
        for item in raw:
            _append(values, item)
        return
    value = field_value(raw)
    if value:
        values.append(value)
