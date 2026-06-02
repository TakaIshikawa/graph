"""Summarize rel=preload Link hints in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id


def summarize_source_preload_hints(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    as_counts: Counter[str] = Counter()
    missing_as = cross_origin = with_preload = 0
    samples: list[dict[str, str]] = []
    limit = max(0, sample_limit)
    for index, source in enumerate(source_list):
        sid = source_id(source) or str(index)
        links = [link for value in _lookup_headers(source, "link") for link in _split_links(value)]
        preload_links = [attrs for attrs in (_parse_link(link) for link in links) if "preload" in attrs.get("rel", "").casefold().split()]
        if preload_links:
            with_preload += 1
        for attrs in preload_links:
            as_value = attrs.get("as", "").casefold()
            if as_value:
                as_counts[as_value] += 1
            else:
                missing_as += 1
            if "crossorigin" in attrs:
                cross_origin += 1
            if len(samples) < limit:
                samples.append({"source_id": sid, "url": attrs.get("url", ""), "as": as_value, "crossorigin": attrs.get("crossorigin", "")})
    samples.sort(key=lambda row: (sort_key(row["source_id"]), sort_key(row["url"])))
    return {
        "total_sources": len(source_list),
        "sources_with_preload": with_preload,
        "as_counts": {key: as_counts[key] for key in sorted(as_counts, key=sort_key)},
        "missing_as_count": missing_as,
        "cross_origin_count": cross_origin,
        "samples": samples[:limit],
    }


def _split_links(value: str) -> list[str]:
    rows: list[str] = []
    current = ""
    in_quote = False
    for char in value:
        if char == '"':
            in_quote = not in_quote
        if char == "," and not in_quote:
            rows.append(current.strip())
            current = ""
        else:
            current += char
    if current.strip():
        rows.append(current.strip())
    return rows


def _parse_link(value: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for index, part in enumerate(part.strip() for part in value.split(";") if part.strip()):
        if index == 0 and part.startswith("<") and ">" in part:
            attrs["url"] = part[1 : part.index(">")]
            continue
        key, _, raw = part.partition("=")
        attrs[key.strip().casefold()] = raw.strip().strip('"')
    return attrs


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
