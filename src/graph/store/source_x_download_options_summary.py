"""Summarize X-Download-Options headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id


def summarize_source_x_download_options(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    others: Counter[str] = Counter()
    noopen = with_header = 0
    samples: list[dict[str, str]] = []
    limit = max(0, sample_limit)
    for index, source in enumerate(source_list):
        sid = source_id(source) or str(index)
        value = _lookup_header(source, "x-download-options").casefold()
        if not value:
            continue
        with_header += 1
        if value == "noopen":
            noopen += 1
        else:
            others[value] += 1
        if len(samples) < limit:
            samples.append({"source_id": sid, "value": value})
    samples.sort(key=lambda row: sort_key(row["source_id"]))
    return {
        "total_sources": len(source_list),
        "sources_with_x_download_options": with_header,
        "noopen_count": noopen,
        "other_value_counts": {key: others[key] for key in sorted(others, key=sort_key)},
        "missing_x_download_options_count": len(source_list) - with_header,
        "samples": samples[:limit],
    }


def _lookup_header(source: Mapping[str, Any] | object, header: str) -> str:
    data = metadata(source)
    for container in (source, data, get(source, "headers"), get(source, "response_headers"), data.get("headers"), data.get("response_headers")):
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == header:
                    return field_value(value)
    return ""
