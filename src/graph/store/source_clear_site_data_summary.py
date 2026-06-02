"""Summarize Clear-Site-Data headers in sources."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "clear-site-data"
_QUOTED_RE = re.compile(r'"([^"]+)"')


def summarize_source_clear_site_data_headers(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows = [_row(source, index) for index, source in enumerate(source_list)]
    present = [row for row in rows if row["value"]]
    samples_source = [row for row in present if row["wildcard"] or row["malformed"]]
    samples = [
        {"source_id": row["source_id"], "value": row["value"], "directives": row["directives"], "malformed": row["malformed"]}
        for row in sorted(samples_source, key=lambda row: sort_key(row["source_id"]))[: max(0, sample_limit)]
    ]
    return {
        "total_sources": len(source_list),
        "sources_with_clear_site_data": len(present),
        "directive_counts": dict(sorted(Counter(directive for row in present for directive in row["directives"]).items())),
        "wildcard_count": sum(1 for row in present if row["wildcard"]),
        "malformed_count": sum(1 for row in present if row["malformed"]),
        "missing_header_count": len(source_list) - len(present),
        "samples": samples,
    }


def _row(source: Mapping[str, Any] | object, index: int) -> dict[str, Any]:
    value = _lookup_header(source, _HEADER)
    directives = [directive.casefold() for directive in _QUOTED_RE.findall(value)]
    return {
        "source_id": source_id(source) or str(index),
        "value": value,
        "directives": directives,
        "wildcard": "*" in directives,
        "malformed": bool(value) and not directives,
    }


def _lookup_header(source: Mapping[str, Any] | object, header: str) -> str:
    data = metadata(source)
    for container_name, container in (("source", source), ("metadata", data)):
        for key in (header, header.replace("-", "_"), header.title()):
            value = field_value(get(container, key) if container_name == "source" else container.get(key))
            if value:
                return value
    for container in (get(source, "headers"), get(source, "response_headers"), data.get("headers"), data.get("response_headers")):
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == header:
                    return field_value(value)
    return ""
