"""Summarize NEL policy headers in sources."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "nel"


def summarize_source_nel_policies(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows = [_row(source, index) for index, source in enumerate(source_list)]
    present = [row for row in rows if row["value"]]
    rows_sorted = sorted(present, key=lambda row: sort_key(row["source_id"]))
    return {
        "total_sources": len(source_list),
        "sources_with_nel": len(present),
        "report_to_counts": dict(sorted(Counter(row["report_to"] for row in present if row["report_to"]).items())),
        "include_subdomains_count": sum(1 for row in present if row["include_subdomains"]),
        "success_fraction_count": sum(1 for row in present if row["success_fraction_present"]),
        "failure_fraction_count": sum(1 for row in present if row["failure_fraction_present"]),
        "malformed_count": sum(1 for row in present if row["malformed"]),
        "missing_count": len(source_list) - len(present),
        "rows": rows_sorted,
        "samples": rows_sorted[: max(0, sample_limit)],
    }


def _row(source: Mapping[str, Any] | object, index: int) -> dict[str, Any]:
    value = _lookup_header(source, _HEADER)
    parsed: Any
    try:
        parsed = json.loads(value) if value else {}
    except json.JSONDecodeError:
        parsed = {}
        malformed = True
    else:
        malformed = bool(value) and not isinstance(parsed, Mapping)
    return {
        "source_id": source_id(source) or str(index),
        "value": value,
        "report_to": field_value(parsed.get("report_to")) if isinstance(parsed, Mapping) else "",
        "include_subdomains": bool(isinstance(parsed, Mapping) and parsed.get("include_subdomains") is True),
        "success_fraction_present": bool(isinstance(parsed, Mapping) and "success_fraction" in parsed),
        "failure_fraction_present": bool(isinstance(parsed, Mapping) and "failure_fraction" in parsed),
        "malformed": malformed,
    }


def _lookup_header(source: Mapping[str, Any] | object, header: str) -> str:
    data = metadata(source)
    for container_name, container in (("source", source), ("metadata", data)):
        for key in (header, header.replace("-", "_"), header.upper(), header.title()):
            value = field_value(get(container, key) if container_name == "source" else container.get(key))
            if value:
                return value
    for container in (get(source, "headers"), get(source, "response_headers"), data.get("headers"), data.get("response_headers")):
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == header:
                    return field_value(value)
    return ""
