"""Summarize HTTP Warning headers in sources."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "warning"
_CODE_RE = re.compile(r"^\s*(\d{3})\b")


def summarize_source_warning_headers(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    limit = max(0, sample_limit)
    code_counts: Counter[str] = Counter()
    invalid_samples: list[dict[str, str]] = []
    sources_with = stale = transformed = 0

    for index, source in enumerate(source_list):
        sid = source_id(source) or str(index)
        value = _lookup_header(source, _HEADER)
        if not value:
            continue
        sources_with += 1
        for warning in [field_value(part) for part in value.split(",") if field_value(part)]:
            match = _CODE_RE.match(warning)
            if not match:
                if len(invalid_samples) < limit:
                    invalid_samples.append({"source_id": sid, "value": warning})
                continue
            code = match.group(1)
            code_counts[code] += 1
            stale += code == "110"
            transformed += code == "214"

    return {
        "total_sources": len(source_list),
        "sources_with_warning": sources_with,
        "sources_missing_warning": len(source_list) - sources_with,
        "warning_code_counts": {key: code_counts[key] for key in sorted(code_counts, key=sort_key)},
        "stale_warning_count": stale,
        "transformation_warning_count": transformed,
        "invalid_warning_samples": invalid_samples,
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
