"""Summarize Retry-After headers in sources."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from email.utils import parsedate_to_datetime
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "retry-after"


def summarize_source_retry_after_headers(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    limit = max(0, sample_limit)
    samples: list[dict[str, str]] = []
    sources_with = empty_value_count = numeric_delay_count = http_date_count = invalid_value_count = 0
    max_delay_seconds: int | None = None

    for index, source in enumerate(source_list):
        sid = source_id(source) or str(index)
        found, raw = _lookup_header(source, _HEADER)
        if not found:
            continue
        value = field_value(raw)
        if not value:
            empty_value_count += 1
            continue
        sources_with += 1
        kind = "invalid"
        if value.isdecimal():
            kind = "numeric_delay"
            delay = int(value)
            numeric_delay_count += 1
            max_delay_seconds = delay if max_delay_seconds is None else max(max_delay_seconds, delay)
        elif _is_http_date(value):
            kind = "http_date"
            http_date_count += 1
        else:
            invalid_value_count += 1
        if len(samples) < limit:
            samples.append({"source_id": sid, "retry_after": value, "kind": kind})

    samples.sort(key=lambda row: sort_key(row["source_id"]))
    return {
        "total_sources": len(source_list),
        "sources_with_retry_after": sources_with,
        "missing_header_count": len(source_list) - sources_with - empty_value_count,
        "empty_value_count": empty_value_count,
        "numeric_delay_count": numeric_delay_count,
        "http_date_count": http_date_count,
        "invalid_value_count": invalid_value_count,
        "max_delay_seconds": max_delay_seconds,
        "samples": samples[:limit],
    }


def _is_http_date(value: str) -> bool:
    try:
        return parsedate_to_datetime(value) is not None
    except (TypeError, ValueError, IndexError, OverflowError):
        return False


def _lookup_header(source: Mapping[str, Any] | object, header: str) -> tuple[bool, str]:
    data = metadata(source)
    for container_name, container in (("source", source), ("metadata", data)):
        for key in (header, header.replace("-", "_"), header.title()):
            raw = get(container, key) if container_name == "source" else container.get(key)
            if raw is not None:
                return True, field_value(raw)
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == header:
                    return True, field_value(value)
    for container in (get(source, "headers"), get(source, "response_headers"), data.get("headers"), data.get("response_headers")):
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == header:
                    return True, field_value(value)
    return False, ""
