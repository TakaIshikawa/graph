"""Summarize Last-Modified headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "last-modified"


def summarize_source_last_modified_headers(
    sources: Iterable[Mapping[str, Any] | object],
    sample_limit: int = 5,
) -> dict[str, Any]:
    source_list = list(sources)
    limit = max(0, sample_limit)
    date_counts: Counter[str] = Counter()
    rows_by_date: dict[str, dict[str, Any]] = {}
    invalid_count = 0
    sources_with = 0

    for index, source in enumerate(source_list):
        sid = source_id(source) or str(index)
        value = _lookup_header(source, _HEADER)
        if not value:
            continue
        sources_with += 1
        parsed = _parse_http_date(value)
        if parsed is None:
            invalid_count += 1
            continue
        date = parsed.date().isoformat()
        date_counts[date] += 1
        row = rows_by_date.setdefault(date, {"date": date, "count": 0, "source_ids": [], "examples": []})
        row["count"] += 1
        if sid not in row["source_ids"] and len(row["source_ids"]) < limit:
            row["source_ids"].append(sid)
        if value not in row["examples"] and len(row["examples"]) < limit:
            row["examples"].append(value)

    invalid_examples = _invalid_examples(source_list, limit)
    return {
        "total_sources": len(source_list),
        "sources_with_last_modified": sources_with,
        "missing_last_modified_count": len(source_list) - sources_with,
        "invalid_last_modified_count": invalid_count,
        "rows": [rows_by_date[key] for key in sorted(rows_by_date, key=sort_key)],
        "date_counts": {key: date_counts[key] for key in sorted(date_counts, key=sort_key)},
        "invalid_examples": invalid_examples,
    }


def _invalid_examples(sources: list[Mapping[str, Any] | object], limit: int) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    for index, source in enumerate(sources):
        value = _lookup_header(source, _HEADER)
        if not value or _parse_http_date(value) is not None:
            continue
        examples.append({"source_id": source_id(source) or str(index), "value": value})
        if len(examples) >= limit:
            break
    return examples


def _parse_http_date(value: str) -> datetime | None:
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


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
