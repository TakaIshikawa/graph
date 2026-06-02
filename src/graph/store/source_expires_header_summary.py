"""Summarize Expires headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "expires"


def summarize_source_expires_headers(
    sources: Iterable[Mapping[str, Any] | object],
    reference_time: datetime | None = None,
    sample_limit: int = 5,
) -> dict[str, Any]:
    source_list = list(sources)
    now = reference_time or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    limit = max(0, sample_limit)
    date_counts: Counter[str] = Counter()
    invalid_samples: list[dict[str, str]] = []
    sources_with = expired = future = 0

    for index, source in enumerate(source_list):
        sid = source_id(source) or str(index)
        value = _lookup_header(source, _HEADER)
        if not value:
            continue
        sources_with += 1
        parsed = _parse_http_date(value)
        if parsed is None:
            if len(invalid_samples) < limit:
                invalid_samples.append({"source_id": sid, "value": value})
            continue
        date_counts[parsed.date().isoformat()] += 1
        if parsed <= now:
            expired += 1
        else:
            future += 1

    return {
        "total_sources": len(source_list),
        "sources_with_expires": sources_with,
        "sources_missing_expires": len(source_list) - sources_with,
        "expired_count": expired,
        "future_expiry_count": future,
        "invalid_expires_samples": invalid_samples,
        "top_expiry_dates": {key: date_counts[key] for key in sorted(date_counts, key=lambda key: (-date_counts[key], sort_key(key)))[:10]},
    }


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
