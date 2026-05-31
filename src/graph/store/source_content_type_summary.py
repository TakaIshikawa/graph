"""Summarize content types declared on source records."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_TOKEN_RE = re.compile(r"^[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+$")


def summarize_source_content_types(sources: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total = missing = invalid = 0
    media_type_counts: Counter[str] = Counter()
    top_level_type_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    samples: list[dict[str, str]] = []

    for source in sources:
        total += 1
        raw_value = _content_type_value(source)
        normalized = _normalize(raw_value)
        if normalized is None:
            bucket = "missing" if not field_value(raw_value) else "invalid"
            missing += bucket == "missing"
            invalid += bucket == "invalid"
            if len(samples) < limit:
                samples.append({"source_id": source_id(source), "content_type": field_value(raw_value), "bucket": bucket, "media_type": ""})
            continue
        media_type_counts[normalized] += 1
        top_level_type_counts[normalized.split("/", 1)[0]] += 1
        source_counts[source_id(source)] += 1
        if len(samples) < limit:
            samples.append({"source_id": source_id(source), "content_type": field_value(raw_value), "bucket": "valid", "media_type": normalized})

    samples.sort(key=lambda row: (sort_key(row["source_id"]), sort_key(row["content_type"])))
    return {
        "total_sources": total,
        "missing_content_type_count": missing,
        "invalid_content_type_count": invalid,
        "media_type_counts": {key: media_type_counts[key] for key in sorted(media_type_counts, key=sort_key)},
        "top_level_type_counts": {key: top_level_type_counts[key] for key in sorted(top_level_type_counts, key=sort_key)},
        "source_counts": {key: source_counts[key] for key in sorted(source_counts, key=sort_key)},
        "samples": samples[:limit],
    }


def _content_type_value(source: Any) -> Any:
    meta = metadata(source)
    for key in ("content_type", "mime_type"):
        value = meta.get(key)
        if field_value(value):
            return value
    for key in ("content_type", "mime_type"):
        value = get(source, key)
        if field_value(value):
            return value
    return None


def _normalize(value: Any) -> str | None:
    media_type = field_value(value).split(";", 1)[0].strip().casefold()
    if not media_type or not _TOKEN_RE.match(media_type):
        return None
    return media_type
