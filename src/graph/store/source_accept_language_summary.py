"""Summarize Accept-Language headers in sources."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_TAG_RE = re.compile(r"^(?:\*|[A-Za-z]{1,8}(?:-[A-Za-z0-9]{1,8})*)$")


def summarize_source_accept_languages(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    languages: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    limit = max(0, sample_limit)

    for index, source in enumerate(source_list):
        sid = source_id(source) or str(index)
        value = _lookup_header(source, "accept-language")
        if not value:
            continue
        language_ranges = _language_ranges(value)
        primary_languages = [item.split("-", 1)[0] for item in language_ranges if item != "*"]
        languages.update(primary_languages)
        row = {
            "source_id": sid,
            "accept_language": value,
            "language_ranges": language_ranges,
            "primary_languages": primary_languages,
            "language_count": len(language_ranges),
        }
        rows.append(row)
        if len(samples) < limit:
            samples.append(row)

    rows.sort(key=lambda row: sort_key(row["source_id"]))
    samples.sort(key=lambda row: sort_key(row["source_id"]))
    return {
        "total_sources": len(source_list),
        "sources_with_accept_language": len(rows),
        "missing_accept_language_count": len(source_list) - len(rows),
        "language_counts": {key: languages[key] for key in sorted(languages, key=sort_key)},
        "max_language_count": max((row["language_count"] for row in rows), default=0),
        "rows": rows,
        "samples": samples[:limit],
    }


def _language_ranges(value: str) -> list[str]:
    ranges: list[str] = []
    for part in value.split(","):
        tag, *params = [piece.strip() for piece in part.strip().split(";")]
        if not tag or not _TAG_RE.match(tag):
            continue
        if any(not _valid_param(param) for param in params if param):
            continue
        ranges.append(tag.casefold())
    return ranges


def _valid_param(value: str) -> bool:
    key, separator, raw_q = value.partition("=")
    return separator == "=" and key.strip().casefold() == "q" and _valid_q(raw_q.strip())


def _valid_q(value: str) -> bool:
    try:
        number = float(value)
    except ValueError:
        return False
    return 0 <= number <= 1


def _lookup_header(source: Mapping[str, Any] | object, header: str) -> str:
    data = metadata(source)
    for container in (
        source,
        data,
        get(source, "request_headers"),
        get(source, "headers"),
        get(source, "response_headers"),
        data.get("request_headers"),
        data.get("headers"),
        data.get("response_headers"),
    ):
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == header:
                    return field_value(value)
    return ""
