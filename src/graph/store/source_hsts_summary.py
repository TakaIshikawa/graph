"""Summarize Strict-Transport-Security headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "strict-transport-security"


def summarize_source_hsts_policies(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    limit = max(0, sample_limit)
    max_age_buckets: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    sources_with = include_subdomains_count = preload_count = missing_max_age_count = invalid_max_age_count = 0

    for index, source in enumerate(source_list):
        sid = source_id(source) or str(index)
        value = _lookup_header(source, _HEADER)
        if not value:
            continue
        sources_with += 1
        directives = _directives(value)
        if "includesubdomains" in directives:
            include_subdomains_count += 1
        if "preload" in directives:
            preload_count += 1
        max_age = directives.get("max-age")
        sample: dict[str, Any] = {"source_id": sid, "value": value}
        if max_age is None:
            missing_max_age_count += 1
        else:
            sample["max_age"] = field_value(max_age)
            try:
                max_age_buckets[_max_age_bucket(int(field_value(max_age)))] += 1
            except ValueError:
                invalid_max_age_count += 1
        if len(samples) < limit:
            samples.append(sample)

    return {
        "total_sources": len(source_list),
        "sources_with_hsts": sources_with,
        "max_age_buckets": {key: max_age_buckets[key] for key in ("lt_1_day", "1_29_days", "30_364_days", "gte_1_year") if max_age_buckets[key]},
        "include_subdomains_count": include_subdomains_count,
        "preload_count": preload_count,
        "missing_max_age_count": missing_max_age_count,
        "invalid_max_age_count": invalid_max_age_count,
        "missing_hsts_count": len(source_list) - sources_with,
        "samples": samples,
    }


def _directives(value: str) -> dict[str, str | None]:
    directives: dict[str, str | None] = {}
    for part in value.split(";"):
        key, sep, raw_value = part.strip().partition("=")
        name = field_value(key).casefold()
        if name:
            directives.setdefault(name, field_value(raw_value) if sep else None)
    return directives


def _max_age_bucket(value: int) -> str:
    if value < 86400:
        return "lt_1_day"
    if value < 30 * 86400:
        return "1_29_days"
    if value < 365 * 86400:
        return "30_364_days"
    return "gte_1_year"


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
