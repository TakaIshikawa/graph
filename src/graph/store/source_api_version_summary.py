"""Summarize API version hints in sources."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import parse_qsl, urlparse

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_PATH_RE = re.compile(r"/(v\d+(?:\.\d+)?)(?:/|$)", re.I)
_HEADER_KEYS = ("x-api-version", "api-version", "api_version", "version")
_URL_KEYS = ("url", "uri", "endpoint", "api_url", "source_url")


def summarize_source_api_versions(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows = [_row(source, index) for index, source in enumerate(source_list)]
    present = [row for row in rows if row["version"]]
    samples = [
        {"source_id": row["source_id"], "version": row["version"], "location": row["location"]}
        for row in sorted(present, key=lambda row: sort_key(row["source_id"]))[: max(0, sample_limit)]
    ]
    return {
        "total_sources": len(source_list),
        "sources_with_version": len(present),
        "version_counts": dict(sorted(Counter(row["version"] for row in present).items())),
        "location_counts": dict(sorted(Counter(row["location"] for row in present).items())),
        "deprecated_version_count": sum(1 for row in rows if row["deprecated"]),
        "samples": samples,
    }


def _row(source: Mapping[str, Any] | object, index: int) -> dict[str, Any]:
    version, location = _version(source)
    return {"source_id": source_id(source) or str(index), "version": version, "location": location, "deprecated": _deprecated(source)}


def _version(source: Mapping[str, Any] | object) -> tuple[str, str]:
    data = metadata(source)
    for field in _URL_KEYS:
        text = field_value(get(source, field)) or field_value(data.get(field))
        if not text:
            continue
        parsed = urlparse(text)
        if match := _PATH_RE.search(parsed.path):
            return match.group(1).lower(), "url_path"
        for key, value in parse_qsl(parsed.query):
            if key.casefold().replace("_", "-") in {"api-version", "version"} and field_value(value):
                return field_value(value).lower(), "url_query"
    for container in (source, data, get(source, "headers"), get(source, "response_headers"), data.get("headers"), data.get("response_headers")):
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") in {name.replace("_", "-") for name in _HEADER_KEYS} and field_value(value):
                    return field_value(value).lower(), "header"
    return "", ""


def _deprecated(source: Mapping[str, Any] | object) -> bool:
    data = metadata(source)
    for key in ("deprecated", "is_deprecated", "deprecation_status"):
        text = field_value(get(source, key)) or field_value(data.get(key))
        if text.casefold() in {"true", "yes", "deprecated", "1"}:
            return True
    return False
