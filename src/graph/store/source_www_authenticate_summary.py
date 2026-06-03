"""Summarize WWW-Authenticate headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "www-authenticate"
_KNOWN = {"bearer", "basic", "digest", "negotiate"}


def summarize_source_www_authenticate_headers(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    limit = max(0, sample_limit)
    scheme_counts: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    sources_with = empty_value_count = unknown_scheme_count = 0

    for index, source in enumerate(source_list):
        sid = source_id(source) or str(index)
        found, raw_values = _lookup_header(source, _HEADER)
        if not found:
            continue
        schemes = [_scheme(value) for value in raw_values if field_value(value)]
        schemes = [scheme for scheme in schemes if scheme]
        if not schemes:
            empty_value_count += 1
            continue
        sources_with += 1
        scheme_counts.update(schemes)
        unknown_scheme_count += sum(1 for scheme in schemes if scheme not in _KNOWN)
        if len(samples) < limit:
            samples.append({"source_id": sid, "schemes": sorted(dict.fromkeys(schemes), key=sort_key)})

    samples.sort(key=lambda row: sort_key(row["source_id"]))
    return {
        "total_sources": len(source_list),
        "sources_with_www_authenticate": sources_with,
        "missing_header_count": len(source_list) - sources_with - empty_value_count,
        "empty_value_count": empty_value_count,
        "scheme_counts": {key: scheme_counts[key] for key in sorted(scheme_counts, key=sort_key)},
        "bearer_count": scheme_counts["bearer"],
        "basic_count": scheme_counts["basic"],
        "digest_count": scheme_counts["digest"],
        "unknown_scheme_count": unknown_scheme_count,
        "samples": samples[:limit],
    }


def _scheme(value: Any) -> str:
    text = field_value(value).strip()
    if not text:
        return ""
    return text.split(None, 1)[0].casefold().rstrip(":")


def _values(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [field_value(item) for item in value]
    return [field_value(value)]


def _lookup_header(source: Mapping[str, Any] | object, header: str) -> tuple[bool, list[str]]:
    data = metadata(source)
    for container_name, container in (("source", source), ("metadata", data)):
        for key in (header, header.replace("-", "_"), header.title()):
            raw = get(container, key) if container_name == "source" else container.get(key)
            if raw is not None:
                return True, _values(raw)
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == header:
                    return True, _values(value)
    for container in (get(source, "headers"), get(source, "response_headers"), data.get("headers"), data.get("response_headers")):
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == header:
                    return True, _values(value)
    return False, []
