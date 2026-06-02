"""Summarize Content-MD5 headers in sources."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, source_id

_HEADER = "content-md5"
_BASE64_RE = re.compile(r"^[A-Za-z0-9+/]{22}==$|^[A-Fa-f0-9]{32}$")


def summarize_source_content_md5s(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    limit = max(0, sample_limit)
    digest_counts: Counter[str] = Counter()
    samples: list[dict[str, str]] = []
    sources_with = invalid = 0

    for index, source in enumerate(source_list):
        sid = source_id(source) or str(index)
        digest = field_value(_lookup_header(source, _HEADER)).strip("\"'")
        if not digest:
            continue
        sources_with += 1
        normalized = digest.casefold() if re.fullmatch(r"[A-Fa-f0-9]{32}", digest) else digest
        digest_counts[normalized] += 1
        if not _BASE64_RE.fullmatch(digest):
            invalid += 1
        if len(samples) < limit:
            samples.append({"source_id": sid, "digest": normalized, "valid": str(bool(_BASE64_RE.fullmatch(digest))).lower()})

    return {
        "total_sources": len(source_list),
        "sources_with_content_md5": sources_with,
        "missing_content_md5_count": len(source_list) - sources_with,
        "duplicate_digest_count": sum(count - 1 for count in digest_counts.values() if count > 1),
        "invalid_content_md5_count": invalid,
        "samples": samples,
    }


def _lookup_header(source: Mapping[str, Any] | object, header: str) -> str:
    data = metadata(source)
    for container_name, container in (("source", source), ("metadata", data)):
        for key in (header, header.replace("-", "_"), header.title()):
            value = field_value(get(container, key) if container_name == "source" else container.get(key))
            if value:
                return value
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == header:
                    return field_value(value)
    for container in (get(source, "headers"), get(source, "response_headers"), data.get("headers"), data.get("response_headers")):
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == header:
                    return field_value(value)
    return ""
