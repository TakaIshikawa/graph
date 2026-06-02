"""Summarize Accept-Language request headers in sources."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_TAG_RE = re.compile(r"^\*|[A-Za-z]{1,8}(?:-[A-Za-z0-9]{1,8})*$")


def summarize_source_accept_languages(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    languages: Counter[str] = Counter()
    regions: Counter[str] = Counter()
    wildcard = malformed = with_header = 0
    samples: list[dict[str, Any]] = []
    limit = max(0, sample_limit)
    for index, source in enumerate(source_list):
        sid = source_id(source) or str(index)
        value = _lookup_header(source, "accept-language")
        if not value:
            continue
        with_header += 1
        parsed: list[str] = []
        for part in value.split(","):
            tag, _, params = part.strip().partition(";")
            q_ok = True
            if params.strip():
                key, _, raw_q = params.strip().partition("=")
                q_ok = key.strip().casefold() == "q" and _valid_q(raw_q.strip())
            if not tag or not q_ok or not _TAG_RE.match(tag):
                malformed += 1
                continue
            normalized = tag.casefold()
            parsed.append(normalized)
            if normalized == "*":
                wildcard += 1
                continue
            pieces = normalized.split("-")
            languages[pieces[0]] += 1
            if len(pieces) > 1:
                regions[pieces[1]] += 1
        if parsed and len(samples) < limit:
            samples.append({"source_id": sid, "languages": parsed})
    samples.sort(key=lambda row: sort_key(row["source_id"]))
    return {
        "total_sources": len(source_list),
        "sources_with_accept_language": with_header,
        "language_counts": {key: languages[key] for key in sorted(languages, key=sort_key)},
        "region_counts": {key: regions[key] for key in sorted(regions, key=sort_key)},
        "wildcard_count": wildcard,
        "malformed_count": malformed,
        "missing_accept_language_count": len(source_list) - with_header,
        "samples": samples[:limit],
    }


def _valid_q(value: str) -> bool:
    try:
        number = float(value)
    except ValueError:
        return False
    return 0 <= number <= 1


def _lookup_header(source: Mapping[str, Any] | object, header: str) -> str:
    data = metadata(source)
    for container in (source, data, get(source, "request_headers"), data.get("request_headers"), get(source, "headers"), data.get("headers")):
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == header:
                    return field_value(value)
    return ""
