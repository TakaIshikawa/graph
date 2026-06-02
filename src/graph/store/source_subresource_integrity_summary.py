"""Summarize subresource integrity usage in source HTML."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlparse

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_TAG_RE = re.compile(r"<(script|link)\b[^>]*(?:src|href)\s*=\s*['\"]([^'\"]+)['\"][^>]*>", re.I)
_ATTR_RE = re.compile(r"([a-zA-Z:-]+)\s*=\s*['\"]([^'\"]*)['\"]")
_ALGORITHMS = {"sha256", "sha384", "sha512"}


def summarize_source_subresource_integrity(
    sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5
) -> dict[str, Any]:
    total = missing = invalid = 0
    algorithms: Counter[str] = Counter()
    samples: list[dict[str, str]] = []
    limit = max(0, sample_limit)
    for index, source in enumerate(sources):
        total += 1
        sid = source_id(source) or str(index)
        for html in _html_values(source):
            for match in _TAG_RE.finditer(html):
                attrs = {key.casefold(): value for key, value in _ATTR_RE.findall(match.group(0))}
                url = attrs.get("src") or attrs.get("href") or match.group(2)
                if not _external(url):
                    continue
                integrity = attrs.get("integrity", "").strip()
                if not integrity:
                    missing += 1
                    kind = "missing"
                else:
                    alg = integrity.split("-", 1)[0].casefold()
                    if alg in _ALGORITHMS and "-" in integrity:
                        algorithms[alg] += 1
                        kind = alg
                    else:
                        invalid += 1
                        kind = "invalid"
                if len(samples) < limit:
                    samples.append({"source_id": sid, "tag": match.group(1).casefold(), "url": url, "classification": kind})
    samples.sort(key=lambda row: (sort_key(row["source_id"]), sort_key(row["url"])))
    return {
        "total_sources": total,
        "integrity_algorithm_counts": {key: algorithms[key] for key in sorted(algorithms, key=sort_key)},
        "missing_integrity_count": missing,
        "invalid_integrity_count": invalid,
        "samples": samples[:limit],
    }


def _html_values(source: Mapping[str, Any] | object) -> list[str]:
    data = metadata(source)
    values = [field_value(get(source, "content")), field_value(get(source, "html")), field_value(data.get("content")), field_value(data.get("html"))]
    return [value for value in values if value]


def _external(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
