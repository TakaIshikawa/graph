"""Summarize generator meta values in sources."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_GENERATOR_KEYS = ("generator", "meta_generator")
_HTML_KEYS = ("html", "content", "text", "snippet", "description")
_META_GENERATOR_RE = re.compile(
    r"<meta\b(?=[^>]*\bname\s*=\s*['\"]?generator['\"]?)[^>]*\bcontent\s*=\s*['\"]([^'\"]*)['\"][^>]*>",
    re.I,
)
_STATIC_SITE_GENERATORS = ("hugo", "jekyll", "gatsby", "next.js", "nextjs", "next")


def summarize_source_generator_meta(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows = [_row(source, index) for index, source in enumerate(source_list)]
    present = [row for row in rows if row["generator"]]
    limit = max(0, sample_limit)
    samples = [
        {"source_id": row["source_id"], "generator": row["generator"]}
        for row in sorted(present, key=lambda row: sort_key(row["source_id"]))[:limit]
    ]
    return {
        "total_sources": len(source_list),
        "sources_with_generator": len(present),
        "generator_counts": dict(sorted(Counter(row["generator"] for row in present).items())),
        "wordpress_count": sum(1 for row in present if "wordpress" in row["generator"].casefold()),
        "static_site_generator_count": sum(1 for row in present if _is_static_site_generator(row["generator"])),
        "missing_generator_count": len(source_list) - len(present),
        "samples": samples,
    }


def _row(source: Mapping[str, Any] | object, index: int) -> dict[str, Any]:
    return {"source_id": source_id(source) or str(index), "generator": _generator_value(source)}


def _generator_value(source: Mapping[str, Any] | object) -> str:
    data = metadata(source)
    for key in _GENERATOR_KEYS:
        value = field_value(get(source, key)) or field_value(data.get(key))
        if value:
            return value
    for container in (source, data):
        for key in _HTML_KEYS:
            html = field_value(get(container, key) if container is source else container.get(key))
            match = _META_GENERATOR_RE.search(html)
            if match:
                return field_value(match.group(1))
    return ""


def _is_static_site_generator(value: str) -> bool:
    lowered = value.casefold()
    return any(generator in lowered for generator in _STATIC_SITE_GENERATORS)
