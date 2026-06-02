"""Summarize viewport meta coverage in sources."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, flatten_values, get, metadata, sort_key, source_id

_META_VIEWPORT_RE = re.compile(
    r"<meta\b(?=[^>]*\bname\s*=\s*['\"]?viewport['\"]?)[^>]*\bcontent\s*=\s*['\"]([^'\"]*)['\"][^>]*>",
    re.I,
)


def summarize_source_viewport_meta(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows = [_row(source, index) for index, source in enumerate(source_list)]
    present = [row for row in rows if row["viewport"]]
    directive_counts = Counter(directive for row in present for directive in row["directives"])
    limit = max(0, sample_limit)
    samples = [
        {"source_id": row["source_id"], "viewport": row["viewport"], "directives": row["directives"]}
        for row in sorted(present, key=lambda row: sort_key(row["source_id"]))[:limit]
    ]
    return {
        "total_sources": len(source_list),
        "sources_with_viewport_meta": len(present),
        "width_device_width_count": sum(1 for row in present if row["width_device_width"]),
        "initial_scale_count": sum(1 for row in present if row["initial_scale"]),
        "user_scalable_disabled_count": sum(1 for row in present if row["user_scalable_disabled"]),
        "missing_viewport_meta_count": len(source_list) - len(present),
        "directive_counts": dict(sorted(directive_counts.items())),
        "samples": samples,
    }


def _row(source: Mapping[str, Any] | object, index: int) -> dict[str, Any]:
    viewport = _viewport_value(source)
    directives = _directives(viewport)
    return {
        "source_id": source_id(source) or str(index),
        "viewport": viewport,
        "directives": directives,
        "width_device_width": any(_directive_pair(part) == ("width", "device-width") for part in _directive_parts(viewport)),
        "initial_scale": "initial-scale" in directives,
        "user_scalable_disabled": any(_directive_pair(part) in {("user-scalable", "no"), ("user-scalable", "0")} for part in _directive_parts(viewport)),
    }


def _viewport_value(source: Mapping[str, Any] | object) -> str:
    data = metadata(source)
    for key in ("viewport", "meta_viewport", "viewport_meta"):
        value = field_value(get(source, key)) or field_value(data.get(key))
        if value:
            return value
    for value in flatten_values(data):
        text = field_value(value)
        if text and _looks_like_viewport(text):
            return text
    text = " ".join(field_value(value) for key in ("content", "text", "snippet", "description") if (value := get(source, key)))
    match = _META_VIEWPORT_RE.search(text)
    return field_value(match.group(1)) if match else ""


def _looks_like_viewport(value: str) -> bool:
    lowered = value.casefold()
    return any(token in lowered for token in ("width=device-width", "initial-scale", "user-scalable"))


def _directive_parts(value: object) -> list[str]:
    return [part.strip() for part in field_value(value).split(",") if part.strip()]


def _directives(value: object) -> list[str]:
    return [name for part in _directive_parts(value) if (name := _directive_pair(part)[0])]


def _directive_pair(part: str) -> tuple[str, str]:
    name, _, value = part.partition("=")
    return name.strip().casefold(), value.strip().casefold()
