"""Summarize UTM parameters in Markdown links and bare URLs."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import parse_qsl, urlparse

from graph.export._report_csv import get, metadata, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]\n]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_BARE_URL_RE = re.compile(r"https?://[^\s<>)]+")


def summarize_unit_markdown_link_utm_parameters(units: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    """Summarize URLs carrying ``utm_*`` query parameters."""
    unit_list = list(units)
    utm_urls = 0
    units_with: set[str] = set()
    param_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    medium_counts: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    for index, unit in enumerate(unit_list):
        uid = unit_id(unit) or str(index)
        for url in _urls(_content(unit)):
            params = [(name.casefold(), value) for name, value in parse_qsl(urlparse(url).query, keep_blank_values=True) if name.casefold().startswith("utm_")]
            if not params:
                continue
            names = sorted({name for name, _ in params}, key=sort_key)
            utm_urls += 1
            units_with.add(uid)
            param_counts.update(name for name, _ in params)
            source_counts.update(value for name, value in params if name == "utm_source" and value)
            medium_counts.update(value for name, value in params if name == "utm_medium" and value)
            samples.append({"unit_id": uid, "url": url, "parameter_names": names})
    samples.sort(key=lambda row: (sort_key(row["unit_id"]), sort_key(row["url"])))
    return {
        "total_units": len(unit_list),
        "utm_link_count": utm_urls,
        "units_with_utm_links": len(units_with),
        "parameter_name_counts": dict(sorted(param_counts.items(), key=lambda item: sort_key(item[0]))),
        "source_counts": dict(sorted(source_counts.items(), key=lambda item: sort_key(item[0]))),
        "medium_counts": dict(sorted(medium_counts.items(), key=lambda item: sort_key(item[0]))),
        "samples": samples[:sample_limit],
    }


def _content(unit: Mapping[str, Any] | object) -> str:
    return str(get(unit, "content") or metadata(unit).get("content") or "")


def _urls(content: str) -> list[str]:
    rows: list[str] = []
    in_fence = False
    for line in content.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        markdown_spans: list[tuple[int, int]] = []
        for match in _MARKDOWN_LINK_RE.finditer(line):
            rows.append(match.group(1))
            markdown_spans.append(match.span())
        for match in _BARE_URL_RE.finditer(line):
            if not any(start <= match.start() < end for start, end in markdown_spans):
                rows.append(match.group(0).rstrip(".,;:"))
    return rows
