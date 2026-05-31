"""Summarize external URL schemes found in unit content and metadata."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlparse

from graph.export._report_csv import field_value, flatten_values, get, metadata, sort_key, unit_id

_URL_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9+.-]*:[^\s<>()\[\]\"']+")
_MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]]*]\(([^)\s]+)(?:\s+[^)]*)?\)")
_URL_FIELD_HINTS = ("url", "uri", "link", "href")


def summarize_unit_external_url_schemes(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total = 0
    url_counts: Counter[str] = Counter()
    unit_counts: defaultdict[str, set[str]] = defaultdict(set)
    examples: defaultdict[str, list[dict[str, str]]] = defaultdict(list)

    for index, unit in enumerate(units):
        total += 1
        uid = unit_id(unit) or str(index)
        seen_in_unit: set[tuple[str, str]] = set()
        for url, source in _urls(unit):
            scheme = _scheme(url)
            if not scheme:
                continue
            url_counts[scheme] += 1
            seen_in_unit.add((scheme, url))
            if len(examples[scheme]) < limit:
                examples[scheme].append({"unit_id": uid, "url": url, "source": source})
        for scheme, _url in seen_in_unit:
            unit_counts[scheme].add(uid)

    rows = [
        {"scheme": scheme, "url_count": url_counts[scheme], "unit_count": len(unit_counts[scheme]), "examples": examples[scheme]}
        for scheme in url_counts
    ]
    rows.sort(key=lambda row: (-int(row["url_count"]), sort_key(row["scheme"])))
    return {"total_units": total, "schemes": rows}


def _urls(unit: Any) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    content = str(get(unit, "content") or "")
    masked = content
    for match in _MARKDOWN_LINK_RE.finditer(content):
        url = _clean(match.group(1))
        if _scheme(url):
            rows.append((url, "content"))
        masked = masked.replace(match.group(0), " ")
    for match in _URL_RE.finditer(masked):
        url = _clean(match.group(0))
        if _scheme(url):
            rows.append((url, "content"))

    for key, value in metadata(unit).items():
        if not any(hint in str(key).casefold() for hint in _URL_FIELD_HINTS):
            continue
        for item in flatten_values(value):
            url = _clean(field_value(item))
            if _scheme(url):
                rows.append((url, f"metadata.{key}"))
    return rows


def _scheme(url: str) -> str:
    try:
        parsed = urlparse(url)
    except ValueError:
        return ""
    if not parsed.scheme:
        return ""
    if parsed.scheme.casefold() in {"http", "https", "ftp", "file"} and not (parsed.netloc or parsed.path.startswith("/")):
        return ""
    return parsed.scheme.casefold()


def _clean(url: str) -> str:
    return field_value(url).rstrip(".,;:!?\"]}'")
