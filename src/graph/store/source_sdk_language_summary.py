"""Summarize SDK and client-library language hints in sources."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, flatten_values, get, metadata, sort_key, source_id

_LANGUAGES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("python", (r"\bpython\b", r"\bpip\s+install\b", r"pypi\.org")),
    ("javascript", (r"\bjavascript\b", r"\bnode\.?js\b", r"\bnpm\s+install\b", r"npmjs\.com")),
    ("typescript", (r"\btypescript\b",)),
    ("go", (r"\bgolang\b", r"\bgo\s+sdk\b", r"\bgo\s+client\b", r"pkg\.go\.dev")),
    ("java", (r"\bjava\b", r"\bmaven\b")),
    ("ruby", (r"\bruby\b", r"\bgem\s+install\b", r"rubygems\.org")),
    ("php", (r"\bphp\b", r"\bcomposer\s+require\b", r"packagist\.org")),
    ("dotnet", (r"\b\.net\b", r"\bdotnet\b", r"\bnuget\b")),
    ("rust", (r"\brust\b", r"\bcargo\s+add\b", r"crates\.io")),
)
_SDK_CONTEXT = re.compile(r"\b(?:sdk|client\s+library|client[-\s]?libraries|api\s+client|github\.com|npm|pypi|maven|nuget|rubygems|packagist|crates\.io|pkg\.go\.dev)\b", re.I)


def summarize_source_sdk_languages(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows = [_row(source, index) for index, source in enumerate(source_list)]
    hinted = [row for row in rows if row["languages"]]
    counts = Counter(language for row in hinted for language in row["languages"])
    package_hints = Counter(hint for row in hinted for hint in row["package_hints"])
    return {
        "total_sources": len(source_list),
        "sources_with_sdk_language_hints": len(hinted),
        "language_counts": dict(sorted(counts.items())),
        "package_hint_counts": dict(sorted(package_hints.items())),
        "samples": sorted(hinted, key=lambda row: sort_key(row["source_id"]))[: max(0, sample_limit)],
    }


def _row(source: Mapping[str, Any] | object, index: int) -> dict[str, Any]:
    text = _source_text(source)
    has_sdk_context = bool(_SDK_CONTEXT.search(text))
    languages = [language for language, patterns in _LANGUAGES if has_sdk_context and any(re.search(pattern, text, re.I) for pattern in patterns)]
    package_hints = [hint for hint, pattern in (("github", r"github\.com"), ("npm", r"npmjs\.com|\bnpm\s+install\b"), ("pypi", r"pypi\.org|\bpip\s+install\b"), ("maven", r"\bmaven\b"), ("nuget", r"\bnuget\b")) if re.search(pattern, text, re.I)]
    return {"source_id": source_id(source) or str(index), "url": field_value(get(source, "url") or get(source, "source_url")), "languages": languages, "package_hints": package_hints}


def _source_text(source: Mapping[str, Any] | object) -> str:
    values = [get(source, key) for key in ("url", "source_url", "title", "content", "text", "snippet", "description")]
    values.extend(flatten_values(metadata(source)))
    return " ".join(field_value(value) for value in values if field_value(value))
