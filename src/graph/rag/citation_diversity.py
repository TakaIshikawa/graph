"""Analyze citation diversity across source metadata fields."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence, Mapping, Iterable
from datetime import date, datetime
from typing import Any
from urllib.parse import urlparse

from graph.rag._analysis_utils import domain_for, result_id, string, value


def _iter_strings(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, Mapping):
        return [text for item in raw.values() for text in _iter_strings(item)]
    if isinstance(raw, Iterable) and not isinstance(raw, str | bytes):
        return [text for item in raw for text in _iter_strings(item)]
    text = string(raw)
    return [] if text is None else [text]


def _domain(citation: Any) -> str | None:
    found = domain_for(citation)
    if found:
        return found
    for key in ("url", "source_url", "link", "uri"):
        text = string(value(citation, key))
        if text:
            host = urlparse(text).netloc.casefold()
            if host:
                return host.removeprefix("www.")
    return None


def _year(raw: Any) -> str | None:
    if isinstance(raw, datetime | date):
        return str(raw.year)
    text = string(raw)
    if not text:
        return None
    for token in text.replace("/", "-").split("-"):
        if len(token) == 4 and token.isdigit():
            return token
    return None


def _first(citation: Any, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        text = string(value(citation, key))
        if text:
            return text.casefold()
    return None


def _groups(citation: Any, index: int) -> dict[str, list[str]]:
    authors = _iter_strings(value(citation, "authors")) or _iter_strings(value(citation, "author")) or _iter_strings(value(citation, "creator"))
    year = _year(value(citation, "year")) or _year(value(citation, "date")) or _year(value(citation, "published_at"))
    return {
        "domain": [_domain(citation) or "unknown"],
        "author": sorted({author.casefold() for author in authors}) or ["unknown"],
        "source_type": [_first(citation, ("source_type", "type", "kind")) or "unknown"],
        "year": [year or "unknown"],
        "position": [f"position-{index + 1}"],
    }


def _metric(counter: Counter[str], total: int) -> dict[str, Any]:
    if total == 0:
        return {"unique_count": 0, "diversity_ratio": 0.0, "dominant": None}
    value, count = sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0]
    return {
        "unique_count": len(counter),
        "diversity_ratio": round(len(counter) / total, 3),
        "dominant": {"value": value, "count": count, "ratio": round(count / total, 3)},
    }


def analyze_citation_diversity(
    citations: Sequence[Mapping[str, Any]],
    *,
    dominance_threshold: float = 0.75,
) -> dict[str, Any]:
    """Return diversity ratios, dominant groups, and monoculture warnings."""
    if not 0 < dominance_threshold <= 1:
        raise ValueError("dominance_threshold must be greater than 0 and at most 1")
    counters = {name: Counter() for name in ("domain", "author", "source_type", "year", "position")}
    rows: list[dict[str, Any]] = []
    for index, citation in enumerate(citations):
        groups = _groups(citation, index)
        for name, values in groups.items():
            counters[name].update(values)
        rows.append({"citation_id": result_id(citation, index), "groups": groups})

    total = len(citations)
    metrics = {name: _metric(counter, total) for name, counter in counters.items()}
    warnings = []
    for name, metric in metrics.items():
        dominant = metric["dominant"]
        if name != "position" and dominant and dominant["ratio"] >= dominance_threshold and total > 1:
            warnings.append(f"dominant_{name}:{dominant['value']}")
    return {
        "citation_count": total,
        "metrics": metrics,
        "dominant_groups": {name: metric["dominant"] for name, metric in metrics.items()},
        "citations": rows,
        "warnings": warnings,
    }
