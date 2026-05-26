"""Classify evidence source level and compute primary-source ratio."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlsplit

_LEVELS = ("primary", "secondary", "tertiary", "unknown")
_PRIMARY_DOMAINS = (".gov", ".edu", ".int")


def analyze_evidence_primary_source_ratio(evidence: Iterable[Any]) -> dict[str, Any]:
    classifications = [_classify(item) for item in evidence]
    counts = Counter(classifications)
    total = len(classifications)
    return {
        "counts": {level: counts.get(level, 0) for level in _LEVELS},
        "primary_ratio": round((counts["primary"] / total) if total else 0.0, 4),
        "flagged_gaps": _gaps(counts, total),
    }


def _classify(item: Any) -> str:
    explicit = _text(_first(item, ("source_type", "publication_type", "evidence_type"))).casefold()
    for level in _LEVELS:
        if level in explicit:
            return level
    if any(term in explicit for term in ("journal", "review", "news", "analysis")):
        return "secondary"
    if any(term in explicit for term in ("encyclopedia", "wiki", "summary")):
        return "tertiary"
    domain = _domain(_first(item, ("url", "source_url", "link", "domain")))
    if domain and (domain.endswith(_PRIMARY_DOMAINS) or domain in {"who.int", "sec.gov"}):
        return "primary"
    return "unknown"


def _gaps(counts: Counter[str], total: int) -> list[str]:
    gaps = []
    if total == 0:
        gaps.append("no_evidence")
    elif counts["primary"] == 0:
        gaps.append("no_primary_sources")
    if counts["unknown"]:
        gaps.append("unknown_source_level")
    return gaps


def _first(item: Any, keys: tuple[str, ...]) -> Any:
    for container in (item, _value(item, "metadata")):
        if container is None:
            continue
        for key in keys:
            value = _value(container, key)
            if value not in (None, ""):
                return value
    return None


def _value(item: Any, key: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(key)
    return getattr(item, key, None)


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _domain(value: Any) -> str | None:
    if value in (None, ""):
        return None
    parsed = urlsplit(str(value) if "://" in str(value) else f"https://{value}")
    return parsed.hostname.casefold() if parsed.hostname else None
