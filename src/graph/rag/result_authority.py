"""Heuristic authority scoring for RAG/search results."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import log10
from typing import Any
from urllib.parse import urlsplit

_MISSING = object()
_ID_KEYS = ("id", "unit_id", "source_id")
_URL_KEYS = ("url", "source_url", "canonical_url", "link", "permalink")
_DOMAIN_KEYS = ("domain", "source_domain", "hostname", "site")
_AUTHOR_KEYS = ("author", "authors", "creator", "byline")
_VENUE_KEYS = ("venue", "publication", "journal", "publisher", "source_project")
_SOURCE_TYPE_KEYS = ("source_type", "type", "kind")
_CITATION_KEYS = ("citation_count", "citations", "reference_count")
_CONFIDENCE_KEYS = ("confidence", "score", "source_confidence")

_SOURCE_TYPE_WEIGHTS = {
    "peer_reviewed": 0.22,
    "journal": 0.2,
    "academic": 0.18,
    "government": 0.18,
    "standards": 0.16,
    "official": 0.14,
    "news": 0.08,
    "blog": 0.02,
    "social": -0.08,
    "forum": -0.08,
}

_TRUSTED_SUFFIXES = (".gov", ".edu", ".int")
_KNOWN_AUTHORITY_DOMAINS = {
    "nature.com",
    "science.org",
    "who.int",
    "nih.gov",
    "nasa.gov",
    "un.org",
    "worldbank.org",
}


def _payload(result: Any) -> Any:
    if isinstance(result, tuple) and result:
        return result[0]
    return result


def _field_value(item: Any, key: str) -> Any:
    if item is _MISSING or item is None:
        return _MISSING
    if isinstance(item, Mapping):
        return item.get(key, _MISSING)
    return getattr(item, key, _MISSING)


def _result_value(result: Any, key: str) -> Any:
    payload = _payload(result)
    value = _field_value(payload, key)
    if value is not _MISSING and value is not None:
        return value

    metadata = _field_value(payload, "metadata")
    if isinstance(metadata, Mapping):
        value = metadata.get(key, _MISSING)
        if value is not _MISSING and value is not None:
            return value

    unit = _field_value(payload, "unit")
    if unit is not _MISSING and unit is not None:
        value = _field_value(unit, key)
        if value is not _MISSING and value is not None:
            return value
        metadata = _field_value(unit, "metadata")
        if isinstance(metadata, Mapping):
            return metadata.get(key, _MISSING)

    return value


def _string(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    if isinstance(value, Iterable) and not isinstance(value, str | bytes | Mapping):
        text = ", ".join(str(item).strip() for item in value if str(item).strip())
    else:
        text = str(value)
    text = " ".join(text.strip().split())
    return text or None


def _result_id(result: Any, index: int) -> str:
    for key in _ID_KEYS:
        value = _string(_result_value(result, key))
        if value is not None:
            return value
    return f"result-{index + 1}"


def _first_string(result: Any, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = _string(_result_value(result, key))
        if value is not None:
            return value
    return None


def _numeric(value: Any) -> float | None:
    if value is _MISSING or value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    if isinstance(value, Iterable) and not isinstance(value, str | bytes | Mapping):
        return float(len(list(value)))
    return None


def _first_numeric(result: Any, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = _numeric(_result_value(result, key))
        if value is not None:
            return value
    return None


def _domain_from_url(value: Any) -> str | None:
    text = _string(value)
    if text is None:
        return None
    parsed = urlsplit(text if "://" in text else f"https://{text}")
    domain = parsed.hostname or parsed.netloc
    if domain is None:
        return None
    domain = domain.casefold().rstrip(".")
    if domain.startswith("www."):
        domain = domain[4:]
    return domain or None


def _result_domain(result: Any) -> str | None:
    for key in _DOMAIN_KEYS:
        domain = _domain_from_url(_result_value(result, key))
        if domain is not None:
            return domain
    for key in _URL_KEYS:
        domain = _domain_from_url(_result_value(result, key))
        if domain is not None:
            return domain
    return None


def _bounded(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _add_signal(signals: list[str], label: str, delta: float) -> None:
    signals.append(f"{label} ({delta:+.2f})")


def score_result_authority(results: Iterable[Any]) -> list[dict[str, Any]]:
    """Score authority signals for each result without mutating inputs."""
    rows = []
    for index, result in enumerate(results):
        score = 0.25
        signals = ["baseline authority (+0.25)"]
        warnings = []

        source_type = _first_string(result, _SOURCE_TYPE_KEYS)
        if source_type is None:
            warnings.append("missing source type")
        else:
            normalized = source_type.casefold().replace("-", "_").replace(" ", "_")
            delta = _SOURCE_TYPE_WEIGHTS.get(normalized, 0.04)
            score += delta
            _add_signal(signals, f"source type {source_type}", delta)

        author = _first_string(result, _AUTHOR_KEYS)
        if author is None:
            warnings.append("missing author")
        else:
            score += 0.12
            _add_signal(signals, "author present", 0.12)

        venue = _first_string(result, _VENUE_KEYS)
        if venue is None:
            warnings.append("missing publication venue")
        else:
            score += 0.1
            _add_signal(signals, f"venue {venue}", 0.1)

        domain = _result_domain(result)
        if domain is None:
            warnings.append("missing URL or domain")
        elif domain in _KNOWN_AUTHORITY_DOMAINS or domain.endswith(_TRUSTED_SUFFIXES):
            score += 0.12
            _add_signal(signals, f"authority domain {domain}", 0.12)
        else:
            score += 0.04
            _add_signal(signals, f"domain {domain}", 0.04)

        citations = _first_numeric(result, _CITATION_KEYS)
        if citations is None:
            warnings.append("missing citation count")
        elif citations > 0:
            delta = min(log10(citations + 1.0) / 4.0, 0.16)
            score += delta
            _add_signal(signals, f"citations {citations:g}", delta)
        else:
            warnings.append("zero citations")

        confidence = _first_numeric(result, _CONFIDENCE_KEYS)
        if confidence is None:
            warnings.append("missing confidence")
        else:
            normalized_confidence = confidence if confidence <= 1 else confidence / 100
            delta = _bounded(normalized_confidence) * 0.13
            score += delta
            _add_signal(signals, f"confidence {confidence:g}", delta)

        rows.append(
            {
                "result_id": _result_id(result, index),
                "authority_score": round(_bounded(score), 3),
                "signals": signals,
                "warnings": warnings,
            }
        )

    return rows
