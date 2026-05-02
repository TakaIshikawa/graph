"""Explainable source credibility scoring for RAG/search results."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime, timezone
from math import log10
from typing import Any
from urllib.parse import urlsplit

_MISSING = object()

_URL_KEYS = (
    "url",
    "source_url",
    "canonical_url",
    "external_url",
    "link",
    "permalink",
    "uri",
)
_DOMAIN_KEYS = ("domain", "source_domain", "site", "hostname", "host")
_DATE_KEYS = (
    "published_at",
    "publication_date",
    "updated_at",
    "created_at",
    "timestamp",
    "date",
    "crawled_at",
)
_CITATION_KEYS = (
    "citation_count",
    "citations",
    "reference_count",
    "references",
    "inbound_reference_count",
)
_CITATION_ID_KEYS = ("doi", "pmid", "arxiv_id", "isbn")


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return min(max(value, lower), upper)


def _field_value(item: Any, key: str) -> Any:
    if item is _MISSING or item is None:
        return _MISSING
    if isinstance(item, Mapping):
        return item.get(key, _MISSING)
    return getattr(item, key, _MISSING)


def _result_value(result: Any, key: str) -> Any:
    value = _field_value(result, key)
    if value is not _MISSING and value is not None:
        return value

    metadata = _field_value(result, "metadata")
    if isinstance(metadata, Mapping):
        metadata_value = metadata.get(key, _MISSING)
        if metadata_value is not _MISSING and metadata_value is not None:
            return metadata_value

    unit = _field_value(result, "unit")
    if unit is not _MISSING and unit is not None:
        unit_value = _field_value(unit, key)
        if unit_value is not _MISSING and unit_value is not None:
            return unit_value
        unit_metadata = _field_value(unit, "metadata")
        if isinstance(unit_metadata, Mapping):
            return unit_metadata.get(key, _MISSING)

    return value


def _string_value(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    text = " ".join(str(value).strip().split())
    return text or None


def _result_id(result: Any, index: int) -> str:
    for key in ("id", "unit_id", "source_id"):
        value = _string_value(_result_value(result, key))
        if value is not None:
            return value
    return f"result-{index + 1}"


def _domain_from_url(value: Any) -> str | None:
    text = _string_value(value)
    if text is None:
        return None

    parsed = urlsplit(text if "://" in text else f"https://{text}")
    domain = parsed.hostname or parsed.netloc
    if domain is None:
        return None
    return _normalize_domain(domain)


def _normalize_domain(value: Any) -> str | None:
    text = _string_value(value)
    if text is None:
        return None
    if "/" in text or "://" in text:
        return _domain_from_url(text)
    domain = text.casefold().rstrip(".")
    if domain.startswith("www."):
        domain = domain[4:]
    return domain or None


def _result_domain(result: Any) -> str | None:
    for key in _DOMAIN_KEYS:
        domain = _normalize_domain(_result_value(result, key))
        if domain is not None:
            return domain

    for key in _URL_KEYS:
        domain = _domain_from_url(_result_value(result, key))
        if domain is not None:
            return domain

    return None


def _coerce_now(now: datetime | date | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if isinstance(now, datetime):
        value = now
    elif isinstance(now, date):
        value = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    else:
        raise ValueError("now must be a date, datetime, or None")
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_datetime(value: Any) -> datetime | None:
    if value is _MISSING or value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            try:
                parsed_date = date.fromisoformat(text)
            except ValueError:
                return None
            parsed = datetime(
                parsed_date.year,
                parsed_date.month,
                parsed_date.day,
                tzinfo=timezone.utc,
            )
    else:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _result_timestamp(result: Any) -> datetime | None:
    dates = [
        parsed
        for key in _DATE_KEYS
        if (parsed := _parse_datetime(_result_value(result, key))) is not None
    ]
    if not dates:
        return None
    return max(dates)


def _numeric_value(value: Any) -> float | None:
    if value is _MISSING or value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        text = value.strip().rstrip("%")
        if not text:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
        if value.strip().endswith("%"):
            return number / 100.0
        return number
    return None


def _normalized_unit_interval(value: Any) -> float | None:
    number = _numeric_value(value)
    if number is None:
        return None
    if number > 1.0 and number <= 100.0:
        number = number / 100.0
    return _clamp(number)


def _trusted_domains(values: Iterable[str] | None) -> set[str]:
    if values is None:
        return set()
    return {domain for value in values if (domain := _normalize_domain(value))}


def _is_trusted_domain(domain: str | None, trusted_domains: set[str]) -> bool:
    if domain is None:
        return False
    return any(domain == trusted or domain.endswith(f".{trusted}") for trusted in trusted_domains)


def _citation_score(result: Any) -> tuple[float, str | None]:
    counts = [_numeric_value(_result_value(result, key)) for key in _CITATION_KEYS]
    count = max((value for value in counts if value is not None and value > 0), default=0.0)

    score = min(log10(count + 1.0) / 3.0, 1.0) if count > 0 else 0.0
    for key in _CITATION_ID_KEYS:
        if _string_value(_result_value(result, key)) is not None:
            score = max(score, 0.45)

    if score <= 0:
        return 0.0, None
    if count >= 1:
        return score, "citation signals present"
    return score, "stable citation identifier present"


def _content_text(result: Any) -> str:
    values = [
        _string_value(_result_value(result, "title")),
        _string_value(_result_value(result, "content")),
        _string_value(_result_value(result, "snippet")),
        _string_value(_result_value(result, "summary")),
    ]
    return " ".join(value for value in values if value)


def _band(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def _score_result(
    result: Any,
    *,
    index: int,
    trusted_domains: set[str],
    recency_half_life_days: float,
    now: datetime,
) -> dict[str, Any]:
    result_id = _result_id(result, index)
    title = _string_value(_result_value(result, "title"))
    domain = _result_domain(result)
    score = 0.45
    reasons = ["baseline source credibility"]

    if _is_trusted_domain(domain, trusted_domains):
        score += 0.18
        reasons.append(f"trusted domain: {domain}")
    elif domain is not None:
        score += 0.04
        reasons.append(f"identified domain: {domain}")
    else:
        score -= 0.08
        reasons.append("missing source domain")

    timestamp = _result_timestamp(result)
    if timestamp is None:
        score -= 0.03
        reasons.append("missing usable timestamp")
    else:
        age_days = max((now - timestamp).total_seconds() / 86400.0, 0.0)
        freshness = 0.5 ** (age_days / recency_half_life_days)
        score += (freshness - 0.5) * 0.22
        if age_days <= recency_half_life_days:
            reasons.append("recent timestamp")
        elif age_days >= recency_half_life_days * 2:
            reasons.append("stale timestamp")
        else:
            reasons.append("aging timestamp")

    confidence = _normalized_unit_interval(_result_value(result, "confidence"))
    if confidence is not None:
        score += (confidence - 0.5) * 0.16
        reasons.append(f"confidence {confidence:.2f}")

    utility = _normalized_unit_interval(_result_value(result, "utility_score"))
    if utility is not None:
        score += (utility - 0.5) * 0.12
        reasons.append(f"utility score {utility:.2f}")

    citation_score, citation_reason = _citation_score(result)
    if citation_reason is not None:
        score += citation_score * 0.12
        reasons.append(citation_reason)

    if title is None:
        score -= 0.03
        reasons.append("missing title")
    else:
        score += 0.03
        reasons.append("title present")

    content_text = _content_text(result)
    word_count = len(content_text.split())
    if word_count >= 80:
        score += 0.04
        reasons.append("substantive content")
    elif word_count < 8:
        score -= 0.04
        reasons.append("thin content")

    bounded_score = round(_clamp(score), 3)
    return {
        "id": result_id,
        "title": title,
        "domain": domain,
        "score": bounded_score,
        "band": _band(bounded_score),
        "reasons": reasons,
    }


def _validate_half_life(value: int | float) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool) or value <= 0:
        raise ValueError("recency_half_life_days must be a positive number")
    return float(value)


def score_source_credibility(
    results: Iterable[Mapping[str, Any]],
    *,
    trusted_domains: Iterable[str] | None = None,
    recency_half_life_days: int | float = 180,
    now: datetime | date | None = None,
) -> list[dict[str, Any]]:
    """Return deterministic credibility scores for RAG/search result dictionaries.

    The input results are only inspected, never modified. Higher scores indicate
    stronger source-level confidence based on explainable metadata signals.
    """
    half_life = _validate_half_life(recency_half_life_days)
    now_value = _coerce_now(now)
    trusted = _trusted_domains(trusted_domains)

    scored = [
        _score_result(
            result,
            index=index,
            trusted_domains=trusted,
            recency_half_life_days=half_life,
            now=now_value,
        )
        for index, result in enumerate(results)
    ]
    return sorted(
        scored,
        key=lambda item: (
            -item["score"],
            item["domain"] or "",
            item["title"] or "",
            item["id"],
        ),
    )
