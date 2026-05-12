"""Explainable source reliability scoring for retrieved RAG results."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import date, datetime, timezone
from math import log10
from typing import Any
from urllib.parse import urlsplit

_MISSING = object()

_ID_KEYS = ("id", "unit_id", "source_id")
_SOURCE_PROJECT_KEYS = ("source_project", "source", "project")
_DOMAIN_KEYS = ("domain", "source_domain", "site", "hostname", "host")
_URL_KEYS = ("url", "source_url", "canonical_url", "external_url", "link", "permalink", "uri")
_TIMESTAMP_KEYS = (
    "updated_at",
    "published_at",
    "publication_date",
    "created_at",
    "timestamp",
    "date",
)
_AUTHOR_KEYS = ("author", "authors", "creator", "byline")
_CITATION_KEYS = ("citation_count", "citations")
_INBOUND_KEYS = ("inbound_reference_count", "inbound_references", "backlinks")


def _validate_non_negative_int(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer or None")
    return value


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
        metadata_value = metadata.get(key, _MISSING)
        if metadata_value is not _MISSING and metadata_value is not None:
            return metadata_value

    unit = _field_value(payload, "unit")
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
    if isinstance(value, Iterable) and not isinstance(value, str | bytes | Mapping):
        text = ", ".join(str(item).strip() for item in value if str(item).strip())
    else:
        text = str(value)
    normalized = " ".join(text.strip().split())
    return normalized or None


def _result_id(result: Any, index: int) -> str:
    for key in _ID_KEYS:
        value = _string_value(_result_value(result, key))
        if value is not None:
            return value
    return f"result-{index + 1}"


def _source_project(result: Any) -> str:
    for key in _SOURCE_PROJECT_KEYS:
        value = _string_value(_result_value(result, key))
        if value is not None:
            return value
    return "unknown"


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


def _domain_from_url(value: Any) -> str | None:
    text = _string_value(value)
    if text is None:
        return None
    parsed = urlsplit(text if "://" in text else f"https://{text}")
    domain = parsed.hostname or parsed.netloc
    if domain is None:
        return None
    return _normalize_domain(domain)


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
            parsed = datetime(parsed_date.year, parsed_date.month, parsed_date.day, tzinfo=timezone.utc)
    else:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _result_timestamp(result: Any) -> datetime | None:
    timestamps = [
        parsed
        for key in _TIMESTAMP_KEYS
        if (parsed := _parse_datetime(_result_value(result, key))) is not None
    ]
    return max(timestamps) if timestamps else None


def _numeric_value(value: Any) -> float | None:
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


def _max_positive(result: Any, keys: tuple[str, ...]) -> float:
    return max(
        (number for key in keys if (number := _numeric_value(_result_value(result, key))) is not None),
        default=0.0,
    )


def _has_author(result: Any) -> bool:
    return any(_string_value(_result_value(result, key)) is not None for key in _AUTHOR_KEYS)


def _grade(score: float) -> str:
    if score >= 0.8:
        return "A"
    if score >= 0.65:
        return "B"
    if score >= 0.45:
        return "C"
    return "D"


def _bounded(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _score_result(
    result: Any,
    *,
    index: int,
    now: datetime,
    source_project_counts: Counter[str],
    total_results: int,
) -> dict[str, Any]:
    result_id = _result_id(result, index)
    title = _string_value(_result_value(result, "title"))
    source_project = _source_project(result)
    domain = _result_domain(result)
    timestamp = _result_timestamp(result)
    score = 0.25
    reasons = ["baseline reliability"]

    citation_count = _max_positive(result, _CITATION_KEYS)
    if citation_count > 0:
        score += min(log10(citation_count + 1.0) / 3.0, 1.0) * 0.22
        reasons.append(f"citation count {citation_count:g}")
    else:
        reasons.append("missing citation count")

    inbound_count = _max_positive(result, _INBOUND_KEYS)
    if inbound_count > 0:
        score += min(log10(inbound_count + 1.0) / 2.0, 1.0) * 0.16
        reasons.append(f"inbound reference count {inbound_count:g}")
    else:
        reasons.append("missing inbound references")

    if _has_author(result):
        score += 0.1
        reasons.append("author present")
    else:
        score -= 0.04
        reasons.append("missing author")

    if domain is not None:
        score += 0.09
        reasons.append(f"domain present: {domain}")
    else:
        score -= 0.06
        reasons.append("missing domain")

    if timestamp is None:
        score -= 0.04
        reasons.append("missing usable timestamp")
    else:
        age_days = max((now - timestamp).total_seconds() / 86400.0, 0.0)
        if age_days <= 90:
            score += 0.12
            reasons.append("recent timestamp")
        elif age_days <= 365:
            score += 0.06
            reasons.append("current timestamp")
        elif age_days <= 1095:
            score += 0.01
            reasons.append("aging timestamp")
        else:
            score -= 0.04
            reasons.append("stale timestamp")

    if source_project == "unknown":
        score -= 0.03
        reasons.append("missing source project")
    else:
        prevalence = source_project_counts[source_project] / max(total_results, 1)
        if prevalence < 0.5:
            score += 0.08
            reasons.append("adds source project diversity")
        else:
            score += 0.02
            reasons.append("common source project")

    bounded_score = round(_bounded(score), 3)
    return {
        "result_id": result_id,
        "title": title,
        "score": bounded_score,
        "grade": _grade(bounded_score),
        "reasons": reasons,
        "source_project": source_project,
        "domain": domain,
        "timestamp": timestamp.isoformat() if timestamp is not None else None,
    }


def score_source_reliability(
    results: Iterable[Any],
    *,
    now: datetime | date | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return deterministic reliability rows for retrieved result payloads."""
    limit_value = _validate_non_negative_int(limit, "limit")
    now_value = _coerce_now(now)
    result_list = list(results)
    source_project_counts = Counter(_source_project(result) for result in result_list)

    rows = [
        _score_result(
            result,
            index=index,
            now=now_value,
            source_project_counts=source_project_counts,
            total_results=len(result_list),
        )
        for index, result in enumerate(result_list)
    ]
    rows.sort(
        key=lambda item: (
            -item["score"],
            item["source_project"],
            item["domain"] or "",
            item["title"] or "",
            item["result_id"],
        )
    )
    if limit_value is not None:
        return rows[:limit_value]
    return rows
