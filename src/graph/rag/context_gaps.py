"""Detect weak or missing coverage in retrieved RAG context."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import date, datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlsplit

_MISSING = object()
_RECENT_WINDOW_DAYS = 30

_DATE_KEYS = (
    "published_at",
    "publication_date",
    "updated_at",
    "created_at",
    "timestamp",
    "date",
    "crawled_at",
)
_DOMAIN_KEYS = ("domain", "source_domain", "site", "hostname", "host")
_URL_KEYS = (
    "url",
    "source_url",
    "canonical_url",
    "external_url",
    "link",
    "permalink",
    "uri",
)
_REQUIRED_ALIASES = {
    "tag": "tags",
    "tags": "tags",
    "source_project": "source_projects",
    "source_projects": "source_projects",
    "source": "source_projects",
    "sources": "source_projects",
    "domain": "domains",
    "domains": "domains",
    "content_type": "content_types",
    "content_types": "content_types",
}


def _validate_non_negative_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _field_value(item: Any, key: str) -> Any:
    if item is _MISSING or item is None:
        return _MISSING
    if isinstance(item, Mapping):
        return item.get(key, _MISSING)
    return getattr(item, key, _MISSING)


def _result_payload(result: Any) -> Any:
    if isinstance(result, tuple) and result:
        return result[0]
    return result


def _result_value(result: Any, key: str) -> Any:
    payload = _result_payload(result)
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
    text = " ".join(str(value).strip().split())
    return text or None


def _result_id(result: Any, index: int) -> str:
    for key in ("id", "unit_id", "source_id"):
        value = _string_value(_result_value(result, key))
        if value is not None:
            return value
    return f"result-{index + 1}"


def _iter_string_values(value: Any) -> list[str]:
    if value is _MISSING or value is None:
        return []
    if isinstance(value, Iterable) and not isinstance(value, str | bytes | Mapping):
        values = {_string_value(item) for item in value}
        return sorted(item for item in values if item is not None)
    string = _string_value(value)
    return [] if string is None else [string]


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


def _facet_counts(counter: Counter[str]) -> list[dict[str, Any]]:
    return [
        {"value": value, "count": count}
        for value, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _add_representative(
    representatives: dict[str, dict[str, list[str]]],
    facet: str,
    value: str,
    result_id: str,
) -> None:
    values = representatives[facet].setdefault(value, [])
    if result_id not in values:
        values.append(result_id)


def _required_values(required_facets: Mapping[str, Any] | None) -> dict[str, list[str]]:
    required = {
        "tags": [],
        "source_projects": [],
        "domains": [],
        "content_types": [],
    }
    if required_facets is None:
        return required
    if not isinstance(required_facets, Mapping):
        raise ValueError("required_facets must be a mapping or None")

    for key, raw_value in required_facets.items():
        facet = _REQUIRED_ALIASES.get(str(key))
        if facet is None:
            continue
        if facet == "domains":
            values = {
                domain
                for item in _iter_string_values(raw_value)
                if (domain := _normalize_domain(item)) is not None
            }
        else:
            values = set(_iter_string_values(raw_value))
        required[facet] = sorted(set(required[facet]) | values)
    return required


def _missing_required(
    required: Mapping[str, list[str]],
    coverage: Mapping[str, Counter[str]],
) -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {}
    for facet, values in required.items():
        present = coverage[facet]
        missing_values = [value for value in values if value not in present]
        if missing_values:
            missing[facet] = missing_values
    return missing


def _format_values(values: Iterable[str]) -> str:
    return ", ".join(f"`{value}`" for value in values)


def detect_context_gaps(
    results: Iterable[Mapping[str, Any]],
    *,
    required_facets: Mapping[str, Any] | None = None,
    min_sources: int = 2,
    min_recent_items: int = 0,
    recency_window_days: int = _RECENT_WINDOW_DAYS,
    now: datetime | date | None = None,
) -> dict[str, Any]:
    """Return deterministic coverage gaps for retrieved RAG result payloads.

    Recent items are results with a parseable timestamp in the recency window.
    Flat result fields take precedence over metadata and optional nested
    ``unit`` fields. Inputs are inspected only and are not mutated.
    """

    min_source_count = _validate_non_negative_int(min_sources, "min_sources")
    min_recent_count = _validate_non_negative_int(
        min_recent_items,
        "min_recent_items",
    )
    recent_window_days = _validate_non_negative_int(
        recency_window_days,
        "recency_window_days",
    )
    now_value = _coerce_now(now)
    recent_cutoff = now_value - timedelta(days=recent_window_days)
    required = _required_values(required_facets)
    result_list = list(results)

    source_projects: Counter[str] = Counter()
    domains: Counter[str] = Counter()
    content_types: Counter[str] = Counter()
    tags: Counter[str] = Counter()
    representatives: dict[str, dict[str, list[str]]] = {
        "source_projects": {},
        "domains": {},
        "content_types": {},
        "tags": {},
        "recent": {},
    }
    timestamps: list[datetime] = []
    recent_ids: list[str] = []

    for index, result in enumerate(result_list):
        result_id = _result_id(result, index)

        source_project = _string_value(_result_value(result, "source_project"))
        if source_project is not None:
            source_projects[source_project] += 1
            _add_representative(
                representatives,
                "source_projects",
                source_project,
                result_id,
            )

        domain = _result_domain(result)
        if domain is not None:
            domains[domain] += 1
            _add_representative(representatives, "domains", domain, result_id)

        content_type = _string_value(_result_value(result, "content_type"))
        if content_type is not None:
            content_types[content_type] += 1
            _add_representative(
                representatives,
                "content_types",
                content_type,
                result_id,
            )

        for tag in _iter_string_values(_result_value(result, "tags")):
            tags[tag] += 1
            _add_representative(representatives, "tags", tag, result_id)

        timestamp = _result_timestamp(result)
        if timestamp is None:
            continue
        timestamps.append(timestamp)
        if recent_cutoff <= timestamp <= now_value:
            recent_ids.append(result_id)

    newest_at = max(timestamps).isoformat() if timestamps else None
    oldest_at = min(timestamps).isoformat() if timestamps else None
    coverage_counters = {
        "source_projects": source_projects,
        "domains": domains,
        "content_types": content_types,
        "tags": tags,
    }
    missing = _missing_required(required, coverage_counters)

    gaps: list[dict[str, Any]] = []
    suggestions: list[str] = []
    if not result_list:
        gaps.append(
            {
                "type": "empty_results",
                "severity": "error",
                "message": "No retrieved results are available for context.",
            }
        )
        suggestions.append("Run a broader retrieval before generating an answer.")

    if len(source_projects) < min_source_count:
        gaps.append(
            {
                "type": "source_diversity",
                "severity": "warning",
                "message": "Retrieved context does not cover enough source projects.",
                "current": len(source_projects),
                "required": min_source_count,
            }
        )
        suggestions.append(
            f"Retrieve from at least {min_source_count} distinct source projects."
        )

    for facet, values in missing.items():
        gaps.append(
            {
                "type": f"missing_required_{facet}",
                "severity": "warning",
                "message": f"Required {facet.replace('_', ' ')} are missing.",
                "missing": values,
            }
        )
        suggestions.append(f"Add context matching {facet}: {_format_values(values)}.")

    if len(recent_ids) < min_recent_count:
        gaps.append(
            {
                "type": "recency",
                "severity": "warning",
                "message": "Retrieved context does not include enough recent items.",
                "current": len(recent_ids),
                "required": min_recent_count,
                "window_days": recent_window_days,
            }
        )
        suggestions.append(
            "Retrieve newer context from the last "
            f"{recent_window_days} days until at least "
            f"{min_recent_count} recent items are available."
        )

    representatives["recent"] = {"last_30_days": recent_ids}

    return {
        "coverage": {
            "result_count": len(result_list),
            "source_projects": _facet_counts(source_projects),
            "domains": _facet_counts(domains),
            "content_types": _facet_counts(content_types),
            "tags": _facet_counts(tags),
            "recency": {
                "dated_count": len(timestamps),
                "recent_count": len(recent_ids),
                "window_days": recent_window_days,
                "oldest_at": oldest_at,
                "newest_at": newest_at,
            },
        },
        "gaps": gaps,
        "suggestions": suggestions,
        "representative_result_ids": representatives,
    }
