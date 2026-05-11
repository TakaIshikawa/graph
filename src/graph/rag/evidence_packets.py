"""Build compact evidence packets from RAG/search result payloads."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime, timezone
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
_CITATION_KEYS = (
    "citation",
    "citation_url",
    "citations",
    "reference",
    "reference_url",
    "references",
    "source_citation",
    "source_citations",
)
_IDENTIFIER_KEYS = ("doi", "arxiv", "arxiv_id", "isbn", "isbn10", "isbn13", "pmid")
_DATE_KEYS = (
    "published_at",
    "publication_date",
    "created_at",
    "updated_at",
    "ingested_at",
    "timestamp",
    "date",
)
_SOURCE_KEYS = ("source", "source_name", "source_project", "domain", "source_id")


def _validate_limit(limit: int | None) -> int | None:
    if limit is None:
        return None
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        raise ValueError("limit must be a non-negative integer or None")
    return limit


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


def _tuple_score(result: Any) -> Any:
    if isinstance(result, tuple) and len(result) > 1:
        return result[1]
    return _MISSING


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

    if key == "score":
        tuple_value = _tuple_score(result)
        if tuple_value is not _MISSING:
            return tuple_value
    return value


def _string_value(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    text = " ".join(str(value).strip().split())
    return text or None


def _float_value(value: Any) -> float | None:
    if value is _MISSING or value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _iter_strings(value: Any) -> list[str]:
    if value is _MISSING or value is None:
        return []
    if isinstance(value, Iterable) and not isinstance(value, str | bytes | Mapping):
        values = {_string_value(item) for item in value}
        return sorted(item for item in values if item is not None)
    string = _string_value(value)
    return [] if string is None else [string]


def _result_id(result: Any, index: int) -> str:
    for key in ("id", "unit_id", "source_id"):
        value = _string_value(_result_value(result, key))
        if value is not None:
            return value
    return f"result-{index + 1}"


def _source(result: Any) -> str | None:
    for key in _SOURCE_KEYS:
        value = _string_value(_result_value(result, key))
        if value is not None:
            return value
    return None


def _domain_from_url(value: Any) -> str | None:
    text = _string_value(value)
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


def _date_fields(result: Any) -> dict[str, str]:
    fields: dict[str, str] = {}
    for key in _DATE_KEYS:
        parsed = _parse_datetime(_result_value(result, key))
        if parsed is not None:
            fields[key] = parsed.isoformat()
    return fields


def _citation_fields(result: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for key in (*_URL_KEYS, *_IDENTIFIER_KEYS, *_CITATION_KEYS):
        value = _result_value(result, key)
        if _string_value(value) is None and not (
            isinstance(value, Iterable) and not isinstance(value, str | bytes | Mapping)
        ):
            continue
        if isinstance(value, Iterable) and not isinstance(value, str | bytes | Mapping):
            values = _iter_strings(value)
            if values:
                fields[key] = values
        else:
            text = _string_value(value)
            if text is not None:
                fields[key] = text
    return fields


def _snippet(result: Any, query: str | None, max_chars: int = 320) -> str | None:
    explicit = _string_value(_result_value(result, "snippet"))
    if explicit is not None:
        return explicit[:max_chars].strip()

    content = _string_value(_result_value(result, "content"))
    if content is None:
        return None
    if query:
        lowered = content.casefold()
        positions = [
            lowered.find(term)
            for term in str(query).casefold().split()
            if term and lowered.find(term) >= 0
        ]
        if positions:
            start = max(0, min(positions) - max_chars // 4)
            return content[start : start + max_chars].strip()
    return content[:max_chars].strip()


def _score_component(score: float | None) -> float:
    if score is None:
        return 0.0
    if score < 0:
        return 0.0
    if score > 1:
        return 0.15
    return score * 0.15


def _evidence_strength(
    *,
    has_citation: bool,
    has_date: bool,
    has_snippet: bool,
    score: float | None,
) -> float:
    strength = 0.0
    if has_citation:
        strength += 0.35
    if has_date:
        strength += 0.2
    if has_snippet:
        strength += 0.3
    strength += _score_component(score)
    return round(min(strength, 1.0), 6)


def _packet(result: Any, index: int, query: str | None) -> dict[str, Any]:
    citation_fields = _citation_fields(result)
    date_fields = _date_fields(result)
    snippet = _snippet(result, query)
    score = _float_value(_result_value(result, "score"))
    source_project = _string_value(_result_value(result, "source_project"))
    source_id = _string_value(_result_value(result, "source_id"))
    source_entity_type = _string_value(_result_value(result, "source_entity_type"))
    url = next((citation_fields[key] for key in _URL_KEYS if key in citation_fields), None)
    domain = _domain_from_url(url)
    has_citation = bool(citation_fields)
    warnings = [] if has_citation else ["missing_citation"]

    return {
        "rank": index + 1,
        "id": _result_id(result, index),
        "title": _string_value(_result_value(result, "title")),
        "source": _source(result),
        "source_project": source_project,
        "source_id": source_id,
        "source_entity_type": source_entity_type,
        "citation_fields": citation_fields,
        "url": url,
        "domain": domain,
        "date_fields": date_fields,
        "date": min(date_fields.values()) if date_fields else None,
        "snippet": snippet,
        "tags": _iter_strings(_result_value(result, "tags")),
        "score": score,
        "evidence_strength": _evidence_strength(
            has_citation=has_citation,
            has_date=bool(date_fields),
            has_snippet=snippet is not None,
            score=score,
        ),
        "missing_citation_warnings": warnings,
    }


def build_evidence_packets(
    results: Iterable[Any],
    query: str | None = None,
    limit: int | None = 10,
) -> list[dict[str, Any]]:
    """Normalize search results into deterministic answer evidence packets."""
    limit_value = _validate_limit(limit)
    packets = [_packet(result, index, query) for index, result in enumerate(list(results))]
    packets.sort(
        key=lambda packet: (
            -packet["evidence_strength"],
            -(packet["score"] if packet["score"] is not None else -1.0),
            str(packet["title"] or "").casefold(),
            str(packet["id"]),
            packet["rank"],
        )
    )
    for rank, packet in enumerate(packets, start=1):
        packet["rank"] = rank
    if limit_value is None:
        return packets
    return packets[:limit_value]
