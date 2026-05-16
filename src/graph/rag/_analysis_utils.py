"""Small normalization helpers for pure RAG analysis functions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime, timezone
from typing import Any
from urllib.parse import urlparse

from graph.rag.keywords import COMMON_STOPWORDS, TOKEN_RE

MISSING = object()


def payload(result: Any) -> Any:
    if isinstance(result, tuple) and result:
        return result[0]
    return result


def field_value(item: Any, key: str) -> Any:
    if item is MISSING or item is None:
        return MISSING
    if isinstance(item, Mapping):
        return item.get(key, MISSING)
    return getattr(item, key, MISSING)


def value(result: Any, key: str) -> Any:
    item = payload(result)
    direct = field_value(item, key)
    if direct is not MISSING and direct is not None:
        return direct

    metadata = field_value(item, "metadata")
    if isinstance(metadata, Mapping):
        nested = metadata.get(key, MISSING)
        if nested is not MISSING and nested is not None:
            return nested

    unit = field_value(item, "unit")
    if unit is not MISSING and unit is not None:
        nested = field_value(unit, key)
        if nested is not MISSING and nested is not None:
            return nested
        unit_metadata = field_value(unit, "metadata")
        if isinstance(unit_metadata, Mapping):
            nested = unit_metadata.get(key, MISSING)
            if nested is not MISSING and nested is not None:
                return nested

    if key == "score" and isinstance(result, tuple) and len(result) > 1:
        return result[1]
    return direct


def metadata(result: Any) -> Mapping[str, Any]:
    item = payload(result)
    meta = field_value(item, "metadata")
    if isinstance(meta, Mapping):
        return meta
    unit = field_value(item, "unit")
    if unit is not MISSING and unit is not None:
        unit_meta = field_value(unit, "metadata")
        if isinstance(unit_meta, Mapping):
            return unit_meta
    return {}


def string(value_: Any) -> str | None:
    if value_ is MISSING or value_ is None:
        return None
    if hasattr(value_, "value"):
        value_ = value_.value
    text = " ".join(str(value_).strip().split())
    return text or None


def number(value_: Any) -> float | None:
    if value_ is MISSING or value_ is None or isinstance(value_, bool):
        return None
    try:
        return float(value_)
    except (TypeError, ValueError):
        return None


def result_id(result: Any, index: int) -> str:
    for key in ("id", "result_id", "unit_id", "source_id"):
        text = string(value(result, key))
        if text is not None:
            return text
    return f"result-{index + 1}"


def source_id(result: Any) -> str | None:
    for key in ("source_id", "source", "source_name", "source_project", "domain"):
        text = string(value(result, key))
        if text is not None:
            return text
    return None


def content_text(result: Any) -> str:
    parts = []
    for key in ("title", "content", "text", "summary", "snippet"):
        text = string(value(result, key))
        if text:
            parts.append(text)
    return " ".join(parts)


def iter_strings(value_: Any) -> list[str]:
    if value_ is MISSING or value_ is None:
        return []
    if isinstance(value_, Mapping):
        return [item for nested in value_.values() for item in iter_strings(nested)]
    if isinstance(value_, Iterable) and not isinstance(value_, str | bytes):
        return [item for nested in value_ for item in iter_strings(nested)]
    text = string(value_)
    return [] if text is None else [text]


def tokens(text: Any, *, min_length: int = 3) -> set[str]:
    normalized = string(text)
    if normalized is None:
        return set()
    return {
        token
        for token in TOKEN_RE.findall(normalized.casefold())
        if len(token) >= min_length and token not in COMMON_STOPWORDS
    }


def ordered_terms(text: Any, *, min_length: int = 3) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []
    normalized = string(text)
    if normalized is None:
        return terms
    for token in TOKEN_RE.findall(normalized.casefold()):
        if len(token) < min_length or token in COMMON_STOPWORDS or token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return terms


def has_evidence(value_: Any) -> bool:
    if value_ is MISSING or value_ is None:
        return False
    if isinstance(value_, bool):
        return value_
    if isinstance(value_, int | float):
        return value_ > 0
    if isinstance(value_, Mapping):
        return any(has_evidence(nested) for nested in value_.values())
    if isinstance(value_, Iterable) and not isinstance(value_, str | bytes):
        return any(has_evidence(nested) for nested in value_)
    return string(value_) is not None


def any_present(result: Any, keys: Iterable[str]) -> bool:
    return any(has_evidence(value(result, key)) for key in keys)


def parse_date(value_: Any) -> date | None:
    text = string(value_)
    if text is None:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    for candidate in (text, text[:10]):
        try:
            parsed = datetime.fromisoformat(candidate)
            return parsed.date()
        except ValueError:
            pass
        try:
            return date.fromisoformat(candidate)
        except ValueError:
            pass
    return None


def result_date(result: Any) -> date | None:
    for key in (
        "published_at",
        "publication_date",
        "updated_at",
        "created_at",
        "timestamp",
        "date",
    ):
        parsed = parse_date(value(result, key))
        if parsed is not None:
            return parsed
    return None


def coerce_now(now: Any = None) -> date:
    if now is None:
        return datetime.now(timezone.utc).date()
    if isinstance(now, datetime):
        return now.date()
    if isinstance(now, date):
        return now
    parsed = parse_date(now)
    return parsed or datetime.now(timezone.utc).date()


def domain_for(result: Any) -> str | None:
    for key in ("domain", "source_domain"):
        text = string(value(result, key))
        if text is not None:
            return text.casefold()
    for key in ("url", "source_url", "canonical_url", "external_url", "link", "permalink", "uri"):
        text = string(value(result, key))
        if text is None:
            continue
        parsed = urlparse(text if "://" in text else f"https://{text}")
        host = parsed.netloc.casefold()
        if host.startswith("www."):
            host = host[4:]
        if host:
            return host
    return None


def rounded_ratio(count: int, total: int) -> float:
    return 0.0 if total == 0 else round(count / total, 4)
