"""Format compact citations for RAG/search results."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime
from typing import Any
from urllib.parse import urlsplit, urlunsplit

_MISSING = object()
_VALID_STYLES = frozenset({"inline", "footnote", "markdown"})

_TITLE_KEYS = ("title", "name", "label")
_SOURCE_KEYS = (
    "source",
    "source_name",
    "source_project",
    "publisher",
    "publication",
    "domain",
    "site",
    "source_id",
)
_URL_KEYS = (
    "url",
    "source_url",
    "canonical_url",
    "external_url",
    "link",
    "permalink",
    "uri",
)
_AUTHOR_KEYS = ("author", "authors", "creator", "byline")
_DATE_KEYS = (
    "published_at",
    "publication_date",
    "updated_at",
    "created_at",
    "timestamp",
    "date",
)
_ID_KEYS = ("id", "unit_id", "source_id", "doi", "arxiv_id", "pmid", "isbn")


def format_result_citations(
    results: Iterable[Any],
    *,
    style: str = "inline",
) -> list[str]:
    """Return deterministic short citations for result-like records.

    Results may be mappings, objects, ``(payload, score)``-style tuples, or
    wrappers containing nested ``metadata`` and/or ``unit`` mappings/objects.
    Duplicate source records are collapsed by normalized URL when present,
    otherwise by their formatted metadata.
    """
    style_value = _validate_style(style)
    citations: list[_Citation] = []
    seen: set[tuple[str, str]] = set()

    for index, result in enumerate(results):
        citation = _citation_from_result(result, index)
        key = citation.dedupe_key
        if key in seen:
            continue
        seen.add(key)
        citations.append(citation)

    if style_value == "inline":
        return [_format_inline(citation) for citation in citations]
    return [
        _format_footnote(citation, number)
        for number, citation in enumerate(citations, start=1)
    ]


def _validate_style(style: str) -> str:
    value = _inline_text(style).casefold()
    if value not in _VALID_STYLES:
        valid = ", ".join(sorted(_VALID_STYLES))
        raise ValueError(f"style must be one of: {valid}")
    if value == "markdown":
        return "footnote"
    return value


class _Citation:
    def __init__(
        self,
        *,
        title: str | None,
        source: str | None,
        url: str | None,
        author: str | None,
        date_text: str | None,
        fallback_id: str,
    ) -> None:
        self.title = title
        self.source = source
        self.url = url
        self.author = author
        self.date_text = date_text
        self.fallback_id = fallback_id

    @property
    def dedupe_key(self) -> tuple[str, str]:
        normalized_url = _normalize_url(self.url)
        if normalized_url is not None:
            return ("url", normalized_url)
        parts = [
            self.title,
            self.source,
            self.author,
            self.date_text,
            self.fallback_id,
        ]
        return ("metadata", "\x1f".join(part.casefold() for part in parts if part))


def _citation_from_result(result: Any, index: int) -> _Citation:
    return _Citation(
        title=_first_string(result, _TITLE_KEYS),
        source=_first_string(result, _SOURCE_KEYS),
        url=_first_string(result, _URL_KEYS),
        author=_format_author(_first_value(result, _AUTHOR_KEYS)),
        date_text=_format_date(_first_value(result, _DATE_KEYS)),
        fallback_id=_result_id(result, index),
    )


def _format_inline(citation: _Citation) -> str:
    label = _title_label(citation)
    details = [
        value
        for value in (citation.source, citation.author, citation.date_text, citation.url)
        if value and value != label
    ]
    if details:
        return f"[{label}; {', '.join(details)}]"
    return f"[{label}]"


def _format_footnote(citation: _Citation, number: int) -> str:
    label = _title_label(citation)
    details = [
        value
        for value in (citation.source, citation.author, citation.date_text, citation.url)
        if value and value != label
    ]
    body = f"{label}. {' '.join(details)}" if details else label
    return f"[^{number}]: {body}"


def _title_label(citation: _Citation) -> str:
    for value in (citation.title, citation.source, citation.url, citation.fallback_id):
        if value:
            return value
    return "Untitled source"


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


def _first_value(result: Any, keys: Iterable[str]) -> Any:
    for key in keys:
        value = _result_value(result, key)
        if value is not _MISSING and value is not None:
            return value
    return _MISSING


def _first_string(result: Any, keys: Iterable[str]) -> str | None:
    for key in keys:
        value = _string_value(_result_value(result, key))
        if value is not None:
            return value
    return None


def _result_id(result: Any, index: int) -> str:
    for key in _ID_KEYS:
        value = _string_value(_result_value(result, key))
        if value is not None:
            return value
    return f"result-{index + 1}"


def _format_author(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    if isinstance(value, Iterable) and not isinstance(value, str | bytes | Mapping):
        authors = [_string_value(author) for author in value]
        return ", ".join(author for author in authors if author) or None
    return _string_value(value)


def _format_date(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = _string_value(value)
    if text is None:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text).date().isoformat()
    except ValueError:
        try:
            return date.fromisoformat(text).isoformat()
        except ValueError:
            return text


def _normalize_url(value: str | None) -> str | None:
    text = _string_value(value)
    if text is None:
        return None
    parsed = urlsplit(text if "://" in text else f"https://{text}")
    netloc = parsed.netloc.casefold()
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit(
        (
            parsed.scheme.casefold() or "https",
            netloc,
            path,
            parsed.query,
            "",
        )
    )


def _string_value(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    return _inline_text(value) or None


def _inline_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())
