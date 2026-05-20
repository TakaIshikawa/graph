"""CSV export for unit URL quality issues."""

from __future__ import annotations

import csv
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["source", "unit_id", "title", "url", "issue", "detail"]
_URL_METADATA_KEYS = ("url", "uri", "link", "permalink", "source_url", "external_url", "canonical_url", "web_url")
_SUPPORTED_SCHEMES = {"http", "https"}
_LONG_URL_LENGTH = 2048
_URL_RE = re.compile(r"""(?i)\b(?:https?://|ftp://|www\.|[a-z0-9.-]+\.[a-z]{2,})(?:[^\s<>"']*)""")
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_url_quality_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write URL quality findings without network access."""
    unit_list = list(units)
    rows = _quality_rows(unit_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _quality_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for unit in units:
        urls = _unit_urls(unit)
        counts = Counter(urls)
        emitted: set[tuple[str, str, str]] = set()
        for url in sorted(counts, key=_sort_key):
            for issue, detail in _url_issues(url, counts[url]):
                key = (url, issue, detail)
                if key in emitted:
                    continue
                emitted.add(key)
                rows.append(
                    {
                        "source": _field_value(_get(unit, "source_project")) or "Unknown",
                        "unit_id": _unit_id(unit),
                        "title": _field_value(_get(unit, "title")),
                        "url": url,
                        "issue": issue,
                        "detail": detail,
                    }
                )
    return sorted(
        rows,
        key=lambda row: (
            _sort_key(row["source"]),
            _sort_key(row["unit_id"]),
            _sort_key(row["url"]),
            _sort_key(row["issue"]),
        ),
    )


def _url_issues(url: str, count: int) -> list[tuple[str, str]]:
    issues: list[tuple[str, str]] = []
    stripped = url.strip()
    if stripped != url or re.search(r"\s", url):
        issues.append(("whitespace", "URL contains leading, trailing, or embedded whitespace"))

    parsed = urlparse(stripped)
    if not parsed.scheme:
        issues.append(("missing_scheme", "URL has no scheme"))
    elif parsed.scheme.casefold() not in _SUPPORTED_SCHEMES:
        issues.append(("unsupported_scheme", parsed.scheme.casefold()))

    if parsed.scheme and not parsed.netloc:
        issues.append(("malformed", "URL has a scheme but no network location"))
    elif parsed.scheme in _SUPPORTED_SCHEMES and ("." not in parsed.netloc and parsed.netloc not in {"localhost"}):
        issues.append(("malformed", "URL host does not look fully qualified"))

    if len(stripped) > _LONG_URL_LENGTH:
        issues.append(("suspiciously_long", f"{len(stripped)} characters"))
    if count > 1:
        issues.append(("duplicate", f"{count} occurrences in unit"))
    return issues


def _unit_urls(unit: KnowledgeUnit | Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    metadata = _metadata(unit)
    for key in _URL_METADATA_KEYS:
        values.extend(_urls_from_value(_casefold_get(metadata, key)))
        values.extend(_urls_from_value(_get(unit, key)))
    values.extend(_urls_from_text(_field_value(_get(unit, "content"))))
    return [url for url in values if url]


def _urls_from_value(value: object) -> list[str]:
    if value is None or isinstance(value, bytes):
        return []
    if isinstance(value, Mapping):
        return [url for item in value.values() for url in _urls_from_value(item)]
    if isinstance(value, list | tuple | set):
        return [url for item in value for url in _urls_from_value(item)]
    text = str(value)
    return [text] if text.strip() else []


def _urls_from_text(text: str) -> list[str]:
    return [match.group(0).rstrip(").,;]") for match in _URL_RE.finditer(text)]


def _metadata(unit: KnowledgeUnit | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _get(unit, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _casefold_get(mapping: Mapping[str, Any], key: str) -> object:
    for candidate_key, value in mapping.items():
        if _field_value(candidate_key).casefold() == key.casefold():
            return value
    return None


def _render_csv(rows: list[dict[str, str]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _unit_id(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    return _field_value(_get(unit, "id")) or _field_value(_get(unit, "source_id"))


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
