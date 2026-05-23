"""CSV export for URL scheme inventory across source records."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_FIELDNAMES = ["source_id", "source_name", "scheme", "url_count", "sample_urls"]
_URL_RE = re.compile(r"\b(?:[A-Za-z][A-Za-z0-9+.-]*://|[A-Za-z][A-Za-z0-9+.-]*:)[^\s<>'\"]+")
_WHITESPACE_RE = re.compile(r"\s+")
_SAMPLE_LIMIT = 3


def export_source_url_scheme_inventory_csv(
    sources: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write per-source URL scheme counts from source fields and metadata."""
    source_list = list(sources)
    rows = _inventory_rows(source_list)
    text = _render_csv(rows)
    if path is None:
        return text
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {"path": str(output_path), "source_count": len(source_list), "rows_exported": len(rows), "bytes_written": output_path.stat().st_size}


def _inventory_rows(sources: list[Mapping[str, Any] | object]) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for source in sources:
        groups: dict[str, list[str]] = defaultdict(list)
        for url in _source_urls(source):
            groups[_scheme(url)].append(url)
        for scheme, urls in groups.items():
            rows.append(
                {
                    "source_id": _source_id(source),
                    "source_name": _source_name(source),
                    "scheme": scheme,
                    "url_count": len(urls),
                    "sample_urls": "; ".join(urls[:_SAMPLE_LIMIT]),
                }
            )
    return sorted(rows, key=lambda row: (_sort_key(row["source_id"]), _sort_key(row["scheme"])))


def _source_urls(source: Mapping[str, Any] | object) -> list[str]:
    values = [_get(source, key) for key in ("url", "source_url", "external_url", "homepage", "link")]
    metadata = _get(source, "metadata")
    if isinstance(metadata, Mapping):
        values.extend(_metadata_values(metadata))
    urls: list[str] = []
    for value in values:
        urls.extend(_urls_from_text(value))
    return urls


def _urls_from_text(value: object) -> list[str]:
    if value is None or isinstance(value, bytes):
        return []
    urls: list[str] = []
    for candidate in _URL_RE.findall(str(value)):
        url = candidate.rstrip(".,);]")
        parsed = urlparse(url)
        if parsed.scheme and (parsed.netloc or parsed.path):
            urls.append(url)
    text = _field_value(value)
    if text and "://" not in text and re.match(r"^[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:/|$)", text):
        urls.append(text)
    return urls


def _scheme(url: str) -> str:
    parsed = urlparse(url)
    return parsed.scheme.casefold() if parsed.scheme else "missing"


def _metadata_values(value: object) -> list[object]:
    if isinstance(value, Mapping):
        return [item for child in value.values() for item in _metadata_values(child)]
    if isinstance(value, list | tuple | set):
        return [item for child in value for item in _metadata_values(child)]
    return [value]


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _source_id(source: Mapping[str, Any] | object) -> str:
    return _field_value(_get(source, "id")) or _field_value(_get(source, "source_id"))


def _source_name(source: Mapping[str, Any] | object) -> str:
    return _field_value(_get(source, "name")) or _field_value(_get(source, "title"))


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
