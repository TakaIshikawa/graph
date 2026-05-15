"""CSV export for URLs attached to units."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["unit_id", "title", "link_count", "unique_domain_count", "domains", "urls"]
_URL_KEYS = {
    "url",
    "urls",
    "link",
    "links",
    "source_url",
    "external_url",
    "canonical_url",
    "web_url",
    "webpage_url",
}
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_link_inventory_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write per-unit URL counts and domain summaries."""
    unit_list = list(units)
    rows = _inventory_rows(unit_list)
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


def _inventory_rows(units: list[KnowledgeUnit]) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for unit in sorted(units, key=_unit_sort_key):
        urls = [_normalized_url(value) for value in _url_values(unit.metadata if isinstance(unit.metadata, Mapping) else {})]
        urls = [url for url in urls if url]
        domains = sorted({_domain(url) for url in urls if _domain(url)}, key=_sort_key)
        unique_urls = sorted(set(urls), key=_sort_key)
        rows.append(
            {
                "unit_id": _unit_id(unit),
                "title": _inline_text(unit.title),
                "link_count": len(urls),
                "unique_domain_count": len(domains),
                "domains": "; ".join(domains),
                "urls": "; ".join(unique_urls),
            }
        )
    return rows


def _url_values(value: object, *, key: str = "") -> list[object]:
    if isinstance(value, Mapping):
        values: list[object] = []
        for nested_key, nested_value in value.items():
            normalized = _normalized_key(nested_key)
            if normalized in _URL_KEYS:
                values.extend(_flat_values(nested_value))
            elif normalized in {"source", "sources"} or key in {"source", "sources"}:
                values.extend(_url_values(nested_value, key=normalized))
        return values
    return _flat_values(value)


def _flat_values(value: object) -> list[object]:
    if isinstance(value, list | tuple | set):
        return [item for entry in value for item in _flat_values(entry)]
    if isinstance(value, Mapping):
        return _url_values(value)
    return [value]


def _normalized_url(value: object) -> str:
    text = _inline_text(value)
    if not text:
        return ""
    parsed = urlparse(text)
    if not parsed.scheme and "." in text and "/" not in text:
        parsed = urlparse(f"https://{text}")
        text = parsed.geturl()
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc:
        return ""
    return text


def _domain(url: str) -> str:
    parsed = urlparse(url)
    return parsed.hostname.casefold() if parsed.hostname else ""


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _normalized_key(value: object) -> str:
    return _inline_text(value).casefold().replace("-", "_").replace(" ", "_")


def _unit_id(unit: KnowledgeUnit) -> str:
    return _inline_text(unit.id) or _inline_text(unit.source_id)


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[tuple[str, str], tuple[str, str]]:
    return (_sort_key(_unit_id(unit)), _sort_key(unit.title))
