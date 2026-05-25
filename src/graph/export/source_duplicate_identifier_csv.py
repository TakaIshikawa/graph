"""CSV export for duplicate source identifiers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, source_id, write_csv

_FIELDNAMES = ["identifier_key", "identifier_value", "source_count", "source_ids", "source_names", "collision_severity"]
_IDENTIFIER_KEYS = ("url", "canonical_url", "doi", "isbn", "external_id", "guid", "source_id")


def export_source_duplicate_identifier_csv(
    sources: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write source identifier values shared by multiple sources."""
    source_list = list(sources)
    rows = _rows(source_list)
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "source_count": len(source_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(sources: list[Mapping[str, Any] | object]) -> list[dict[str, str | int]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any] | object]] = defaultdict(list)
    for source in sources:
        data = metadata(source)
        for key in _IDENTIFIER_KEYS:
            raw = get(source, key) or data.get(key)
            normalized = _normalize_identifier(key, raw)
            if normalized:
                groups[(key, normalized)].append(source)

    rows = []
    for (key, value), group in groups.items():
        ids = sorted({source_id(source) for source in group if source_id(source)}, key=sort_key)
        if len(ids) <= 1:
            continue
        names = sorted({_source_name(source) for source in group if _source_name(source)}, key=sort_key)
        rows.append(
            {
                "identifier_key": key,
                "identifier_value": value,
                "source_count": len(ids),
                "source_ids": "; ".join(ids),
                "source_names": "; ".join(names),
                "collision_severity": "same_name" if len(names) <= 1 else "conflicting_names",
            }
        )
    return sorted(rows, key=lambda row: (sort_key(row["identifier_key"]), sort_key(row["identifier_value"])))


def _normalize_identifier(key: str, value: object) -> str:
    text = field_value(value)
    if not text:
        return ""
    if key in {"url", "canonical_url"}:
        parsed = urlparse(text if "://" in text else f"https://{text}")
        scheme = parsed.scheme.lower() or "https"
        host = parsed.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        path = parsed.path.rstrip("/")
        query = urlencode(sorted(parse_qsl(parsed.query, keep_blank_values=True)))
        return urlunparse((scheme, host, path, "", query, "")).lower()
    if key == "doi":
        return text.lower().removeprefix("https://doi.org/").removeprefix("http://doi.org/").removeprefix("doi:").strip()
    if key == "isbn":
        return "".join(ch for ch in text if ch.isdigit() or ch.upper() == "X").upper()
    return text.casefold()


def _source_name(source: Mapping[str, Any] | object) -> str:
    return field_value(get(source, "name") or get(source, "title") or metadata(source).get("name"))
