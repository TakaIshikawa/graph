"""Netscape bookmarks HTML export helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, overload

from graph.types.models import KnowledgeUnit

URL_METADATA_KEYS = ("source_url", "external_url", "url")
DATE_METADATA_KEYS = ("add_date", "ADD_DATE", "created", "created_at", "date", "published_at")


@overload
def export_units_to_bookmarks_html(
    units: KnowledgeUnit | Iterable[KnowledgeUnit],
    path: None = None,
) -> str: ...


@overload
def export_units_to_bookmarks_html(
    units: KnowledgeUnit | Iterable[KnowledgeUnit],
    path: str | Path,
) -> dict[str, Any]: ...


def export_units_to_bookmarks_html(
    units: KnowledgeUnit | Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write units with URLs as Netscape bookmarks HTML."""
    unit_list = [units] if isinstance(units, KnowledgeUnit) else list(units)
    bookmarks = [
        (url, unit)
        for unit in sorted(unit_list, key=_unit_key)
        if (url := _first_text(unit.metadata, URL_METADATA_KEYS))
    ]

    lines = [
        "<!DOCTYPE NETSCAPE-Bookmark-file-1>",
        '<META HTTP-EQUIV="Content-Type" CONTENT="text/html; charset=UTF-8">',
        "<TITLE>Bookmarks</TITLE>",
        "<H1>Bookmarks</H1>",
        "<DL><p>",
    ]
    for url, unit in bookmarks:
        lines.append(f"    <DT>{_anchor(unit, url)}")
    lines.append("</DL><p>")
    text = "\n".join(lines) + "\n"

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return {
        "path": str(output_path),
        "unit_count": len(bookmarks),
        "skipped_count": len(unit_list) - len(bookmarks),
        "bytes_written": output_path.stat().st_size,
    }


def _anchor(unit: KnowledgeUnit, url: str) -> str:
    attrs = [f'HREF="{escape(url, quote=True)}"']
    add_date = _add_date(unit.metadata)
    if add_date:
        attrs.append(f'ADD_DATE="{escape(add_date, quote=True)}"')
    tags = sorted(tag for tag in (_clean_text(tag) for tag in unit.tags) if tag)
    if tags:
        attrs.append(f'TAGS="{escape(",".join(tags), quote=True)}"')
    title = _clean_text(unit.title) or url
    return f"<A {' '.join(attrs)}>{escape(title)}</A>"


def _add_date(metadata: Mapping[str, Any]) -> str:
    value = _first_value(metadata, DATE_METADATA_KEYS)
    if isinstance(value, datetime):
        return str(int(value.timestamp()))
    if isinstance(value, date):
        midnight = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
        return str(int(midnight.timestamp()))
    text = _clean_text(value)
    return text


def _first_text(metadata: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    value = _first_value(metadata, keys)
    return _clean_text(value)


def _first_value(metadata: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in metadata:
            return metadata.get(key)
    return None


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\r\n", "\n").replace("\r", "\n").split())


def _unit_key(unit: KnowledgeUnit) -> tuple[str, str, str]:
    return (str(unit.source_project or ""), str(unit.source_id or ""), str(unit.title or ""))
