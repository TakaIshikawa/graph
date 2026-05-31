"""CSV export for source redirect hint inventory."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, source_id, write_csv

_FIELDNAMES = ["source_id", "name", "original_url", "final_url", "redirect_count", "status_code"]
_ORIGINAL_KEYS = ("original_url", "url", "source_url", "fetch_url")
_FINAL_KEYS = ("final_url", "redirected_url", "redirect_url", "redirect_target", "canonical_url")


def export_sources_to_redirect_hint_inventory_csv(
    sources: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write one row per source with redirect metadata."""
    source_list = list(sources)
    rows = [_row(source) for source in source_list]
    rows = sorted((row for row in rows if _has_redirect_hint(row)), key=lambda row: sort_key(row["source_id"]))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "source_count": len(source_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(source: Mapping[str, Any] | object) -> dict[str, str]:
    data = metadata(source)
    original = _first(source, data, _ORIGINAL_KEYS)
    final = _first(source, data, _FINAL_KEYS) or original
    return {
        "source_id": source_id(source),
        "name": field_value(get(source, "name") or get(source, "title") or data.get("name") or data.get("title")),
        "original_url": original,
        "final_url": final,
        "redirect_count": field_value(get(source, "redirect_count") if get(source, "redirect_count") is not None else data.get("redirect_count")),
        "status_code": field_value(get(source, "status_code") if get(source, "status_code") is not None else data.get("status_code")),
    }


def _first(source: Mapping[str, Any] | object, data: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = field_value(get(source, key))
        if value:
            return value
    for key in keys:
        value = field_value(data.get(key))
        if value:
            return value
    return ""


def _has_redirect_hint(row: Mapping[str, str]) -> bool:
    count = row["redirect_count"]
    if count and count != "0":
        return True
    return bool(row["original_url"] and row["final_url"] and row["original_url"] != row["final_url"])
