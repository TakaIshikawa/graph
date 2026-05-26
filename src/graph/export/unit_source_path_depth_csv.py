"""CSV export for source path depth signals."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import urlparse

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "source_path", "depth", "basename", "extension", "is_url"]
_PATH_KEYS = ("path", "source_path", "file_path", "filename", "source_url")


def export_units_to_source_path_depth_csv(units: Iterable[Mapping[str, Any] | object], path: str | PurePosixPath | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = sorted((_row(unit) for unit in unit_list), key=lambda row: sort_key(row["unit_id"]))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(unit: Mapping[str, Any] | object) -> dict[str, str | int]:
    source_path = _source_path(unit)
    parsed = urlparse(source_path)
    is_url = parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    path_text = parsed.path if is_url else source_path
    parts = [part for part in path_text.replace("\\", "/").strip("/").split("/") if part]
    basename = parts[-1] if parts else ""
    suffix = PurePosixPath(basename).suffix or PureWindowsPath(basename).suffix
    return {
        "unit_id": unit_id(unit),
        "source_path": source_path,
        "depth": max(len(parts) - 1, 0),
        "basename": basename,
        "extension": suffix.casefold(),
        "is_url": str(is_url).lower(),
    }


def _source_path(unit: Mapping[str, Any] | object) -> str:
    meta = metadata(unit)
    for key in _PATH_KEYS:
        text = field_value(get(unit, key)) or field_value(meta.get(key))
        if text:
            return text
    return ""
