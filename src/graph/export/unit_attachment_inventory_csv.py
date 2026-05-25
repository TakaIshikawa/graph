"""CSV export for per-unit attachment metadata."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, normalized_key, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = [
    "unit_id",
    "attachment_count",
    "attachment_paths",
    "attachment_extensions",
    "has_images",
    "has_documents",
    "missing_attachment_metadata",
]
_ATTACHMENT_KEYS = {"attachment", "attachments", "file", "files", "asset", "assets", "enclosure", "enclosures", "path", "paths"}
_PATH_KEYS = {"path", "url", "href", "uri", "src", "file", "filename", "name"}
_IMAGE_EXTENSIONS = {".apng", ".avif", ".gif", ".jpeg", ".jpg", ".png", ".svg", ".webp"}
_DOCUMENT_EXTENSIONS = {".csv", ".doc", ".docx", ".epub", ".md", ".pdf", ".ppt", ".pptx", ".rtf", ".txt", ".xls", ".xlsx"}


def export_units_to_attachment_inventory_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write one attachment inventory row per unit."""
    unit_list = list(units)
    rows = sorted((_row(unit) for unit in unit_list), key=lambda row: sort_key(row["unit_id"]))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def export_unit_attachment_inventory_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Backward-compatible alias for attachment inventory export."""
    return export_units_to_attachment_inventory_csv(units, path)


def _row(unit: Mapping[str, Any] | object) -> dict[str, str | int]:
    paths = sorted(set(_attachment_paths(unit)), key=sort_key)
    extensions = sorted({_extension(path) for path in paths if _extension(path)}, key=sort_key)
    return {
        "unit_id": unit_id(unit),
        "attachment_count": len(paths),
        "attachment_paths": "; ".join(paths),
        "attachment_extensions": "; ".join(extensions),
        "has_images": _flag(any(extension in _IMAGE_EXTENSIONS for extension in extensions)),
        "has_documents": _flag(any(extension in _DOCUMENT_EXTENSIONS for extension in extensions)),
        "missing_attachment_metadata": _flag(not _has_attachment_metadata(unit)),
    }


def _has_attachment_metadata(unit: Mapping[str, Any] | object) -> bool:
    return any(normalized_key(key) in _ATTACHMENT_KEYS for key in metadata(unit))


def _attachment_paths(unit: Mapping[str, Any] | object) -> list[str]:
    paths: list[str] = []
    for key, value in metadata(unit).items():
        if normalized_key(key) in _ATTACHMENT_KEYS:
            paths.extend(_extract_paths(value))
    return [path for path in paths if path]


def _extract_paths(value: object) -> list[str]:
    if value is None or isinstance(value, bytes):
        return []
    if isinstance(value, Mapping):
        direct = [field_value(value.get(key)) for key in _PATH_KEYS if field_value(value.get(key))]
        nested = [item for key, child in value.items() if normalized_key(key) not in _PATH_KEYS for item in _extract_paths(child)]
        return direct + nested
    if isinstance(value, list | tuple | set):
        return [item for child in value for item in _extract_paths(child)]
    return [field_value(value)]


def _extension(value: str) -> str:
    name = field_value(value).split("?", 1)[0].split("#", 1)[0].rstrip("/")
    suffix = Path(name).suffix.casefold()
    return suffix


def _flag(value: bool) -> str:
    return "true" if value else "false"
