"""Analyze result format coverage for RAG result payloads."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_FORMAT_ORDER = ("html", "pdf", "markdown", "transcript", "dataset", "image", "unknown")
_MIME_FORMATS = {
    "text/html": "html",
    "application/xhtml+xml": "html",
    "application/pdf": "pdf",
    "text/markdown": "markdown",
    "text/x-markdown": "markdown",
    "text/csv": "dataset",
    "application/json": "dataset",
    "application/vnd.ms-excel": "dataset",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "dataset",
    "text/vtt": "transcript",
    "application/x-subrip": "transcript",
}
_EXTENSION_FORMATS = {
    ".html": "html",
    ".htm": "html",
    ".pdf": "pdf",
    ".md": "markdown",
    ".markdown": "markdown",
    ".csv": "dataset",
    ".json": "dataset",
    ".jsonl": "dataset",
    ".xlsx": "dataset",
    ".xls": "dataset",
    ".vtt": "transcript",
    ".srt": "transcript",
    ".txt": "transcript",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".gif": "image",
    ".webp": "image",
    ".svg": "image",
}


def analyze_result_format_coverage(results: Iterable[Mapping[str, Any] | object]) -> dict[str, Any]:
    """Return deterministic counts, shares, and dominant format for results."""
    result_list = list(results)
    counts = Counter(_format(result) for result in result_list)
    total = len(result_list)
    formats = [
        {"format": label, "count": counts[label], "share": round(counts[label] / total, 4) if total else 0.0}
        for label in _FORMAT_ORDER
        if counts[label] or total == 0
    ]
    dominant = max(_FORMAT_ORDER, key=lambda label: (counts[label], -_FORMAT_ORDER.index(label))) if total else "unknown"
    return {"result_count": total, "formats": formats, "dominant_format": dominant}


def _format(result: Mapping[str, Any] | object) -> str:
    for key in ("mime_type", "content_type"):
        value = _text(_get(result, key))
        label = _from_mime(value)
        if label:
            return label
    metadata = _get(result, "metadata")
    if isinstance(metadata, Mapping):
        for key in ("mime_type", "content_type", "format", "file_extension", "extension", "url"):
            label = _label(_text(metadata.get(key)), key)
            if label != "unknown":
                return label
    for key in ("format", "file_extension", "extension", "url"):
        label = _label(_text(_get(result, key)), key)
        if label != "unknown":
            return label
    return "unknown"


def _label(value: str, key: str) -> str:
    if not value:
        return "unknown"
    mime = _from_mime(value)
    if mime:
        return mime
    compact = value.casefold().strip().lstrip(".")
    if compact in _FORMAT_ORDER:
        return compact
    suffix = Path(urlparse(value).path if key == "url" else value).suffix.casefold()
    return _EXTENSION_FORMATS.get(suffix, _EXTENSION_FORMATS.get(f".{compact}", "unknown"))


def _from_mime(value: str) -> str:
    mime = value.split(";", 1)[0].casefold().strip()
    if mime.startswith("image/"):
        return "image"
    return _MIME_FORMATS.get(mime, "")


def _get(value: object, key: str) -> object:
    return value.get(key) if isinstance(value, Mapping) else getattr(value, key, None)


def _text(value: object) -> str:
    return "" if value is None else " ".join(str(value).strip().split())
