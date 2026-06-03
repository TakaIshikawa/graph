"""CSV export for embedded Obsidian-style Markdown wikilinks."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "target", "alias", "heading", "media_type_hint", "line_number", "column_number"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_EMBED_WIKILINK_RE = re.compile(r"(?<!\\)!\[\[([^\[\]\n]+)\]\]")
_MEDIA_TYPE_HINTS = {
    ".apng": "image",
    ".avif": "image",
    ".gif": "image",
    ".jpeg": "image",
    ".jpg": "image",
    ".png": "image",
    ".svg": "image",
    ".webp": "image",
    ".bmp": "image",
    ".tif": "image",
    ".tiff": "image",
    ".pdf": "pdf",
    ".mp3": "audio",
    ".m4a": "audio",
    ".ogg": "audio",
    ".wav": "audio",
    ".flac": "audio",
    ".mp4": "video",
    ".mov": "video",
    ".mkv": "video",
    ".webm": "video",
    ".avi": "video",
}


def export_units_to_markdown_embed_wikilink_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write one row per embedded wikilink outside fenced code."""
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), int(row["column_number"]), sort_key(row["target"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    uid = unit_id(unit)
    rows: list[dict[str, str | int]] = []
    in_fence = False
    for line_number, line in enumerate(str(get(unit, "content") or metadata(unit).get("content") or "").splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in _EMBED_WIKILINK_RE.finditer(line):
            parsed = _parse_embed(match.group(1))
            if parsed is None:
                continue
            target, heading, alias = parsed
            rows.append(
                {
                    "unit_id": uid,
                    "target": target,
                    "alias": alias,
                    "heading": heading,
                    "media_type_hint": _media_type_hint(target),
                    "line_number": line_number,
                    "column_number": match.start() + 1,
                }
            )
    return rows


def _parse_embed(raw: str) -> tuple[str, str, str] | None:
    if "|" in raw:
        destination, alias = raw.split("|", 1)
    else:
        destination, alias = raw, ""
    destination = destination.strip()
    alias = alias.strip()
    if not destination or not destination.strip("#") or "|" in alias:
        return None
    if "#" in destination:
        target, heading = destination.split("#", 1)
    else:
        target, heading = destination, ""
    target = field_value(target)
    heading = field_value(heading)
    if not target:
        return None
    return target, heading, field_value(alias)


def _media_type_hint(target: str) -> str:
    path = target.split("?", 1)[0].split("#", 1)[0].casefold()
    suffix = Path(path).suffix
    return _MEDIA_TYPE_HINTS.get(suffix, "")
