"""CSV inventory for Obsidian Markdown embeds."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "line_number", "target", "fragment", "alias", "is_image_embed"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_EMBED_RE = re.compile(r"!\[\[([^\[\]\n]+)\]\]")
_IMAGE_EXTENSIONS = {".apng", ".avif", ".gif", ".jpeg", ".jpg", ".png", ".svg", ".webp"}


def export_units_to_markdown_embed_inventory_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write one row per Obsidian embed outside fenced code."""
    unit_list = list(units)
    rows: list[dict[str, str | int]] = []
    for unit in unit_list:
        title = field_value(get(unit, "title") or metadata(unit).get("title"))
        rows.extend({"unit_id": unit_id(unit), "title": title, **row} for row in _embed_rows(_content(unit)))
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["target"]), sort_key(row["fragment"]), sort_key(row["alias"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _content(unit: Mapping[str, Any] | object) -> str:
    return str(get(unit, "content") or metadata(unit).get("content") or "")


def _embed_rows(content: str) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in _EMBED_RE.finditer(line):
            target, fragment, alias = _parts(match.group(1))
            rows.append({"line_number": line_number, "target": target, "fragment": fragment, "alias": alias, "is_image_embed": _is_image(target)})
    return rows


def _parts(value: str) -> tuple[str, str, str]:
    target_part, alias = (value.split("|", 1) + [""])[:2] if "|" in value else (value, "")
    target, fragment = (target_part.split("#", 1) + [""])[:2] if "#" in target_part else (target_part, "")
    return field_value(target), field_value(fragment), field_value(alias)


def _is_image(target: str) -> str:
    lowered = target.rsplit("?", 1)[0].casefold()
    return "true" if any(lowered.endswith(ext) for ext in _IMAGE_EXTENSIONS) else "false"
