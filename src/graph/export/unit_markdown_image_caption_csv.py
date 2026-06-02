"""CSV export for Markdown image captions."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "image_target", "line_number", "caption", "caption_style"]
_IMAGE_RE = re.compile(r"!\[[^\]]*]\((?P<target>[^)\s]+)(?:\s+\"[^\"]*\")?\)")
_ITALIC_RE = re.compile(r"^\s*(?:\*(?P<star>[^*]+)\*|_(?P<underscore>[^_]+)_)\s*$")
_FIGURE_RE = re.compile(r"^\s*Figure:\s*(?P<caption>.+?)\s*$", re.IGNORECASE)
_HTML_IMG_RE = re.compile(r"<img\b[^>]*\bsrc=[\"'](?P<src>[^\"']+)[\"'][^>]*>", re.IGNORECASE)
_FIGCAPTION_RE = re.compile(r"<figcaption\b[^>]*>(?P<caption>.*?)</figcaption>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")


def export_units_to_markdown_image_caption_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["image_target"]), sort_key(row["caption"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    lines = str(get(unit, "content") or "").splitlines()
    rows: list[dict[str, str | int]] = []
    seen: set[tuple[str, int, str, str]] = set()

    for index, line in enumerate(lines):
        image = _IMAGE_RE.search(line)
        if image:
            target = field_value(image.group("target"))
            for caption_line, caption, style in _adjacent_markdown_captions(lines, index):
                _append_row(rows, seen, uid, title, target, caption_line, caption, style)

        if "<figure" in line.casefold():
            block, start_line = _html_figure_block(lines, index)
            if block:
                html_image = _HTML_IMG_RE.search(block)
                caption_match = _FIGCAPTION_RE.search(block)
                if html_image and caption_match:
                    _append_row(rows, seen, uid, title, html_image.group("src"), start_line, _html_text(caption_match.group("caption")), "html_figcaption")

    return rows


def _adjacent_markdown_captions(lines: list[str], image_index: int) -> list[tuple[int, str, str]]:
    captions: list[tuple[int, str, str]] = []
    for caption_index in (image_index - 1, image_index + 1):
        if caption_index < 0 or caption_index >= len(lines):
            continue
        line = lines[caption_index]
        if italic := _ITALIC_RE.match(line):
            captions.append((caption_index + 1, field_value(italic.group("star") or italic.group("underscore")), "italic"))
        elif figure := _FIGURE_RE.match(line):
            captions.append((caption_index + 1, field_value(figure.group("caption")), "figure"))
    return captions


def _html_figure_block(lines: list[str], start_index: int) -> tuple[str, int] | tuple[None, int]:
    block_lines: list[str] = []
    for index in range(start_index, len(lines)):
        block_lines.append(lines[index])
        if "</figure>" in lines[index].casefold():
            return "\n".join(block_lines), start_index + 1
    return None, start_index + 1


def _html_text(value: str) -> str:
    return field_value(_TAG_RE.sub(" ", value))


def _append_row(
    rows: list[dict[str, str | int]],
    seen: set[tuple[str, int, str, str]],
    uid: str,
    title: str,
    target: str,
    line_number: int,
    caption: str,
    style: str,
) -> None:
    key = (target, line_number, caption, style)
    if key in seen or not caption:
        return
    seen.add(key)
    rows.append({"unit_id": uid, "title": title, "image_target": field_value(target), "line_number": line_number, "caption": caption[:240], "caption_style": style})
