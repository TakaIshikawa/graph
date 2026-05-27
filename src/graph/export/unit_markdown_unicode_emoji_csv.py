"""CSV export for Unicode emoji characters in unit content."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "emoji", "count", "first_line_number", "contexts"]
_VARIATION_SELECTOR = "\ufe0f"


def export_unit_markdown_unicode_emoji_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    unit_list = list(units)
    rows: list[dict[str, str | int]] = []
    for unit in unit_list:
        rows.extend(_rows(unit))
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["first_line_number"]), sort_key(row["emoji"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    seen: dict[str, dict[str, Any]] = {}
    contexts: dict[str, list[str]] = defaultdict(list)
    for line_number, line in enumerate(str(get(unit, "content") or "").splitlines(), start=1):
        line_emojis = _emojis(line)
        for emoji in line_emojis:
            if emoji not in seen:
                seen[emoji] = {"count": 0, "first_line_number": line_number}
            seen[emoji]["count"] += 1
        for emoji in sorted(set(line_emojis), key=sort_key):
            context = field_value(line)
            if context and context not in contexts[emoji]:
                contexts[emoji].append(context)
    return [
        {
            "unit_id": uid,
            "title": title,
            "emoji": emoji,
            "count": values["count"],
            "first_line_number": values["first_line_number"],
            "contexts": "; ".join(contexts[emoji]),
        }
        for emoji, values in seen.items()
    ]


def _emojis(text: str) -> list[str]:
    emojis: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if _is_emoji(char):
            emoji = char
            if index + 1 < len(text) and text[index + 1] == _VARIATION_SELECTOR:
                emoji += _VARIATION_SELECTOR
                index += 1
            emojis.append(emoji)
        index += 1
    return emojis


def _is_emoji(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x1F000 <= codepoint <= 0x1FAFF
        or 0x2600 <= codepoint <= 0x27BF
    )
