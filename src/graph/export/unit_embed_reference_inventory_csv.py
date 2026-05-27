"""CSV export for Obsidian-style embed references in unit content."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "embed_count", "note_embed_count", "file_embed_count", "section_embed_count", "alias_embed_count", "distinct_targets"]
_EMBED_RE = re.compile(r"!\[\[([^\]]*)\]\]")


def export_units_to_embed_reference_inventory_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = sorted((_row(unit) for unit in unit_list), key=lambda row: sort_key(row["unit_id"]))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(unit: Mapping[str, Any] | object) -> dict[str, int | str]:
    embeds = [_parse(match.group(1)) for match in _EMBED_RE.finditer("" if get(unit, "content") is None else str(get(unit, "content")))]
    target_by_key: dict[str, str] = {}
    for target, _section, _alias, _file in embeds:
        if target:
            target_by_key.setdefault(target.casefold(), target)
    targets = sorted(target_by_key.values(), key=sort_key)
    return {
        "unit_id": unit_id(unit),
        "embed_count": len(embeds),
        "note_embed_count": sum(1 for _target, _section, _alias, is_file in embeds if not is_file),
        "file_embed_count": sum(1 for _target, _section, _alias, is_file in embeds if is_file),
        "section_embed_count": sum(1 for _target, section, _alias, _is_file in embeds if section),
        "alias_embed_count": sum(1 for _target, _section, alias, _is_file in embeds if alias),
        "distinct_targets": "; ".join(targets),
    }


def _parse(raw: str) -> tuple[str, str, str, bool]:
    target_part, separator, alias = raw.partition("|")
    target, section_sep, section = target_part.partition("#")
    target = field_value(target)
    extension = target.rsplit(".", 1)[1].casefold() if "." in target.rsplit("/", 1)[-1] else ""
    return target, field_value(section) if section_sep else "", field_value(alias) if separator else "", bool(extension)
