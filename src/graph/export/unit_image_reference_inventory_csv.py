"""CSV export for image references in unit content and metadata."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, flatten_values, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "markdown_image_count", "html_image_count", "metadata_image_count", "missing_alt_count", "remote_image_count", "local_image_count"]
_MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_IMG_RE = re.compile(r"<img\b[^>]*\bsrc\s*=\s*(['\"]?)([^'\"\s>]+)\1[^>]*>", re.IGNORECASE)
_IMAGE_KEYS = {"image", "images", "thumbnail", "cover", "icon"}


def export_units_to_image_reference_inventory_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = sorted((_row(unit) for unit in unit_list), key=lambda row: sort_key(row["unit_id"]))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(unit: Mapping[str, Any] | object) -> dict[str, int | str]:
    content = "" if get(unit, "content") is None else str(get(unit, "content"))
    markdown = [(field_value(match.group(1)), field_value(match.group(2))) for match in _MARKDOWN_IMAGE_RE.finditer(content)]
    html = [field_value(match.group(2)) for match in _IMG_RE.finditer(content)]
    meta = [field_value(value) for key, raw in metadata(unit).items() if field_value(key).casefold() in _IMAGE_KEYS for value in flatten_values(raw) if field_value(value)]
    targets = [target for _alt, target in markdown] + html + meta
    return {
        "unit_id": unit_id(unit),
        "markdown_image_count": len(markdown),
        "html_image_count": len(html),
        "metadata_image_count": len(meta),
        "missing_alt_count": sum(1 for alt, _target in markdown if not alt),
        "remote_image_count": sum(1 for target in targets if target.casefold().startswith(("http://", "https://"))),
        "local_image_count": sum(1 for target in targets if not target.casefold().startswith(("http://", "https://"))),
    }
