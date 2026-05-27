"""CSV export for local file references in Markdown units."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "reference", "path", "scheme", "extension", "line_number", "exists"]
_MD_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_FILE_URI_RE = re.compile(r"\bfile://[^\s<>()\[\]\"']+")


def export_units_to_local_file_reference_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None, *, base_path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows: list[dict[str, str | int]] = []
    base = Path(base_path) if base_path is not None else None
    for unit in unit_list:
        rows.extend(_rows(unit, base))
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["path"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object, base: Path | None) -> list[dict[str, str | int]]:
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    rows: list[dict[str, str | int]] = []
    for line_number, line in enumerate(str(get(unit, "content") or "").splitlines(), start=1):
        targets = list(dict.fromkeys([match.group(1) for match in _MD_LINK_RE.finditer(line)] + [match.group(0) for match in _FILE_URI_RE.finditer(line)]))
        for target in targets:
            parsed = _local_target(target)
            if not parsed:
                continue
            scheme, file_path = parsed
            rows.append(
                {
                    "unit_id": unit_id(unit),
                    "title": title,
                    "reference": target,
                    "path": file_path,
                    "scheme": scheme,
                    "extension": Path(file_path).suffix.casefold().lstrip("."),
                    "line_number": line_number,
                    "exists": _exists(file_path, base),
                }
            )
    return rows


def _local_target(target: str) -> tuple[str, str] | None:
    clean = target.strip().split("#", 1)[0].split("?", 1)[0]
    if not clean:
        return None
    parsed = urlparse(clean)
    scheme = parsed.scheme.casefold()
    if scheme in {"http", "https", "mailto", "tel"}:
        return None
    if scheme == "file":
        return ("file", unquote(parsed.path))
    if scheme:
        return None
    return ("", unquote(clean))


def _exists(file_path: str, base: Path | None) -> str:
    if base is None:
        return ""
    candidate = Path(file_path)
    if not candidate.is_absolute():
        candidate = base / candidate
    try:
        return "true" if candidate.exists() else "false"
    except OSError:
        return "false"
