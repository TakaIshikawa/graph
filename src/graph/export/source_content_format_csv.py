"""CSV export for source content format summaries."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["source_project", "source_entity_type", "content_format", "unit_count", "average_content_chars", "representative_unit_ids"]
_UNKNOWN = "Unknown"
_ATTACHMENT_KEYS = ("attachment", "attachments", "file", "files", "filepath", "filename", "mime_type", "binary")
_URL_RE = re.compile(r"^https?://\S+$", re.IGNORECASE)
_HTML_RE = re.compile(r"<[a-z][\s\S]*?>", re.IGNORECASE)
_MARKDOWN_RE = re.compile(r"(^#{1,6}\s+)|(\[[^\]]+\]\([^)]+\))|(```)|(^[-*]\s+)", re.MULTILINE)
_WHITESPACE_RE = re.compile(r"\s+")


def export_source_content_format_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | Any | None = None,
) -> str | dict[str, Any]:
    """Return or write content format summaries by source project and entity type."""
    unit_list = list(units)
    rows = _format_rows(unit_list)
    text = _render_csv(rows)
    if path is None:
        return text
    return _write_output(path, text, {"unit_count": len(unit_list), "rows_exported": len(rows)})


def _format_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, str | int]]:
    groups: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(lambda: {"unit_ids": set(), "chars": []})
    for unit in units:
        content = _field_value(_get(unit, "content"))
        key = (_unit_source(unit), _unit_source_type(unit), _content_format(unit, content))
        groups[key]["chars"].append(len(content))
        if _unit_id(unit):
            groups[key]["unit_ids"].add(_unit_id(unit))

    rows: list[dict[str, str | int]] = []
    for (source, entity_type, content_format), group in groups.items():
        rows.append(
            {
                "source_project": source,
                "source_entity_type": entity_type,
                "content_format": content_format,
                "unit_count": len(group["chars"]),
                "average_content_chars": f"{sum(group['chars']) / len(group['chars']):.2f}",
                "representative_unit_ids": _joined(group["unit_ids"]),
            }
        )
    return sorted(rows, key=lambda row: (_sort_key(row["source_project"]), _sort_key(row["source_entity_type"]), _sort_key(row["content_format"])))


def _content_format(unit: KnowledgeUnit | Mapping[str, Any], content: str) -> str:
    metadata = _metadata(unit)
    if any(_truthy(metadata.get(key)) for key in _ATTACHMENT_KEYS):
        return "binary_attachment_reference"
    if not content:
        return "empty"
    if _URL_RE.fullmatch(content):
        return "url_only"
    if _HTML_RE.search(content):
        return "html_like"
    if _json_like(content):
        return "json_like"
    if _MARKDOWN_RE.search(content):
        return "markdown_like"
    return "plain_text"


def _json_like(content: str) -> bool:
    stripped = content.strip()
    if not ((stripped.startswith("{") and stripped.endswith("}")) or (stripped.startswith("[") and stripped.endswith("]"))):
        return False
    try:
        json.loads(stripped)
    except ValueError:
        return False
    return True


def _truthy(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return bool(_field_value(value))
    if isinstance(value, Mapping | list | tuple | set):
        return bool(value)
    return True


def _unit_id(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    return _field_value(_get(unit, "id")) or _field_value(_get(unit, "source_id"))


def _unit_source(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    return _field_value(_get(unit, "source_project")) or _UNKNOWN


def _unit_source_type(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    return _field_value(_get(unit, "source_entity_type")) or _UNKNOWN


def _metadata(unit: KnowledgeUnit | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _get(unit, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _joined(values: Iterable[object]) -> str:
    return "; ".join(sorted({_field_value(value) for value in values if _field_value(value)}, key=_sort_key))


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _write_output(path: str | Path | Any, text: str, stats: dict[str, Any]) -> dict[str, Any]:
    if hasattr(path, "write") and not isinstance(path, str | Path):
        written = path.write(text)
        stats["bytes_written"] = len(text.encode("utf-8")) if written is None else written
        return stats
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    stats["path"] = str(output_path)
    stats["bytes_written"] = output_path.stat().st_size
    return stats


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
