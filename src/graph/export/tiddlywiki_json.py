"""TiddlyWiki JSON export helpers for knowledge units."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, overload

from pydantic import BaseModel

from graph.types.models import KnowledgeUnit


@overload
def export_units_to_tiddlywiki_json(
    units: Iterable[KnowledgeUnit],
    path: None = None,
) -> str: ...


@overload
def export_units_to_tiddlywiki_json(
    units: Iterable[KnowledgeUnit],
    path: str | Path,
) -> dict[str, Any]: ...


def export_units_to_tiddlywiki_json(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write units as TiddlyWiki-compatible tiddler JSON."""
    all_units = list(units)
    exported_units = all_units if isinstance(units, Sequence) else sorted(all_units, key=_unit_sort_key)
    title_counts = Counter(_clean_title(unit.title) for unit in exported_units)
    tiddlers = [_tiddler(unit, title_counts=title_counts) for unit in exported_units]
    text = json.dumps(tiddlers, ensure_ascii=False, sort_keys=True, indent=2) + "\n"

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return {
        "path": str(output_path),
        "units_scanned": len(all_units),
        "units_exported": len(tiddlers),
        "bytes_written": output_path.stat().st_size,
    }


def _tiddler(unit: KnowledgeUnit, *, title_counts: Counter[str]) -> dict[str, Any]:
    title = _clean_title(unit.title)
    if title_counts[title] > 1:
        title = f"{title} ({_source_suffix(unit)})"
    return {
        "title": title,
        "text": str(unit.content or ""),
        "tags": _tags_field(unit.tags),
        "created": _tiddly_timestamp(unit.created_at),
        "modified": _tiddly_timestamp(unit.updated_at),
        "type": "text/markdown",
        "source_project": _json_value(unit.source_project),
        "source_id": str(unit.source_id or ""),
        "metadata": _json_value(unit.metadata),
    }


def _tiddly_timestamp(value: datetime | date | object) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc)
        return value.strftime("%Y%m%d%H%M%S") + f"{value.microsecond // 1000:03d}"
    if isinstance(value, date):
        return f"{value.year:04d}{value.month:02d}{value.day:02d}000000000"
    return ""


def _tags_field(tags: Iterable[object]) -> str:
    cleaned = sorted({text for tag in tags if (text := _clean_text(tag))})
    return " ".join(_escape_tag(tag) for tag in cleaned)


def _escape_tag(tag: str) -> str:
    if any(char.isspace() for char in tag) or "[[" in tag or "]]" in tag:
        return f"[[{tag.replace(']]', '] ]')}]]"
    return tag


def _source_suffix(unit: KnowledgeUnit) -> str:
    seed = f"{_json_value(unit.source_project)}:{unit.source_id or unit.id}"
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8]
    source_id = _clean_text(unit.source_id)
    return source_id if source_id and len(source_id) <= 24 and " " not in source_id else digest


def _clean_title(value: object) -> str:
    return _clean_text(value) or "Untitled"


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\r\n", "\n").replace("\r", "\n").split())


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, BaseModel):
        return _json_value(value.model_dump())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items(), key=_item_key)}
    if isinstance(value, list | tuple | set):
        return [_json_value(item) for item in value]
    return str(value)


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[str, str, str]:
    return (str(_json_value(unit.source_project) or ""), str(unit.source_id or ""), str(unit.title or ""))


def _item_key(item: tuple[Any, Any]) -> str:
    return str(item[0])
