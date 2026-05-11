"""JSON export adapter for knowledge unit collections."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel

from graph.types.models import KnowledgeUnit


def export_units_to_json(
    units: Iterable[KnowledgeUnit],
    *,
    mode: str = "full",
    pretty: bool = False,
) -> str:
    """
    Export knowledge units to JSON format.

    Args:
        units: Iterable of KnowledgeUnit instances to export
        mode: Export mode - "full" (all fields) or "compact" (id, title, content only)
        pretty: If True, format JSON with indentation for human readability

    Returns:
        JSON string representation of the units

    Raises:
        ValueError: If mode is not "full" or "compact"
    """
    if mode not in ("full", "compact"):
        raise ValueError(f"mode must be 'full' or 'compact', got: {mode}")

    units_list = list(units)

    if mode == "full":
        data = [_full_unit_dict(unit) for unit in units_list]
    else:  # compact
        data = [_compact_unit_dict(unit) for unit in units_list]

    if pretty:
        return json.dumps(data, indent=2, ensure_ascii=False)
    return json.dumps(data, ensure_ascii=False)


def _full_unit_dict(unit: KnowledgeUnit) -> dict[str, Any]:
    """Convert unit to full dictionary with all fields."""
    return {
        "id": unit.id,
        "source_project": _serialize_value(unit.source_project),
        "source_id": unit.source_id,
        "source_entity_type": unit.source_entity_type,
        "title": unit.title,
        "content": unit.content,
        "content_type": _serialize_value(unit.content_type),
        "metadata": _serialize_value(unit.metadata),
        "tags": sorted(unit.tags),
        "confidence": unit.confidence,
        "utility_score": unit.utility_score,
        "created_at": _serialize_value(unit.created_at),
        "ingested_at": _serialize_value(unit.ingested_at),
        "updated_at": _serialize_value(unit.updated_at),
    }


def _compact_unit_dict(unit: KnowledgeUnit) -> dict[str, str]:
    """Convert unit to compact dictionary with only id, title, and content."""
    return {
        "id": unit.id,
        "title": unit.title,
        "content": unit.content,
    }


def _serialize_value(value: Any) -> Any:
    """Serialize a value for JSON output."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime | date):
        return value.isoformat()
    if value is None:
        return None
    if isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, BaseModel):
        return _serialize_value(value.model_dump())
    if isinstance(value, dict):
        return {
            str(key): _serialize_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, list | tuple):
        return [_serialize_value(item) for item in value]
    return str(value)
