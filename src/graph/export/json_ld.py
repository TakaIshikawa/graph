"""JSON-LD schema.org export helpers for knowledge units."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel

from graph.types.models import KnowledgeUnit

SCHEMA_CONTEXT = "https://schema.org"
UNIT_ID_PREFIX = "urn:knowledge-unit:"
_URL_KEYS = ("url", "source_url", "external_url", "uri")
_SAME_AS_KEYS = ("sameAs", "same_as", "canonical_url", "identifier.url")


def export_units_to_json_ld(
    units: KnowledgeUnit | Iterable[KnowledgeUnit],
    path: str | Path | None = None,
    *,
    unit_id_prefix: str = UNIT_ID_PREFIX,
) -> str:
    """Return units as deterministic schema.org JSON-LD."""
    unit_list = [units] if isinstance(units, KnowledgeUnit) else list(units)
    nodes = [_unit_node(unit, unit_id_prefix=unit_id_prefix) for unit in sorted(unit_list, key=_unit_key)]
    document = nodes[0] if len(nodes) == 1 else {"@context": SCHEMA_CONTEXT, "@graph": nodes}
    if len(nodes) == 1:
        document = {"@context": SCHEMA_CONTEXT, **document}

    text = json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2)
    if path is not None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    return text


def _unit_node(unit: KnowledgeUnit, *, unit_id_prefix: str) -> dict[str, Any]:
    metadata = unit.metadata
    node: dict[str, Any] = {
        "@type": "CreativeWork",
        "@id": f"{unit_id_prefix}{quote(str(unit.id or unit.source_id), safe='')}",
        "name": unit.title,
        "text": unit.content,
        "keywords": sorted(str(tag) for tag in unit.tags),
        "dateCreated": _json_value(unit.created_at),
        "dateModified": _json_value(unit.updated_at),
        "encodingFormat": _json_value(unit.content_type),
        "isBasedOn": {
            "@type": "CreativeWork",
            "identifier": unit.source_id,
            "name": _json_value(unit.source_project),
            "additionalType": unit.source_entity_type,
        },
    }

    url = _first_text(metadata, _URL_KEYS)
    if url:
        node["url"] = url

    same_as = _string_values(_first_metadata_value(metadata, _SAME_AS_KEYS))
    if same_as:
        node["sameAs"] = same_as[0] if len(same_as) == 1 else same_as

    metadata_properties = _metadata_properties(metadata)
    if metadata_properties:
        node["additionalProperty"] = metadata_properties
    return node


def _metadata_properties(metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    properties = []
    for key, value in sorted(metadata.items(), key=lambda item: str(item[0])):
        json_value = _json_value(value)
        if json_value is None:
            continue
        properties.append({"@type": "PropertyValue", "name": str(key), "value": json_value})
    return properties


def _first_text(metadata: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        text = _clean_text(_scalar_text(_nested_value(metadata, key)))
        if text:
            return text
    return ""


def _first_metadata_value(metadata: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = _nested_value(metadata, key)
        if value is not None:
            return value
    return None


def _nested_value(metadata: Mapping[str, Any], key: str) -> Any:
    if key in metadata:
        return metadata.get(key)
    current: Any = metadata
    for part in key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current.get(part)
    return current


def _string_values(value: Any) -> list[str]:
    if value is None:
        return []
    items = value if isinstance(value, list | tuple | set) else [value]
    return [text for item in items if (text := _clean_text(_scalar_text(item)))]


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
        return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_value(item) for item in value)
    return str(value)


def _scalar_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime | date):
        return value.isoformat()
    return str(value)


def _clean_text(value: str) -> str:
    return " ".join(str(value).replace("\r\n", "\n").replace("\r", "\n").split())


def _unit_key(unit: KnowledgeUnit) -> tuple[str, str]:
    return (str(unit.id or ""), str(unit.source_id or ""))
