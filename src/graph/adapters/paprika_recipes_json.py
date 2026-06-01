"""Adapter for Paprika recipe JSON exports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, iter_paths, parse_datetime, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class PaprikaRecipesJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "paprika_recipes_json"

    @property
    def entity_types(self) -> list[str]:
        return ["recipe"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "recipe" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        for path in iter_paths(self.path, {".json"}):
            try:
                records = self._records(path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for index, record in enumerate(records):
                unit = self._unit(record, path.name, index)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)
        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _records(self, path: Path) -> list[dict[str, Any]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            for key in ("recipes", "items", "data"):
                value = parsed.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
            return [parsed]
        return []

    def _unit(self, record: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        name = self._text(record.get("name") or record.get("title"))
        source_url = self._text(record.get("source_url") or record.get("source") or record.get("url"))
        ingredients = self._lines(record.get("ingredients"))
        directions = self._lines(record.get("directions") or record.get("instructions"))
        notes = self._text(record.get("notes") or record.get("description"))
        if not name and not source_url and not ingredients and not directions:
            return None
        categories = split_values(record.get("categories") or record.get("category"))
        created = parse_datetime(record.get("created") or record.get("created_at"))
        modified = parse_datetime(record.get("modified") or record.get("modified_at") or record.get("updated_at")) or created
        metadata = clean_metadata(
            {
                "name": name,
                "source_url": source_url,
                "categories": categories,
                "rating": record.get("rating"),
                "prep_time": self._text(record.get("prep_time") or record.get("preptime")),
                "cook_time": self._text(record.get("cook_time") or record.get("cooktime")),
                "ingredients": ingredients,
                "directions": directions,
                "notes": notes,
                "created": created.isoformat() if created else self._text(record.get("created")),
                "modified": modified.isoformat() if modified else self._text(record.get("modified")),
                "source_file": source_file,
                "record_index": index,
            }
        )
        now = datetime.now(timezone.utc)
        return KnowledgeUnit(
            source_project="paprika_recipes_json",
            source_id=digest_source_id("paprika_recipes_json", record.get("uid") or record.get("id") or source_url or name, index),
            source_entity_type="recipe",
            title=name or source_url or "Untitled recipe",
            content=self._content(name, ingredients, directions, notes, source_url),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=list(dict.fromkeys(["paprika", "recipe", *categories])),
            created_at=created or now,
            updated_at=modified or created or now,
        )

    def _content(self, name: str, ingredients: list[str], directions: list[str], notes: str, source_url: str) -> str:
        parts = [name]
        if ingredients:
            parts.append("Ingredients:\n" + "\n".join(ingredients))
        if directions:
            parts.append("Directions:\n" + "\n".join(directions))
        if notes:
            parts.append("Notes:\n" + notes)
        if source_url:
            parts.append(f"Source URL: {source_url}")
        return "\n\n".join(part for part in parts if part)

    def _lines(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [self._text(item) for item in value if self._text(item)]
        text = self._text(value)
        return [line.strip() for line in text.splitlines() if line.strip()]

    def _text(self, value: Any) -> str:
        return "" if value is None else str(value).strip()
