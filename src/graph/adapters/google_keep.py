"""Adapter for Google Keep Takeout JSON exports."""

from __future__ import annotations

import json
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class GoogleKeepAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "google_keep"

    @property
    def entity_types(self) -> list[str]:
        return ["keep_note", "checklist_item"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        allowed = set(entity_types or ["keep_note"])
        if not allowed.intersection(self.entity_types):
            return result

        sync_at = self._sync_datetime(since) if since else None
        for path in self._iter_paths():
            note = self._read_note(path)
            unit = self._unit_from_note(note, path)
            comparable_at = self._comparable_datetime(unit)
            if sync_at and comparable_at and comparable_at <= sync_at:
                continue
            if "keep_note" in allowed:
                result.units.append(unit)
            item_units = self._checklist_item_units(note, path, unit)
            if "checklist_item" in allowed:
                result.units.extend(item_units)
            if {"keep_note", "checklist_item"}.issubset(allowed):
                result.edges.extend(self._checklist_edges(unit, item_units))

        return result

    def _iter_paths(self) -> list[Path]:
        path = Path(self.path).expanduser() if self.path else None
        if path is None or not path.exists():
            return []
        if path.is_file():
            return [path] if path.suffix.lower() == ".json" else []
        if path.is_dir():
            return sorted(item for item in path.rglob("*.json") if item.is_file())
        return []

    def _read_note(self, path: Path) -> dict[str, Any]:
        try:
            parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed Google Keep JSON in {path}: {exc.msg}") from exc
        except UnicodeDecodeError as exc:
            raise ValueError(f"Could not decode Google Keep JSON file {path}") from exc
        except OSError as exc:
            raise ValueError(f"Could not read Google Keep JSON file {path}") from exc

        if not isinstance(parsed, dict):
            raise ValueError(f"Google Keep JSON file {path} must contain one note object")
        return parsed

    def _unit_from_note(self, note: dict[str, Any], path: Path) -> KnowledgeUnit:
        title = self._string(note.get("title")) or "Untitled Google Keep note"
        text = self._string(note.get("textContent"))
        checklist_items = self._checklist_items(note.get("listContent"))
        content = self._content(title, text, checklist_items)
        created_at = self._timestamp(note, "createdTimestampUsec", "created_at", "createdAt")
        updated_at = self._timestamp(
            note,
            "userEditedTimestampUsec",
            "updatedTimestampUsec",
            "editedTimestampUsec",
            "updated_at",
            "updatedAt",
        )
        tags = self._tags(note.get("labels"))

        metadata = {
            "id": self._note_id(note),
            "source_path": str(path),
            "title": title,
            "textContent": text,
            "labels": tags,
            "checklist": checklist_items,
        }
        for key in (
            "isArchived",
            "isTrashed",
            "isPinned",
            "color",
            "createdTimestampUsec",
            "userEditedTimestampUsec",
            "updatedTimestampUsec",
            "editedTimestampUsec",
            "trashedTimestampUsec",
            "created_at",
            "createdAt",
            "updated_at",
            "updatedAt",
        ):
            if key in note:
                metadata[key] = self._jsonable(note[key])

        return KnowledgeUnit(
            source_project=SourceProject.GOOGLE_KEEP,
            source_id=self._source_id(note, path),
            source_entity_type="keep_note",
            title=title,
            content=content,
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=tags,
            created_at=created_at or updated_at or datetime.now(timezone.utc),
            updated_at=updated_at or created_at or datetime.now(timezone.utc),
        )

    def _content(
        self, title: str, text: str, checklist_items: list[dict[str, Any]]
    ) -> str:
        parts = []
        if title:
            parts.append(title)
        if text:
            parts.append(text)
        if checklist_items:
            parts.extend(
                f"[{'x' if item['checked'] else ' '}] {item['text']}"
                for item in checklist_items
                if item["text"]
            )
        return "\n".join(parts) or title

    def _checklist_items(self, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        items: list[dict[str, Any]] = []
        for index, item in enumerate(value, start=1):
            if not isinstance(item, dict):
                continue
            text = self._string(item.get("text"))
            if not text:
                continue
            items.append(
                {
                    "text": text,
                    "checked": bool(item.get("isChecked")),
                    "position": index,
                }
            )
        return items

    def _checklist_item_units(self, note: dict[str, Any], path: Path, note_unit: KnowledgeUnit) -> list[KnowledgeUnit]:
        items = self._checklist_items(note.get("listContent"))
        units: list[KnowledgeUnit] = []
        for item in items:
            title = item["text"]
            metadata = {
                "text": title,
                "checked": item["checked"],
                "position": item["position"],
                "parent_note_source_id": note_unit.source_id,
                "parent_note_title": note_unit.title,
                "source_path": str(path),
            }
            units.append(
                KnowledgeUnit(
                    source_project=SourceProject.GOOGLE_KEEP,
                    source_id=self._checklist_item_source_id(note_unit.source_id, item["position"], title),
                    source_entity_type="checklist_item",
                    title=title,
                    content=f"[{'x' if item['checked'] else ' '}] {title}",
                    content_type=ContentType.METADATA,
                    metadata=metadata,
                    tags=note_unit.tags,
                    created_at=note_unit.created_at,
                    updated_at=note_unit.updated_at,
                )
            )
        return units

    def _checklist_edges(self, note_unit: KnowledgeUnit, item_units: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        return [
            KnowledgeEdge(
                id=self._checklist_edge_id(note_unit.source_id, item.source_id),
                from_unit_id=note_unit.source_id,
                to_unit_id=item.source_id,
                relation=EdgeRelation.CONTAINS,
                source=EdgeSource.SOURCE,
                metadata={
                    "source_project": SourceProject.GOOGLE_KEEP.value,
                    "relation_type": "note_contains_checklist_item",
                    "position": item.metadata["position"],
                },
            )
            for item in item_units
        ]

    def _tags(self, value: Any) -> list[str]:
        raw_tags: list[Any]
        if isinstance(value, list):
            raw_tags = value
        elif isinstance(value, str):
            raw_tags = re.split(r"[,;|]", value)
        else:
            raw_tags = []

        tags: list[str] = []
        seen: set[str] = set()
        for tag in raw_tags:
            if isinstance(tag, dict):
                tag = tag.get("name") or tag.get("label") or tag.get("tag") or ""
            normalized = re.sub(r"\s+", " ", str(tag).strip().removeprefix("#")).strip()
            key = normalized.casefold()
            if normalized and key not in seen:
                tags.append(normalized)
                seen.add(key)
        return tags

    def _note_id(self, note: dict[str, Any]) -> str:
        for key in ("id", "uuid", "noteId"):
            value = self._string(note.get(key))
            if value:
                return value
        return ""

    def _source_id(self, note: dict[str, Any], path: Path) -> str:
        note_id = self._note_id(note)
        if note_id:
            return f"google_keep:{note_id}"
        return f"google_keep:path:{path}"

    def _checklist_item_source_id(self, note_source_id: str, position: int, text: str) -> str:
        digest = hashlib.sha256(f"{note_source_id}|{position}|{text}".encode("utf-8")).hexdigest()[:24]
        return f"{note_source_id}:checklist_item:{digest}"

    def _checklist_edge_id(self, note_source_id: str, item_source_id: str) -> str:
        digest = hashlib.sha256(f"{note_source_id}|{item_source_id}|contains".encode("utf-8")).hexdigest()[:24]
        return f"google-keep-contains-{digest}"

    def _timestamp(self, note: dict[str, Any], *keys: str) -> datetime | None:
        for key in keys:
            parsed = self._parse_datetime(note.get(key))
            if parsed is not None:
                return parsed
        return None

    def _parse_datetime(self, value: Any) -> datetime | None:
        if value in (None, ""):
            return None
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value / 1_000_000, tz=timezone.utc)
        text = str(value).strip()
        if text.isdigit():
            return datetime.fromtimestamp(int(text) / 1_000_000, tz=timezone.utc)
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _sync_datetime(self, since: SyncState) -> datetime:
        value = since.last_sync_at
        if isinstance(value, datetime):
            parsed = value
        else:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _comparable_datetime(self, unit: KnowledgeUnit) -> datetime | None:
        return unit.updated_at or unit.created_at

    def _string(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return json.dumps(value, sort_keys=True)
        return str(value).strip()

    def _jsonable(self, value: Any) -> Any:
        try:
            json.dumps(value)
        except TypeError:
            return str(value)
        return value
