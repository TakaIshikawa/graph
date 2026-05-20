"""Adapter for Airtable API records JSON exports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters._personal_exports import clean_metadata, ensure_utc, iter_paths, parse_datetime
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class AirtableRecordsJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "airtable_records_json"

    @property
    def entity_types(self) -> list[str]:
        return ["record"]

    def __init__(
        self,
        path: str = "",
        *,
        base_name: str = "",
        table_name: str = "",
        base: str = "",
        table: str = "",
    ) -> None:
        self.path = path
        self.base_name = base_name or base
        self.table_name = table_name or table

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "record" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None

        for path in iter_paths(self.path, {".json"}):
            try:
                payload = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for index, record in enumerate(self._records(payload), start=1):
                unit = self._unit_from_record(record, payload, path.name, index)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)

        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _records(self, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, dict) and isinstance(payload.get("records"), list):
            return [record for record in payload["records"] if isinstance(record, dict)]
        if isinstance(payload, list):
            return [record for record in payload if isinstance(record, dict)]
        return []

    def _unit_from_record(
        self,
        record: dict[str, Any],
        payload: Any,
        source_file: str,
        source_row: int,
    ) -> KnowledgeUnit | None:
        record_id = self._text(record.get("id"))
        fields = record.get("fields") if isinstance(record.get("fields"), dict) else {}
        if not record_id and not fields:
            return None

        created_text = self._text(record.get("createdTime") or record.get("created_time"))
        created_at = parse_datetime(created_text)
        flat_fields = self._flatten_fields(fields)
        base_name = self.base_name or self._context_name(payload, "base")
        table_name = self.table_name or self._context_name(payload, "table")

        metadata = clean_metadata(
            {
                "record_id": record_id,
                "createdTime": created_text,
                "created_time": created_text,
                "base_name": base_name,
                "table_name": table_name,
                "fields": flat_fields,
                "source_file": source_file,
                "source_row": source_row,
            }
        )
        now = datetime.now(timezone.utc)
        title_field, title = self._title(fields)
        if title_field:
            metadata["title_field"] = title_field
        return KnowledgeUnit(
            source_project="airtable_records_json",
            source_id=self._source_id(record_id, source_file, source_row),
            source_entity_type="record",
            title=title,
            content=self._content(title, title_field, flat_fields),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=list(dict.fromkeys(item for item in ["airtable", base_name, table_name] if item)),
            created_at=created_at or now,
            updated_at=created_at or now,
        )

    def _context_name(self, payload: Any, name: str) -> str:
        if not isinstance(payload, dict):
            return ""
        for key in (f"{name}_name", f"{name}Name", name):
            value = payload.get(key)
            if isinstance(value, dict):
                text = self._text(value.get("name") or value.get("title") or value.get("id"))
            else:
                text = self._text(value)
            if text:
                return text
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            return self._context_name(metadata, name)
        return ""

    def _flatten_fields(self, value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        flattened: dict[str, Any] = {}
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(item, dict):
                flattened.update(self._flatten_fields(item, name))
            else:
                flattened[name] = item
        return flattened

    def _title(self, fields: dict[str, Any]) -> tuple[str, str]:
        for preferred in ("Name", "Title", "Summary"):
            text = self._text(fields.get(preferred))
            if text:
                return preferred, text
        for key, value in fields.items():
            text = self._text(value)
            if text:
                return str(key), text
        return "", "Untitled Airtable record"

    def _content(self, title: str, title_field: str, fields: dict[str, Any]) -> str:
        parts = [title]
        for key, value in fields.items():
            if key != title_field and value not in ("", None, []):
                parts.append(f"{key}: {value}")
        return "\n".join(parts)

    def _source_id(self, record_id: str, source_file: str, source_row: int) -> str:
        if record_id:
            return f"airtable_records_json:{record_id}"
        return f"airtable_records_json:{source_file}:{source_row}"

    def _text(self, value: Any) -> str:
        return "" if value is None else str(value).strip()
