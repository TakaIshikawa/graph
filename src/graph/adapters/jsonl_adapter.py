"""Adapter for generic JSON Lines knowledge exports."""

from __future__ import annotations

import json
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class JsonlAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "jsonl"

    @property
    def entity_types(self) -> list[str]:
        return ["jsonl_record"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "jsonl_record" not in entity_types:
            return result

        sync_at = self._sync_datetime(since) if since else None
        malformed_lines = 0
        for path in self._iter_paths():
            try:
                records, malformed = self._read_records(path)
            except (OSError, UnicodeDecodeError):
                continue
            malformed_lines += malformed

            for line_number, record in records:
                source_id = self._field(record, "source_id")
                title = self._field(record, "title")
                content = self._field(record, "content")
                if not source_id or not title or not content:
                    continue

                created_at = self._parse_datetime(record.get("created_at"))
                updated_at = self._parse_datetime(record.get("updated_at"))
                sync_candidate = updated_at or created_at
                if sync_at and sync_candidate and sync_candidate <= sync_at:
                    continue

                unit = KnowledgeUnit(
                    source_project=SourceProject.JSONL,
                    source_id=source_id,
                    source_entity_type="jsonl_record",
                    title=title,
                    content=content,
                    content_type=self._parse_content_type(record.get("content_type")),
                    metadata=self._parse_metadata(record.get("metadata")),
                    tags=self._parse_tags(record.get("tags")),
                    confidence=self._parse_float(record.get("confidence")),
                    utility_score=self._parse_float(record.get("utility_score")),
                    created_at=created_at or datetime.now(timezone.utc),
                )
                if updated_at is not None:
                    unit.updated_at = updated_at
                result.units.append(unit)

        if malformed_lines:
            warnings.warn(
                f"Skipped {malformed_lines} malformed JSONL line(s).",
                stacklevel=2,
            )

        return result

    def _iter_paths(self) -> list[Path]:
        sources = [
            source.strip()
            for source in re.split(r"[\n,]", self.path)
            if source.strip()
        ]
        paths: list[Path] = []
        for source in sources:
            path = Path(source).expanduser()
            if path.is_dir():
                paths.extend(sorted(path.rglob("*.jsonl")))
            elif path.exists() and path.is_file():
                paths.append(path)
        return paths

    def _read_records(self, path: Path) -> tuple[list[tuple[int, dict[str, Any]]], int]:
        records: list[tuple[int, dict[str, Any]]] = []
        malformed = 0
        with path.open(encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                if not isinstance(parsed, dict):
                    malformed += 1
                    continue
                records.append((line_number, parsed))
        return records, malformed

    def _field(self, record: dict[str, Any], key: str) -> str:
        value = record.get(key)
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return json.dumps(value, sort_keys=True)
        return str(value).strip()

    def _parse_tags(self, value: Any) -> list[str]:
        raw_tags = value if isinstance(value, list) else str(value or "").split(",")
        tags: list[str] = []
        for tag in raw_tags:
            normalized = str(tag).strip().removeprefix("#").strip()
            if normalized and normalized not in tags:
                tags.append(normalized)
        return tags

    def _parse_metadata(self, value: Any) -> dict:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return {"metadata": value}
            if isinstance(parsed, dict):
                return parsed
            return {"metadata": parsed}
        return {"metadata": value}

    def _parse_float(self, value: Any) -> float | None:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _parse_content_type(self, value: Any) -> ContentType:
        if not value:
            return ContentType.INSIGHT
        try:
            return ContentType(str(value).strip())
        except ValueError:
            return ContentType.INSIGHT

    def _parse_datetime(self, value: Any) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
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
