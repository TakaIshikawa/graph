"""Adapter for newline-delimited generic note exports."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class JsonlNotesAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "jsonl_notes"

    @property
    def entity_types(self) -> list[str]:
        return ["jsonl_note"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "jsonl_note" not in entity_types:
            return result

        sync_at = self._sync_datetime(since) if since else None
        for path in self._iter_paths():
            with path.open(encoding="utf-8-sig") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    record = self._parse_line(line, path, line_number)
                    content = self._field(record, "text") or self._field(
                        record, "content"
                    )
                    if not content:
                        continue

                    created_at = self._parse_datetime(record.get("created_at"))
                    updated_at = self._parse_datetime(record.get("updated_at"))
                    sync_candidate = updated_at or created_at
                    if sync_at and sync_candidate and sync_candidate <= sync_at:
                        continue

                    title = self._field(record, "title") or self._derive_title(content)
                    unit = KnowledgeUnit(
                        source_project=SourceProject.JSONL_NOTES,
                        source_id=self._source_id(record, path, line_number),
                        source_entity_type="jsonl_note",
                        title=title,
                        content=content,
                        content_type=ContentType.INSIGHT,
                        metadata=self._metadata(record, path, line_number),
                        tags=self._tags(record.get("tags")),
                        created_at=created_at or datetime.now(timezone.utc),
                    )
                    if updated_at is not None:
                        unit.updated_at = updated_at
                    result.units.append(unit)

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

    def _parse_line(self, line: str, path: Path, line_number: int) -> dict[str, Any]:
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in {path} on line {line_number}: {exc.msg}"
            ) from exc
        if not isinstance(parsed, dict):
            raise ValueError(
                f"Invalid JSONL record in {path} on line {line_number}: "
                "expected object"
            )
        return parsed

    def _source_id(self, record: dict[str, Any], path: Path, line_number: int) -> str:
        source_id = self._field(record, "id")
        if source_id:
            return source_id
        return f"{path}:{line_number}"

    def _field(self, record: dict[str, Any], key: str) -> str:
        value = record.get(key)
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return json.dumps(value, sort_keys=True)
        return str(value).strip()

    def _derive_title(self, content: str) -> str:
        first_line = next(
            (line.strip() for line in content.splitlines() if line.strip()), ""
        )
        if not first_line:
            return "Untitled note"
        return first_line[:80]

    def _metadata(
        self, record: dict[str, Any], path: Path, line_number: int
    ) -> dict[str, Any]:
        metadata = self._parse_metadata(record.get("metadata"))
        metadata.setdefault("source_file", path.name)
        metadata.setdefault("file_path", str(path))
        metadata.setdefault("line_number", line_number)
        return metadata

    def _parse_metadata(self, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return dict(value)
        return {"metadata": value}

    def _tags(self, value: Any) -> list[str]:
        if value is None:
            return []
        raw_tags = value if isinstance(value, list) else str(value).split(",")
        tags: list[str] = []
        for raw_tag in raw_tags:
            tag = (
                re.sub(r"\s+", " ", str(raw_tag).strip().removeprefix("#"))
                .strip()
                .lower()
            )
            if tag and tag not in tags:
                tags.append(tag)
        return tags

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
        parsed = (
            value
            if isinstance(value, datetime)
            else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        )
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
