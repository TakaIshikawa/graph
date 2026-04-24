"""Adapter for generic CSV knowledge exports."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class CsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "csv"

    @property
    def entity_types(self) -> list[str]:
        return ["csv_row"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "csv_row" not in entity_types:
            return result

        sync_at = self._sync_datetime(since) if since else None
        for path in self._iter_paths():
            try:
                rows = self._read_rows(path)
            except (OSError, csv.Error, UnicodeDecodeError):
                continue

            for row_number, row in rows:
                title = self._cell(row, "title")
                content = self._cell(row, "content")
                if not title or not content:
                    continue

                created_at = self._parse_datetime(self._cell(row, "created_at"))
                if sync_at and created_at and created_at <= sync_at:
                    continue

                result.units.append(
                    KnowledgeUnit(
                        source_project=SourceProject.CSV,
                        source_id=self._source_id(row, row_number, title),
                        source_entity_type="csv_row",
                        title=title,
                        content=content,
                        content_type=self._parse_content_type(
                            self._cell(row, "content_type")
                        ),
                        metadata=self._parse_metadata(self._cell(row, "metadata_json")),
                        tags=self._parse_tags(self._cell(row, "tags")),
                        confidence=self._parse_float(self._cell(row, "confidence")),
                        utility_score=self._parse_float(
                            self._cell(row, "utility_score")
                        ),
                        created_at=created_at or datetime.now(timezone.utc),
                    )
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
                paths.extend(sorted(path.rglob("*.csv")))
            elif path.exists() and path.is_file():
                paths.append(path)
        return paths

    def _read_rows(self, path: Path) -> list[tuple[int, dict[str, str]]]:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return []
            fields = {field.strip() for field in reader.fieldnames if field}
            if not {"title", "content"}.issubset(fields):
                return []
            return [(index, row) for index, row in enumerate(reader, start=2)]

    def _cell(self, row: dict[str, str], key: str) -> str:
        value = row.get(key)
        if value is None:
            return ""
        return str(value).strip()

    def _source_id(self, row: dict[str, str], row_number: int, title: str) -> str:
        explicit = self._cell(row, "source_id")
        if explicit:
            return explicit
        normalized_title = re.sub(r"\s+", " ", title).strip()
        slug = re.sub(r"[^a-z0-9]+", "-", normalized_title.lower()).strip("-")
        slug = slug[:48] or "untitled"
        digest = hashlib.sha256(f"{row_number}\0{normalized_title}".encode()).hexdigest()
        return f"row-{row_number}-{slug}-{digest[:12]}"

    def _parse_tags(self, value: str) -> list[str]:
        tags: list[str] = []
        for tag in value.split(","):
            normalized = tag.strip().removeprefix("#").strip()
            if normalized and normalized not in tags:
                tags.append(normalized)
        return tags

    def _parse_metadata(self, value: str) -> dict:
        if not value:
            return {}
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"metadata_json": value}
        if isinstance(parsed, dict):
            return parsed
        return {"metadata_json": parsed}

    def _parse_float(self, value: str) -> float | None:
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    def _parse_content_type(self, value: str) -> ContentType:
        if not value:
            return ContentType.INSIGHT
        try:
            return ContentType(value)
        except ValueError:
            return ContentType.INSIGHT

    def _parse_datetime(self, value: str) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
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
