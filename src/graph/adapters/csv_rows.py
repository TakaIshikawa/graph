"""Adapter for generic CSV files where each row is a knowledge unit."""

from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class CsvRowsAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "csv_rows"

    @property
    def entity_types(self) -> list[str]:
        return ["csv_row"]

    def __init__(
        self,
        path: str = "",
        *,
        title_column: str = "title",
        content_column: str = "content",
    ) -> None:
        self.path = path
        self.title_column = title_column
        self.content_column = content_column

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
        for path, source_file in self._iter_paths():
            try:
                updated_at = datetime.fromtimestamp(
                    path.stat().st_mtime, tz=timezone.utc
                )
                if sync_at and updated_at <= sync_at:
                    continue
                rows = self._read_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue

            for row_number, row in rows:
                unit = self._unit_from_row(row, source_file, row_number, updated_at)
                if unit is not None:
                    result.units.append(unit)

        return result

    def _iter_paths(self) -> list[tuple[Path, str]]:
        entries: list[tuple[Path, str]] = []
        for source in re.split(r"[\n,]", self.path):
            source = source.strip()
            if not source:
                continue
            path = Path(source).expanduser()
            if path.is_dir():
                for child in sorted(path.rglob("*.csv")):
                    if child.is_file():
                        entries.append((child, self._relative_path(child, path)))
            elif path.exists() and path.is_file():
                entries.append((path, path.name))
        return entries

    def _read_rows(self, path: Path) -> list[tuple[int, dict[str, Any]]]:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return []

            rows: list[tuple[int, dict[str, Any]]] = []
            for row_number, row in enumerate(reader, start=2):
                normalized = {
                    str(key).strip(): value
                    for key, value in row.items()
                    if key is not None
                }
                if any(self._text(value) for value in normalized.values()):
                    rows.append((row_number, normalized))
            return rows

    def _unit_from_row(
        self,
        row: dict[str, Any],
        source_file: str,
        row_number: int,
        updated_at: datetime,
    ) -> KnowledgeUnit | None:
        title_key = self._matching_key(row, self.title_column)
        content_key = self._matching_key(row, self.content_column)
        title = self._title(row, title_key, row_number)
        content = self._content(row, content_key, title_key, title)
        if not content:
            return None

        metadata: dict[str, Any] = {
            "source_file": source_file,
            "row_number": row_number,
        }
        fields = self._metadata_fields(row, title_key, content_key)
        if fields:
            metadata["fields"] = fields

        return KnowledgeUnit(
            source_project=SourceProject.CSV_ROWS,
            source_id=self._source_id(source_file, row_number),
            source_entity_type="csv_row",
            title=title,
            content=content,
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            created_at=updated_at,
            updated_at=updated_at,
        )

    def _title(
        self, row: dict[str, Any], title_key: str | None, row_number: int
    ) -> str:
        if title_key:
            title = self._text(row.get(title_key))
            if title:
                return title

        for value in row.values():
            title = self._text(value)
            if title:
                return title
        return f"Row {row_number}"

    def _content(
        self,
        row: dict[str, Any],
        content_key: str | None,
        title_key: str | None,
        title: str,
    ) -> str:
        if content_key:
            content = self._text(row.get(content_key))
            if content:
                return content
            return title

        lines = []
        for key, value in row.items():
            if key == title_key:
                continue
            text = self._text(value)
            if text:
                lines.append(f"{key}: {text}")
        return "\n".join(lines) or title

    def _metadata_fields(
        self,
        row: dict[str, Any],
        title_key: str | None,
        content_key: str | None,
    ) -> dict[str, str]:
        fields: dict[str, str] = {}
        for key, value in row.items():
            if key in {title_key, content_key}:
                continue
            text = self._text(value)
            if text:
                fields[key] = text
        return fields

    def _matching_key(self, row: dict[str, Any], wanted: str) -> str | None:
        wanted = wanted.strip()
        if wanted in row:
            return wanted
        wanted_lower = wanted.lower()
        for key in row:
            if key.lower() == wanted_lower:
                return key
        return None

    def _source_id(self, source_file: str, row_number: int) -> str:
        return f"csv_rows:{source_file}:row-{row_number}"

    def _relative_path(self, path: Path, root: Path) -> str:
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            return path.as_posix()

    def _text(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    def _sync_datetime(self, since: SyncState) -> datetime:
        value = since.last_sync_at
        if isinstance(value, datetime):
            parsed = value
        else:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
