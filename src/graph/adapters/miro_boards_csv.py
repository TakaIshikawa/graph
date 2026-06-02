"""Adapter for Miro board CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class MiroBoardsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "miro_boards_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["board"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "board" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit(row, path.name, index)
                if unit and (sync_at is None or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        board_id = first(row, "Board ID", "Board Id", "ID")
        name = first(row, "Name", "Board Name")
        description = first(row, "Description")
        url = first(row, "Board URL", "URL", "Link")
        if not any([board_id, name, description, url]):
            return None
        created = parse_datetime(first(row, "Created At", "Created"))
        modified = parse_datetime(first(row, "Modified At", "Updated At", "Modified"))
        last_opened = parse_datetime(first(row, "Last Opened At", "Last Opened"))
        updated = modified or last_opened or created
        owner = first(row, "Owner")
        team = first(row, "Team", "Project")
        access = first(row, "Access Level", "Sharing", "Permission")
        metadata = clean_metadata(
            {
                "board_id": board_id,
                "owner": owner,
                "team": team,
                "access_level": access,
                "board_url": url,
                "source_url": url,
                "created_at": created.isoformat() if created else "",
                "modified_at": modified.isoformat() if modified else "",
                "last_opened_at": last_opened.isoformat() if last_opened else "",
                "source_file": source_file,
            }
        )
        now = datetime.now(timezone.utc)
        return KnowledgeUnit(
            source_project=SourceProject.MIRO_BOARDS_CSV,
            source_id=digest_source_id("miro_boards_csv", board_id or url or name, index if not board_id else ""),
            source_entity_type="board",
            title=name or board_id or "Miro board",
            content="\n".join(part for part in [name, description, f"URL: {url}" if url else "", f"Owner: {owner}" if owner else ""] if part),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=["miro", "board"],
            created_at=created or updated or now,
            updated_at=updated or created or now,
        )
