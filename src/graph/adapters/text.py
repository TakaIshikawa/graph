"""Adapter for local plain text documents."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class TextAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "text"

    @property
    def entity_types(self) -> list[str]:
        return ["text_document"]

    def __init__(self, root_path: str = "") -> None:
        self.root_path = root_path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "text_document" not in entity_types:
            return result

        root = Path(self.root_path).expanduser()
        if not root.exists() or not root.is_dir():
            return result

        sync_at = self._sync_timestamp(since) if since else None
        for path in sorted(item for item in root.rglob("*.txt") if item.is_file()):
            stat = path.stat()
            if sync_at is not None and stat.st_mtime <= sync_at:
                continue

            try:
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            source_id = path.relative_to(root).as_posix()
            result.units.append(
                KnowledgeUnit(
                    source_project=SourceProject.ME,
                    source_id=source_id,
                    source_entity_type="text_document",
                    title=self._title(content, path),
                    content=content,
                    content_type=ContentType.INSIGHT,
                    metadata={
                        "path": source_id,
                        "file_size": stat.st_size,
                    },
                    created_at=datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc),
                )
            )

        return result

    def _title(self, content: str, path: Path) -> str:
        for line in content.splitlines():
            title = line.strip()
            if title:
                return title
        return path.stem

    def _sync_timestamp(self, since: SyncState) -> float:
        if isinstance(since.last_sync_at, datetime):
            return since.last_sync_at.timestamp()
        return datetime.fromisoformat(str(since.last_sync_at)).timestamp()
