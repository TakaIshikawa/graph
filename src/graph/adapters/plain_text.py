"""Adapter for plain text files."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class PlainTextAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "plain_text"

    @property
    def entity_types(self) -> list[str]:
        return ["plain_text"]

    def __init__(
        self,
        path: str = "",
        *,
        root_path: str = "",
        source_id_root: str | None = None,
    ) -> None:
        self.path = path or root_path
        self.source_id_root = source_id_root

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "plain_text" not in entity_types:
            return result

        root = Path(self.path).expanduser()
        if not root.exists():
            return result

        files = self._text_files(root)
        source_root = Path(self.source_id_root).expanduser() if self.source_id_root else root
        if root.is_file() and not self.source_id_root:
            source_root = root.parent

        sync_at = self._sync_datetime(since) if since else None
        for file_path in files:
            stat = file_path.stat()
            file_updated_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            if sync_at and file_updated_at <= sync_at:
                continue

            relative_path = self._relative_path(file_path, source_root)
            content = file_path.read_text(encoding="utf-8", errors="replace")

            result.units.append(
                KnowledgeUnit(
                    source_project=SourceProject.PLAIN_TEXT,
                    source_id=self._source_id(relative_path),
                    source_entity_type="plain_text",
                    title=file_path.stem,
                    content=content,
                    content_type=ContentType.ARTIFACT,
                    metadata={
                        "source_file": relative_path,
                    },
                    created_at=datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc),
                    updated_at=file_updated_at,
                )
            )

        return result

    def _text_files(self, root: Path) -> list[Path]:
        if root.is_file():
            return [root] if root.suffix.lower() == ".txt" else []
        if not root.is_dir():
            return []
        return sorted(
            path for path in root.rglob("*.txt") if path.is_file()
        )

    def _source_id(self, relative_path: str) -> str:
        digest = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:16]
        return f"plain_text:{digest}"

    def _relative_path(self, path: Path, source_root: Path) -> str:
        try:
            return path.relative_to(source_root).as_posix()
        except ValueError:
            return path.as_posix()

    def _sync_datetime(self, since: SyncState) -> datetime:
        value = since.last_sync_at
        if isinstance(value, datetime):
            parsed = value
        else:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
