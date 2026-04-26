"""Adapter for local PDF documents."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class PdfAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "pdf"

    @property
    def entity_types(self) -> list[str]:
        return ["pdf_document"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "pdf_document" not in entity_types:
            return result

        paths = self._discover_paths()
        if not paths:
            return result

        reader_cls = self._load_pdf_reader()
        sync_at = self._sync_timestamp(since) if since else None
        root = Path(self.path).expanduser()
        root = root if root.is_dir() else root.parent

        for path in paths:
            stat = path.stat()
            if sync_at is not None and stat.st_mtime <= sync_at:
                continue

            unit = self._read_pdf(reader_cls, root, path, stat.st_size, stat.st_ctime)
            if unit is not None:
                result.units.append(unit)

        return result

    def _discover_paths(self) -> list[Path]:
        configured = Path(self.path).expanduser()
        if configured.is_file() and configured.suffix.lower() == ".pdf":
            return [configured]
        if configured.is_dir():
            return sorted(
                item
                for item in configured.rglob("*")
                if item.is_file() and item.suffix.lower() == ".pdf"
            )
        return []

    def _load_pdf_reader(self):
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError(
                "PdfAdapter requires the optional dependency pypdf. "
                "Install it with `uv sync --extra pdf` or `pip install 'graph[pdf]'`."
            ) from exc
        return PdfReader

    def _read_pdf(
        self,
        reader_cls: Any,
        root: Path,
        path: Path,
        file_size: int,
        created_timestamp: float,
    ) -> KnowledgeUnit | None:
        warnings: list[str] = []
        try:
            reader = reader_cls(str(path))
        except Exception as exc:
            warnings.append(f"failed_to_open: {exc}")
            return None

        pages = list(getattr(reader, "pages", []))
        page_text: list[str] = []
        for page_number, page in enumerate(pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:
                warnings.append(f"page_{page_number}: {exc}")
                text = ""
            if text.strip():
                page_text.append(text.strip())

        source_id = path.relative_to(root).as_posix()
        content = "\n\n".join(page_text)
        return KnowledgeUnit(
            source_project=SourceProject.PDF,
            source_id=source_id,
            source_entity_type="pdf_document",
            title=path.stem,
            content=content,
            content_type=ContentType.INSIGHT,
            metadata={
                "source_file": str(path),
                "page_count": len(pages),
                "extraction_warnings": warnings,
                "file_size": file_size,
            },
            created_at=datetime.fromtimestamp(created_timestamp, tz=timezone.utc),
        )

    def _sync_timestamp(self, since: SyncState) -> float:
        if isinstance(since.last_sync_at, datetime):
            return since.last_sync_at.timestamp()
        return datetime.fromisoformat(str(since.last_sync_at)).timestamp()
