"""Adapter for Apple Notes HTML export files."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class _HTMLTextExtractor(HTMLParser):
    """Extract text content from HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        _ = attrs
        if tag.lower() in {"script", "style", "meta", "link"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "meta", "link"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = data.strip()
        if text:
            self.parts.append(text)

    def get_text(self) -> str:
        return "\n".join(self.parts)


class AppleNotesExportAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "apple_notes_export"

    @property
    def entity_types(self) -> list[str]:
        return ["note"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "note" not in entity_types:
            return result

        root = Path(self.path).expanduser() if self.path else None
        if root is None or not root.exists():
            return result

        sync_at = self._sync_datetime(since) if since else None
        files = self._find_html_files(root)

        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except (OSError, UnicodeDecodeError):
                continue

            stat = file_path.stat()
            created_at = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc)
            updated_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

            if sync_at and updated_at <= sync_at:
                continue

            # Extract title and text
            title = self._extract_title(content, file_path.stem)
            parser = _HTMLTextExtractor()
            parser.feed(content)
            text = parser.get_text()

            # Extract folder from path
            folder_path = self._extract_folder(file_path, root)

            note_id = hashlib.sha1(file_path.as_posix().encode()).hexdigest()[:16]
            source_id = f"apple_notes_export:note:{note_id}"

            result.units.append(
                KnowledgeUnit(
                    source_project=SourceProject.APPLE_NOTES_EXPORT,
                    source_id=source_id,
                    source_entity_type="note",
                    title=title or "Untitled Apple Note",
                    content=text or content[:1000],
                    content_type=ContentType.ARTIFACT,
                    metadata={
                        "path": str(file_path),
                        "folder": folder_path,
                        "file_name": file_path.name,
                    },
                    tags=[folder_path] if folder_path else [],
                    created_at=created_at,
                    updated_at=updated_at,
                )
            )

        result.units.sort(key=lambda u: (u.created_at, u.source_id))
        return result

    def _find_html_files(self, root: Path) -> list[Path]:
        """Find all HTML files."""
        if root.is_file():
            return [root] if root.suffix.lower() == ".html" else []
        files: list[Path] = []
        files.extend(root.rglob("*.html"))
        return sorted(f for f in files if f.is_file())

    def _extract_title(self, content: str, fallback: str) -> str:
        """Extract title from HTML."""
        title_match = re.search(r"<title>([^<]+)</title>", content, re.IGNORECASE)
        if title_match:
            return title_match.group(1).strip()
        h1_match = re.search(r"<h1[^>]*>([^<]+)</h1>", content, re.IGNORECASE)
        if h1_match:
            return h1_match.group(1).strip()
        return fallback

    def _extract_folder(self, file_path: Path, root: Path) -> str:
        """Extract folder name from path."""
        try:
            rel_path = file_path.relative_to(root)
            if len(rel_path.parts) > 1:
                return rel_path.parts[0]
        except ValueError:
            pass
        return ""

    def _sync_datetime(self, since: SyncState) -> datetime:
        """Convert SyncState to datetime."""
        value = since.last_sync_at
        if isinstance(value, datetime):
            parsed = value
        else:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
