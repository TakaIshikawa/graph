"""Adapter for Apple Notes Takeout HTML files."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        _ = attrs
        if tag.lower() in {"script", "style"}:
            self.skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style"} and self.skip:
            self.skip -= 1

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text and not self.skip:
            self.parts.append(text)

    def text(self) -> str:
        return "\n".join(self.parts)


class AppleNotesTakeoutHtmlAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "apple_notes_takeout_html"

    @property
    def entity_types(self) -> list[str]:
        return ["note"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types and "note" not in entity_types:
            return result
        root = Path(self.path).expanduser() if self.path else None
        if root is None or not root.exists():
            return result
        sync_at = self._sync_datetime(since) if since else None
        for file_path in self._files(root):
            try:
                html = file_path.read_text(encoding="utf-8", errors="ignore")
                stat = file_path.stat()
            except OSError:
                continue
            created_at = self._meta_date(html, "created", "created_at", "creation-date") or datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc)
            updated_at = self._meta_date(html, "modified", "updated", "updated_at", "modification-date") or datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            if sync_at and updated_at <= sync_at:
                continue
            parser = _TextParser()
            parser.feed(html)
            parser.close()
            title = self._title(html, file_path.stem)
            source_id = hashlib.sha256(str(file_path.resolve()).encode("utf-8")).hexdigest()[:24]
            result.units.append(
                KnowledgeUnit(
                    source_project=SourceProject.APPLE_NOTES_EXPORT,
                    source_id=f"apple_notes_takeout_html:note:{source_id}",
                    source_entity_type="note",
                    title=title,
                    content=parser.text(),
                    content_type=ContentType.ARTIFACT,
                    metadata={"source_file": str(file_path), "created_at": created_at.isoformat(), "updated_at": updated_at.isoformat()},
                    created_at=created_at,
                    updated_at=updated_at,
                )
            )
        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _files(self, root: Path) -> list[Path]:
        if root.is_file():
            return [root] if root.suffix.lower() in {".html", ".htm"} else []
        return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".html", ".htm"})

    def _title(self, html: str, fallback: str) -> str:
        for pattern in (r"<title[^>]*>(.*?)</title>", r"<h1[^>]*>(.*?)</h1>", r"<h2[^>]*>(.*?)</h2>"):
            match = re.search(pattern, html, re.IGNORECASE | re.DOTALL)
            if match:
                text = re.sub(r"<[^>]+>", "", match.group(1)).strip()
                if text:
                    return text
        return fallback

    def _meta_date(self, html: str, *names: str) -> datetime | None:
        for name in names:
            pattern = rf"<meta[^>]+(?:name|property)=[\"']{re.escape(name)}[\"'][^>]+content=[\"']([^\"']+)[\"']"
            match = re.search(pattern, html, re.IGNORECASE)
            if not match:
                continue
            try:
                parsed = datetime.fromisoformat(match.group(1).replace("Z", "+00:00"))
            except ValueError:
                continue
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        return None

    def _sync_datetime(self, since: SyncState) -> datetime:
        value = getattr(since, "last_sync_at", None)
        return value if value and value.tzinfo else (value.replace(tzinfo=timezone.utc) if value else datetime.min.replace(tzinfo=timezone.utc))
