"""Adapter for Apple Notes export directories."""

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
        files = self._find_note_files(root)

        for file_path in files:
            try:
                raw_content = file_path.read_text(encoding="utf-8", errors="ignore")
            except (OSError, UnicodeDecodeError):
                continue

            try:
                stat = file_path.stat()
            except OSError:
                continue

            parsed = self._parse_note(raw_content, file_path)
            created_at = parsed["created_at"] or datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc)
            updated_at = parsed["updated_at"] or datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

            if sync_at and updated_at <= sync_at:
                continue

            folder_path = self._extract_folder(file_path, root)
            tags = self._tags(parsed["text"], folder_path)
            note_id = hashlib.sha256(str(file_path.resolve()).encode("utf-8")).hexdigest()[:24]

            result.units.append(
                KnowledgeUnit(
                    source_project=SourceProject.APPLE_NOTES_EXPORT,
                    source_id=f"apple_notes_export:note:{note_id}",
                    source_entity_type="note",
                    title=parsed["title"] or "Untitled Apple Note",
                    content=parsed["text"],
                    content_type=ContentType.ARTIFACT,
                    metadata={
                        "path": str(file_path),
                        "folder": folder_path,
                        "file_name": file_path.name,
                        "extension": file_path.suffix.lower(),
                        "created_at": created_at.isoformat(),
                        "updated_at": updated_at.isoformat(),
                    },
                    tags=tags,
                    created_at=created_at,
                    updated_at=updated_at,
                )
            )

        result.units.sort(key=lambda u: (u.created_at, u.source_id))
        return result

    def _find_note_files(self, root: Path) -> list[Path]:
        if root.is_file():
            return [root] if root.suffix.lower() in {".html", ".txt"} else []
        return sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".html", ".txt"}
        )

    def _parse_note(self, content: str, file_path: Path) -> dict[str, Any]:
        front_matter, body = self._front_matter(content)
        if file_path.suffix.lower() == ".html":
            title = self._extract_title(body, file_path.stem)
            parser = _HTMLTextExtractor()
            parser.feed(body)
            parser.close()
            text = parser.get_text()
        else:
            text = body.strip()
            title = self._plain_title(text, file_path.stem)

        title = self._metadata_text(front_matter, "title") or title
        created_at = self._metadata_datetime(front_matter, "created", "created_at", "date")
        updated_at = self._metadata_datetime(front_matter, "modified", "updated", "updated_at")
        return {
            "title": title,
            "text": text,
            "created_at": created_at,
            "updated_at": updated_at,
        }

    def _front_matter(self, content: str) -> tuple[dict[str, str], str]:
        if not content.startswith("---"):
            return {}, content
        match = re.match(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", content, re.DOTALL)
        if not match:
            return {}, content
        metadata: dict[str, str] = {}
        for line in match.group(1).splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            metadata[key.strip().lower()] = value.strip().strip('"')
        return metadata, match.group(2)

    def _extract_title(self, content: str, fallback: str) -> str:
        """Extract title from HTML."""
        title_match = re.search(r"<title>([^<]+)</title>", content, re.IGNORECASE)
        if title_match:
            return title_match.group(1).strip()
        h1_match = re.search(r"<h1[^>]*>([^<]+)</h1>", content, re.IGNORECASE)
        if h1_match:
            return h1_match.group(1).strip()
        return fallback

    def _plain_title(self, content: str, fallback: str) -> str:
        for line in content.splitlines():
            title = line.strip().lstrip("#").strip()
            if title:
                return title
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

    def _tags(self, content: str, folder_path: str) -> list[str]:
        tags: list[str] = []
        if folder_path:
            tags.append(folder_path)
        for match in re.finditer(r"(?<!\w)#([A-Za-z0-9][\w-]*)", content):
            tag = match.group(1).lower()
            if tag not in tags:
                tags.append(tag)
        return tags

    def _metadata_text(self, metadata: dict[str, str], key: str) -> str:
        return metadata.get(key, "").strip()

    def _metadata_datetime(self, metadata: dict[str, str], *keys: str) -> datetime | None:
        for key in keys:
            value = metadata.get(key, "").strip()
            if not value:
                continue
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                continue
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        return None

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
