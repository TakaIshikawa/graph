"""Adapter for Bear app export archives with markdown notes and tag syntax."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


TAG_RE = re.compile(r"(?<![\w/])#([\w][\w-]*(?:/[\w][\w-]*)*)", re.UNICODE)


class BearExportAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "bear_export"

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

        # Find markdown files
        files = self._find_markdown_files(root)

        # Track note links for creating relations
        note_titles: dict[str, str] = {}  # title -> source_id
        note_links: dict[str, list[str]] = {}  # source_id -> list of linked titles

        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except (OSError, UnicodeDecodeError):
                continue

            # Get file metadata
            stat = file_path.stat()
            created_at = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc)
            updated_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

            if sync_at and updated_at <= sync_at:
                continue

            # Extract title from first heading or filename
            title = self._extract_title(content, file_path.stem)

            # Extract tags using Bear's syntax (#tag/subtag)
            tags = self._extract_tags(content)

            # Generate source ID
            note_id = hashlib.sha1(file_path.as_posix().encode()).hexdigest()[:16]
            source_id = f"bear_export:note:{note_id}"

            # Extract internal note links
            links = self._extract_note_links(content)
            if links:
                note_links[source_id] = links

            unit = KnowledgeUnit(
                source_project=SourceProject.BEAR_EXPORT,
                source_id=source_id,
                source_entity_type="note",
                title=title or "Untitled Bear note",
                content=content,
                content_type=ContentType.ARTIFACT,
                metadata={
                    "path": str(file_path),
                    "tags": tags,
                    "hierarchical_tags": self._group_hierarchical_tags(tags),
                    "file_name": file_path.name,
                },
                tags=tags,
                created_at=created_at,
                updated_at=updated_at,
            )
            result.units.append(unit)
            note_titles[title.lower()] = source_id

        # Create edges for note links
        for source_id, linked_titles in note_links.items():
            for linked_title in linked_titles:
                target_id = note_titles.get(linked_title.lower())
                if target_id:
                    result.edges.append(
                        KnowledgeEdge(
                            id=self._edge_id(source_id, target_id, "references"),
                            from_unit_id=source_id,
                            to_unit_id=target_id,
                            relation=EdgeRelation.REFERENCES,
                            source=EdgeSource.SOURCE,
                            metadata={
                                "source_project": SourceProject.BEAR_EXPORT.value,
                                "relation_type": "note_link",
                            },
                        )
                    )

        result.units.sort(key=lambda u: (u.created_at, u.source_id))
        result.edges.sort(key=lambda e: (e.from_unit_id, e.to_unit_id))
        return result

    def _find_markdown_files(self, root: Path) -> list[Path]:
        """Find all markdown files in root directory."""
        if root.is_file():
            return [root] if root.suffix.lower() in {".md", ".markdown"} else []
        files: list[Path] = []
        for pattern in ("*.md", "*.markdown"):
            files.extend(root.rglob(pattern))
        return sorted(f for f in files if f.is_file())

    def _extract_title(self, content: str, fallback: str) -> str:
        """Extract title from first heading or use filename."""
        for line in content.splitlines():
            match = re.match(r"^\s*#+\s+(.+)$", line)
            if match:
                return match.group(1).strip()
        return fallback

    def _extract_tags(self, content: str) -> list[str]:
        """Extract Bear-style tags (#tag/subtag)."""
        tags: set[str] = set()
        for match in TAG_RE.finditer(content):
            tag = match.group(1).lower()
            tags.add(tag)
        return sorted(tags)

    def _group_hierarchical_tags(self, tags: list[str]) -> dict[str, list[str]]:
        """Group tags by their hierarchy."""
        hierarchy: dict[str, list[str]] = {}
        for tag in tags:
            parts = tag.split("/")
            if len(parts) > 1:
                parent = parts[0]
                if parent not in hierarchy:
                    hierarchy[parent] = []
                hierarchy[parent].append(tag)
        return hierarchy

    def _extract_note_links(self, content: str) -> list[str]:
        """Extract internal note links [[Note Title]]."""
        links: list[str] = []
        for match in re.finditer(r"\[\[([^\]]+)\]\]", content):
            links.append(match.group(1).strip())
        return links

    def _edge_id(self, from_id: str, to_id: str, relation_type: str) -> str:
        """Generate edge ID."""
        raw = "|".join([SourceProject.BEAR_EXPORT.value, relation_type, from_id, to_id])
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
        return f"bear-export-{relation_type}-{digest}"

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
