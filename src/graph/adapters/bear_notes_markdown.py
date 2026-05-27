"""Adapter for Bear notes exported as Markdown files."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, iter_paths, parse_datetime, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_TAG_RE = re.compile(r"(?<!\w)#([A-Za-z0-9_][A-Za-z0-9_/-]*)")


class BearNotesMarkdownAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "bear_notes_markdown"

    @property
    def entity_types(self) -> list[str]:
        return ["note"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "note" not in entity_types:
            return result
        sync_at = since.last_sync_at if since else None
        if sync_at and sync_at.tzinfo is None:
            sync_at = sync_at.replace(tzinfo=timezone.utc)
        root = Path(self.path).expanduser() if self.path else Path(".")
        for path in iter_paths(self.path, {".md", ".markdown"}):
            unit = self._unit(path, root)
            if unit and (sync_at is None or unit.updated_at > sync_at):
                result.units.append(unit)
        result.units.sort(key=lambda unit: (unit.created_at, unit.source_id))
        return result

    def _unit(self, path: Path, root: Path) -> KnowledgeUnit | None:
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None
        frontmatter, body = _frontmatter(content)
        try:
            relative_path = str(path.relative_to(root))
        except ValueError:
            relative_path = path.name
        title = _title(frontmatter, body, path)
        tags = _tags(frontmatter, body)
        modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        created_at = parse_datetime(frontmatter.get("created") or frontmatter.get("created_at")) or modified_at
        return KnowledgeUnit(
            source_project=SourceProject.BEAR_NOTES_MARKDOWN,
            source_id=digest_source_id("bear_notes_markdown", relative_path),
            source_entity_type="note",
            title=title,
            content=body.strip() or content,
            content_type=ContentType.ARTIFACT,
            metadata=clean_metadata(
                {
                    "title": title,
                    "relative_path": relative_path,
                    "tags": tags,
                    "frontmatter": frontmatter,
                    "modified_at": modified_at.isoformat(),
                }
            ),
            tags=tags,
            created_at=created_at,
            updated_at=modified_at,
        )


def _frontmatter(content: str) -> tuple[dict[str, Any], str]:
    match = _FRONTMATTER_RE.match(content)
    if not match:
        return {}, content
    data: dict[str, Any] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            data[key.strip()] = value.strip().strip("'\"")
    return data, content[match.end() :]


def _title(frontmatter: dict[str, Any], body: str, path: Path) -> str:
    if str(frontmatter.get("title") or "").strip():
        return str(frontmatter["title"]).strip()
    for line in body.splitlines():
        match = re.match(r"\s*#\s+(.+?)\s*$", line)
        if match:
            return match.group(1).strip()
    return path.stem


def _tags(frontmatter: dict[str, Any], body: str) -> list[str]:
    raw = split_values(frontmatter.get("tags"))
    raw.extend(match.group(1) for match in _TAG_RE.finditer(body))
    tags: list[str] = []
    for tag in raw:
        text = str(tag).strip().strip("[]").casefold()
        if text and text not in tags:
            tags.append(text)
    return tags
