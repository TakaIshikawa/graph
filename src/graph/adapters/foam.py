"""Adapter for Foam workspace markdown notes (VS Code-based PKM)."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState

# Wikilinks: [[target]] or [[target|alias]]
WIKILINK_RE = re.compile(r"(?<!!)\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")

# Note embeds: ![[target]]
EMBED_RE = re.compile(r"!\[\[([^\]]+)\]\]")

# Inline tags: #tag (not inside code spans/fences)
TAG_RE = re.compile(r"(?<![\w/])#([\w][\w-]*(?:/[\w][\w-]*)*)", re.UNICODE)

# YAML frontmatter boundaries
FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)

# Daily note filename pattern
DAILY_NOTE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})\.md$")

# Markdown link definition references (Foam generates these)
LINK_DEF_RE = re.compile(r"^\[([^\]]+)\]:\s*(.+)$", re.MULTILINE)


class FoamWorkspaceAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "foam_workspace"

    @property
    def entity_types(self) -> list[str]:
        return ["note", "daily_note"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        root = Path(self.path).expanduser() if self.path else None
        if root is None or not root.exists():
            return result

        sync_at = self._sync_datetime(since) if since else None
        allowed_types = set(entity_types) if entity_types else None

        # Track note links for edge creation
        note_titles: dict[str, str] = {}  # normalized title/path -> source_id
        note_links: dict[str, list[str]] = {}  # source_id -> linked targets
        note_embeds: dict[str, list[str]] = {}  # source_id -> embedded targets

        files = self._find_markdown_files(root)
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

            # Determine entity type
            entity_type = self._classify_note(file_path, root)
            if allowed_types and entity_type not in allowed_types:
                continue

            # Parse frontmatter
            frontmatter, body = self._parse_frontmatter(content)

            # Extract title
            title = frontmatter.get("title") or self._extract_title(body, file_path.stem)

            # Extract tags from frontmatter + inline
            tags = self._extract_tags(frontmatter, body)

            # Extract wikilinks
            links = WIKILINK_RE.findall(body)

            # Extract embeds
            embeds = EMBED_RE.findall(body)

            # Extract link reference definitions
            link_defs = dict(LINK_DEF_RE.findall(content))

            # Generate source ID
            rel_path = file_path.relative_to(root).as_posix()
            note_id = hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:16]
            source_id = f"foam_workspace:{entity_type}:{note_id}"

            metadata: dict = {
                "path": str(file_path),
                "relative_path": rel_path,
                "file_name": file_path.name,
            }
            if frontmatter:
                metadata["frontmatter"] = frontmatter
            if embeds:
                metadata["embeds"] = embeds
            if link_defs:
                metadata["link_definitions"] = link_defs

            # Check for daily note date
            daily_match = DAILY_NOTE_RE.search(file_path.name)
            if daily_match:
                metadata["date"] = daily_match.group(1)

            unit = KnowledgeUnit(
                source_project=SourceProject.FOAM_WORKSPACE,
                source_id=source_id,
                source_entity_type=entity_type,
                title=title or "Untitled",
                content=content,
                content_type=ContentType.ARTIFACT,
                metadata=metadata,
                tags=tags,
                created_at=created_at,
                updated_at=updated_at,
            )
            result.units.append(unit)

            # Register for link resolution
            stem_key = file_path.stem.lower()
            note_titles[stem_key] = source_id
            note_titles[rel_path.lower()] = source_id
            if title:
                note_titles[title.lower()] = source_id

            if links:
                note_links[source_id] = [l.strip().lower() for l in links]
            if embeds:
                note_embeds[source_id] = [e.strip().lower() for e in embeds]

        # Create edges for wikilinks
        for source_id, linked_targets in note_links.items():
            for target in linked_targets:
                target_id = self._resolve_link(target, note_titles)
                if target_id and target_id != source_id:
                    result.edges.append(
                        KnowledgeEdge(
                            id=self._edge_id(source_id, target_id, "references"),
                            from_unit_id=source_id,
                            to_unit_id=target_id,
                            relation=EdgeRelation.REFERENCES,
                            source=EdgeSource.SOURCE,
                            metadata={
                                "source_project": SourceProject.FOAM_WORKSPACE.value,
                                "relation_type": "wikilink",
                            },
                        )
                    )

        # Create edges for embeds
        for source_id, embedded_targets in note_embeds.items():
            for target in embedded_targets:
                target_id = self._resolve_link(target, note_titles)
                if target_id and target_id != source_id:
                    result.edges.append(
                        KnowledgeEdge(
                            id=self._edge_id(source_id, target_id, "contains"),
                            from_unit_id=source_id,
                            to_unit_id=target_id,
                            relation=EdgeRelation.CONTAINS,
                            source=EdgeSource.SOURCE,
                            metadata={
                                "source_project": SourceProject.FOAM_WORKSPACE.value,
                                "relation_type": "embed",
                            },
                        )
                    )

        result.units.sort(key=lambda u: (u.created_at, u.source_id))
        result.edges.sort(key=lambda e: (e.from_unit_id, e.to_unit_id))
        return result

    def _find_markdown_files(self, root: Path) -> list[Path]:
        """Find all markdown files, excluding node_modules and .git."""
        files: list[Path] = []
        for f in root.rglob("*.md"):
            parts = f.relative_to(root).parts
            if any(p.startswith(".git") or p == "node_modules" for p in parts):
                continue
            if f.is_file():
                files.append(f)
        return sorted(files)

    def _classify_note(self, file_path: Path, root: Path) -> str:
        """Classify a note as 'daily_note' or 'note'."""
        rel = file_path.relative_to(root).as_posix()
        # Foam daily notes typically live in .foam/daily/ or journal/
        if ".foam/daily/" in rel or "journal/" in rel:
            if DAILY_NOTE_RE.search(file_path.name):
                return "daily_note"
        # Also catch date-named files anywhere
        if DAILY_NOTE_RE.search(file_path.name):
            return "daily_note"
        return "note"

    def _parse_frontmatter(self, content: str) -> tuple[dict, str]:
        """Parse YAML frontmatter, returning (metadata_dict, body)."""
        match = FRONTMATTER_RE.match(content)
        if not match:
            return {}, content
        raw = match.group(1)
        try:
            parsed = yaml.safe_load(raw)
            if not isinstance(parsed, dict):
                return {}, content
        except yaml.YAMLError:
            return {}, content
        body = content[match.end():]
        return parsed, body

    def _extract_title(self, body: str, fallback: str) -> str:
        """Extract title from first heading or use filename stem."""
        for line in body.splitlines():
            m = re.match(r"^\s*#+\s+(.+)$", line)
            if m:
                return m.group(1).strip()
        return fallback

    def _extract_tags(self, frontmatter: dict, body: str) -> list[str]:
        """Combine frontmatter tags and inline #tags."""
        tags: set[str] = set()

        # Frontmatter tags
        fm_tags = frontmatter.get("tags", [])
        if isinstance(fm_tags, list):
            for t in fm_tags:
                if t:
                    tags.add(str(t).strip().lower())
        elif isinstance(fm_tags, str):
            for t in re.split(r"[,;\s]+", fm_tags):
                t = t.strip().lower()
                if t:
                    tags.add(t)

        # Inline tags
        for m in TAG_RE.finditer(body):
            tags.add(m.group(1).lower())

        return sorted(tags)

    def _resolve_link(self, target: str, titles: dict[str, str]) -> str | None:
        """Resolve a wikilink target to a source_id."""
        # Try exact match
        sid = titles.get(target)
        if sid:
            return sid
        # Try with .md stripped
        if target.endswith(".md"):
            sid = titles.get(target[:-3])
            if sid:
                return sid
        # Try basename only (for path-style links like path/to/file)
        if "/" in target:
            basename = target.rsplit("/", 1)[-1]
            sid = titles.get(basename)
            if sid:
                return sid
        return None

    def _edge_id(self, from_id: str, to_id: str, relation_type: str) -> str:
        raw = "|".join([SourceProject.FOAM_WORKSPACE.value, relation_type, from_id, to_id])
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
        return f"foam-workspace-{relation_type}-{digest}"

    def _sync_datetime(self, since: SyncState) -> datetime:
        value = since.last_sync_at
        if isinstance(value, datetime):
            parsed = value
        else:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
