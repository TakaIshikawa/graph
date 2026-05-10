"""Adapter for Notion export archives (ZIP or directory) with HTML/Markdown content."""

from __future__ import annotations

import hashlib
import re
import zipfile
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class _HTMLTextExtractor(HTMLParser):
    """Extract text content from HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        _ = attrs  # Unused but required by interface
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


class NotionExportAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "notion_export"

    @property
    def entity_types(self) -> list[str]:
        return ["page", "database"]

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
        ingest_pages = not entity_types or "page" in entity_types
        ingest_databases = not entity_types or "database" in entity_types

        # Track hierarchical relationships
        page_hierarchy: dict[str, str] = {}  # child_id -> parent_id

        # Handle ZIP file
        if root.is_file() and root.suffix.lower() == ".zip":
            # First pass: create all units
            path_to_source_id: dict[Path, str] = {}
            with zipfile.ZipFile(root, "r") as zf:
                for entry in zf.namelist():
                    entry_path = Path(entry)
                    if self._is_content_file(entry_path):
                        content = zf.read(entry).decode("utf-8", errors="ignore")
                        unit = self._parse_file(entry_path, content, root)
                        if unit is None:
                            continue
                        if not self._should_ingest(unit, ingest_pages, ingest_databases):
                            continue
                        if sync_at and unit.updated_at <= sync_at:
                            continue
                        result.units.append(unit)
                        path_to_source_id[entry_path] = unit.source_id

            # Second pass: establish hierarchical relationships
            all_files = set(path_to_source_id.keys())
            for file_path, source_id in path_to_source_id.items():
                parent_file = self._find_parent_page_file(file_path, all_files)
                if parent_file and parent_file in path_to_source_id:
                    page_hierarchy[source_id] = path_to_source_id[parent_file]

        # Handle directory
        elif root.is_dir():
            # First pass: create all units
            path_to_source_id: dict[Path, str] = {}
            for file_path in self._find_content_files(root):
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                except (OSError, UnicodeDecodeError):
                    continue
                unit = self._parse_file(file_path, content, root)
                if unit is None:
                    continue
                if not self._should_ingest(unit, ingest_pages, ingest_databases):
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)
                path_to_source_id[file_path] = unit.source_id

            # Second pass: establish hierarchical relationships
            all_files = set(path_to_source_id.keys())
            for file_path, source_id in path_to_source_id.items():
                parent_file = self._find_parent_page_file(file_path, all_files)
                if parent_file and parent_file in path_to_source_id:
                    page_hierarchy[source_id] = path_to_source_id[parent_file]

        # Create hierarchical edges
        included_ids = {unit.source_id for unit in result.units}
        for child_id, parent_id in page_hierarchy.items():
            if child_id in included_ids and parent_id in included_ids:
                result.edges.append(
                    KnowledgeEdge(
                        id=self._edge_id(parent_id, child_id, "contains"),
                        from_unit_id=parent_id,
                        to_unit_id=child_id,
                        relation=EdgeRelation.CONTAINS,
                        source=EdgeSource.SOURCE,
                        metadata={
                            "source_project": SourceProject.NOTION_EXPORT.value,
                            "relation_type": "page_hierarchy",
                        },
                    )
                )

        result.units.sort(key=lambda u: (u.created_at, u.source_id))
        result.edges.sort(key=lambda e: (e.from_unit_id, e.to_unit_id))
        return result

    def _is_content_file(self, path: Path) -> bool:
        """Check if file is a Notion content file (HTML or Markdown)."""
        suffix = path.suffix.lower()
        return suffix in {".html", ".md", ".markdown"}

    def _find_content_files(self, root: Path) -> list[Path]:
        """Find all HTML and Markdown files in directory."""
        files: list[Path] = []
        for pattern in ("*.html", "*.md", "*.markdown"):
            files.extend(root.rglob(pattern))
        return sorted(f for f in files if f.is_file())

    def _find_parent_page_file(self, child_path: Path, all_files: set[Path]) -> Path | None:
        """Find the parent page file for a given child page.

        Notion exports nested pages in subdirectories, with the parent page file
        in the same directory as the child's parent directory, with a matching name.

        For example:
        - Parent Page abc123.md
        - Parent Page abc123/
        - Parent Page abc123/Child Page def456.md
        """
        # If file is in a subdirectory, look for parent page file
        parent_dir = child_path.parent
        if parent_dir == child_path.parent.parent:
            # Already at root, no parent
            return None

        # Look for a page file with the same name as the parent directory
        parent_dir_name = parent_dir.name

        # Try to find a file in the grandparent directory with matching name
        grandparent_dir = parent_dir.parent
        for candidate in all_files:
            if candidate.parent == grandparent_dir:
                # Check if stem matches directory name (ignoring UUID suffixes)
                candidate_title = self._extract_title_from_filename(candidate.stem)
                dir_title = self._extract_title_from_filename(parent_dir_name)
                if candidate_title == dir_title:
                    return candidate

        return None

    def _parse_file(
        self, path: Path, content: str, root: Path
    ) -> KnowledgeUnit | None:
        """Parse a Notion export file into a KnowledgeUnit."""
        if path.suffix.lower() == ".html":
            return self._parse_html(path, content, root)
        else:
            return self._parse_markdown(path, content, root)

    def _parse_html(
        self, path: Path, content: str, root: Path
    ) -> KnowledgeUnit | None:
        """Parse HTML export file."""
        # Extract title from filename (Notion format: "Title UUID.html")
        title = self._extract_title_from_filename(path.stem)

        # Extract text content
        parser = _HTMLTextExtractor()
        parser.feed(content)
        text_content = parser.get_text()

        if not title and not text_content:
            return None

        # Extract metadata from HTML
        metadata = self._extract_html_metadata(content, path, root)

        # Determine entity type
        entity_type = "database" if metadata.get("is_database") else "page"

        # Generate source ID
        page_id = metadata.get("page_id") or self._generate_id_from_path(path)
        source_id = f"notion_export:{entity_type}:{page_id}"

        # Extract properties from content
        properties = self._extract_properties_from_html(content)

        return KnowledgeUnit(
            source_project=SourceProject.NOTION_EXPORT,
            source_id=source_id,
            source_entity_type=entity_type,
            title=title or "Untitled Notion page",
            content=text_content or title or "",
            content_type=ContentType.ARTIFACT,
            metadata={
                "path": str(path),
                "page_id": page_id,
                "properties": properties,
                "media_files": metadata.get("media_files", []),
                "is_database": metadata.get("is_database", False),
            },
            tags=self._extract_tags(properties),
            created_at=metadata.get("created_at") or datetime.now(timezone.utc),
            updated_at=metadata.get("updated_at") or datetime.now(timezone.utc),
        )

    def _parse_markdown(
        self, path: Path, content: str, root: Path
    ) -> KnowledgeUnit | None:
        """Parse Markdown export file."""
        # Extract title from filename or first heading
        title = self._extract_title_from_filename(path.stem)
        if not title:
            title = self._extract_first_heading(content)

        if not title and not content.strip():
            return None

        # Extract metadata
        metadata = self._extract_markdown_metadata(content, path, root)

        # Determine entity type
        entity_type = "database" if metadata.get("is_database") else "page"

        # Generate source ID
        page_id = metadata.get("page_id") or self._generate_id_from_path(path)
        source_id = f"notion_export:{entity_type}:{page_id}"

        # Extract properties
        properties = self._extract_properties_from_markdown(content)

        return KnowledgeUnit(
            source_project=SourceProject.NOTION_EXPORT,
            source_id=source_id,
            source_entity_type=entity_type,
            title=title or "Untitled Notion page",
            content=content,
            content_type=ContentType.ARTIFACT,
            metadata={
                "path": str(path),
                "page_id": page_id,
                "properties": properties,
                "media_files": metadata.get("media_files", []),
                "is_database": metadata.get("is_database", False),
            },
            tags=self._extract_tags(properties),
            created_at=metadata.get("created_at") or datetime.now(timezone.utc),
            updated_at=metadata.get("updated_at") or datetime.now(timezone.utc),
        )

    def _extract_title_from_filename(self, stem: str) -> str:
        """Extract title from Notion filename (removes UUID suffix)."""
        # Notion exports files as "Title UUID" or "Title"
        # UUID pattern: 32 hex characters (without dashes) or other identifier patterns
        # Try multiple patterns
        patterns = [
            r"\s+[a-f0-9]{32}$",  # 32 hex chars
            r"\s+[a-f0-9]{6,}$",  # 6+ hex chars
            r"\s+[a-z0-9]{6,}$",  # 6+ alphanumeric chars
        ]
        cleaned = stem
        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def _extract_first_heading(self, content: str) -> str:
        """Extract first markdown heading from content."""
        for line in content.splitlines():
            match = re.match(r"^\s*#+\s+(.+)$", line)
            if match:
                return match.group(1).strip()
        return ""

    def _generate_id_from_path(self, path: Path) -> str:
        """Generate a stable ID from file path."""
        raw = path.as_posix()
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    def _extract_html_metadata(
        self, content: str, path: Path, _root: Path
    ) -> dict[str, Any]:
        """Extract metadata from HTML content."""
        metadata: dict[str, Any] = {}

        # Look for Notion page ID in HTML comments or meta tags
        page_id_match = re.search(r'data-page-id="([^"]+)"', content)
        if page_id_match:
            metadata["page_id"] = page_id_match.group(1)

        # Check if it's a database view
        if "notion-database" in content or "notion-table" in content:
            metadata["is_database"] = True

        # Extract media file references
        media_files = []
        for match in re.finditer(r'src="([^"]+\.(png|jpg|jpeg|gif|pdf|mp4|mov))"', content, re.IGNORECASE):
            media_files.append(match.group(1))
        if media_files:
            metadata["media_files"] = media_files

        # Use file modification time as timestamp
        stat = path.stat() if isinstance(path, Path) and path.exists() else None
        if stat:
            metadata["created_at"] = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc)
            metadata["updated_at"] = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

        return metadata

    def _extract_markdown_metadata(
        self, content: str, path: Path, _root: Path
    ) -> dict[str, Any]:
        """Extract metadata from Markdown content."""
        metadata: dict[str, Any] = {}

        # Check if it's a database view (CSV-like tables)
        if re.search(r"\|.*\|.*\|", content):
            metadata["is_database"] = True

        # Extract media file references
        media_files = []
        for match in re.finditer(r'!\[.*?\]\(([^)]+\.(png|jpg|jpeg|gif|pdf|mp4|mov))\)', content, re.IGNORECASE):
            media_files.append(match.group(1))
        if media_files:
            metadata["media_files"] = media_files

        # Use file modification time
        stat = path.stat() if isinstance(path, Path) and path.exists() else None
        if stat:
            metadata["created_at"] = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc)
            metadata["updated_at"] = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

        return metadata

    def _extract_properties_from_html(self, content: str) -> dict[str, Any]:
        """Extract Notion properties from HTML."""
        properties: dict[str, Any] = {}

        # Look for property sections in HTML
        # Notion often exports properties as definition lists or specific divs
        property_pattern = r'<div class="property[^"]*"[^>]*>\s*<span[^>]*>([^<]+)</span>\s*:\s*<span[^>]*>([^<]+)</span>'
        for match in re.finditer(property_pattern, content):
            key = match.group(1).strip()
            value = match.group(2).strip()
            properties[key] = value

        return properties

    def _extract_properties_from_markdown(self, content: str) -> dict[str, Any]:
        """Extract Notion properties from Markdown."""
        properties: dict[str, Any] = {}

        # Look for YAML frontmatter
        if content.startswith("---"):
            lines = content.split("\n")
            frontmatter_lines = []
            for line in lines[1:]:
                if line.strip() == "---":
                    break
                frontmatter_lines.append(line)

            if frontmatter_lines:
                try:
                    import yaml
                    parsed = yaml.safe_load("\n".join(frontmatter_lines))
                    if isinstance(parsed, dict):
                        properties = parsed
                except Exception:
                    pass

        # Look for property-like patterns
        for line in content.splitlines()[:20]:  # Check first 20 lines
            match = re.match(r"^([A-Za-z][A-Za-z0-9 _-]+):\s*(.+)$", line)
            if match:
                properties[match.group(1).strip()] = match.group(2).strip()

        return properties

    def _extract_tags(self, properties: dict[str, Any]) -> list[str]:
        """Extract tags from properties."""
        tags: list[str] = []

        # Look for tag-related properties (case-insensitive)
        for prop_key, prop_value in properties.items():
            key_lower = prop_key.lower()
            if key_lower in ("tags", "tag", "labels", "label", "categories", "category"):
                if isinstance(prop_value, list):
                    tags.extend(str(v).strip() for v in prop_value if v)
                elif isinstance(prop_value, str):
                    tags.extend(
                        t.strip()
                        for t in re.split(r"[,;|]", prop_value)
                        if t.strip()
                    )

        # Deduplicate and normalize
        seen: set[str] = set()
        normalized_tags: list[str] = []
        for tag in tags:
            normalized = tag.strip().removeprefix("#").lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                normalized_tags.append(normalized)

        return normalized_tags

    def _should_ingest(
        self, unit: KnowledgeUnit, ingest_pages: bool, ingest_databases: bool
    ) -> bool:
        """Check if unit should be ingested based on entity type filters."""
        if unit.source_entity_type == "page":
            return ingest_pages
        elif unit.source_entity_type == "database":
            return ingest_databases
        return True

    def _edge_id(self, from_id: str, to_id: str, relation_type: str) -> str:
        """Generate edge ID."""
        raw = "|".join([SourceProject.NOTION_EXPORT.value, relation_type, from_id, to_id])
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
        return f"notion-export-{relation_type}-{digest}"

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
