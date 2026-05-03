"""Adapter for documents with YAML frontmatter."""

from __future__ import annotations

import hashlib
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any

import yaml

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


FRONTMATTER_DELIMITER = "---"
DEFAULT_FILE_EXTENSIONS = {".md", ".markdown", ".txt", ".text", ".rst", ".adoc", ".asciidoc"}


class YamlFrontmatterAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "yaml_frontmatter"

    @property
    def entity_types(self) -> list[str]:
        return ["yaml_frontmatter"]

    def __init__(
        self,
        path: str = "",
        *,
        root_path: str = "",
        source_id_root: str | None = None,
        file_extensions: list[str] | None = None,
    ) -> None:
        self.path = path or root_path
        self.source_id_root = source_id_root
        self.file_extensions = set(file_extensions) if file_extensions else DEFAULT_FILE_EXTENSIONS

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "yaml_frontmatter" not in entity_types:
            return result

        root = Path(self.path).expanduser()
        if not root.exists():
            return result

        files = self._document_files(root)
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
            text = file_path.read_text(encoding="utf-8", errors="replace")
            frontmatter, body, parse_error = self._split_frontmatter(text)

            # Build metadata
            metadata: dict[str, Any] = {
                "source_file": relative_path,
                "frontmatter": self._jsonable(frontmatter),
            }
            if parse_error:
                metadata["frontmatter_parse_error"] = parse_error

            # Extract common metadata fields
            title = self._extract_title(frontmatter, body, file_path)
            tags = self._extract_tags(frontmatter)
            created_at = self._extract_datetime(
                frontmatter,
                ("created", "created_at", "date", "created_date"),
                fallback=datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc),
            )
            updated_at = self._extract_datetime(
                frontmatter,
                ("updated", "updated_at", "modified", "lastmod", "modified_date"),
                fallback=file_updated_at,
            )

            # Add author, categories, and other common fields to metadata
            if "author" in frontmatter or "authors" in frontmatter:
                authors = frontmatter.get("authors", frontmatter.get("author"))
                metadata["authors"] = self._normalize_list(authors)

            if "category" in frontmatter or "categories" in frontmatter:
                categories = frontmatter.get("categories", frontmatter.get("category"))
                metadata["categories"] = self._normalize_list(categories)

            if "description" in frontmatter:
                metadata["description"] = str(frontmatter["description"])

            if "source_url" in frontmatter or "url" in frontmatter:
                source_url = frontmatter.get("source_url", frontmatter.get("url"))
                if source_url:
                    metadata["source_url"] = str(source_url)

            result.units.append(
                KnowledgeUnit(
                    source_project=SourceProject.YAML_FRONTMATTER,
                    source_id=self._source_id(relative_path),
                    source_entity_type="yaml_frontmatter",
                    title=title,
                    content=body,
                    content_type=ContentType.ARTIFACT,
                    metadata=metadata,
                    tags=tags,
                    created_at=created_at,
                    updated_at=updated_at,
                )
            )

        return result

    def _document_files(self, root: Path) -> list[Path]:
        """Find all document files with supported extensions."""
        if root.is_file():
            return [root] if root.suffix.lower() in self.file_extensions else []
        if not root.is_dir():
            return []

        all_files: list[Path] = []
        for ext in self.file_extensions:
            pattern = f"*{ext}"
            all_files.extend(path for path in root.rglob(pattern) if path.is_file())
        return sorted(all_files)

    def _split_frontmatter(self, text: str) -> tuple[dict[str, Any], str, str | None]:
        """Split text into frontmatter dict, body, and optional parse error."""
        lines = text.splitlines()
        if not lines or lines[0].strip() != FRONTMATTER_DELIMITER:
            return {}, text, None

        # Find the closing delimiter
        for index, line in enumerate(lines[1:], start=1):
            if line.strip() != FRONTMATTER_DELIMITER:
                continue

            # Extract frontmatter and body
            raw_frontmatter = "\n".join(lines[1:index])
            body = "\n".join(lines[index + 1 :])
            if text.endswith("\n"):
                body += "\n"

            # Parse YAML
            try:
                data = yaml.safe_load(raw_frontmatter) or {}
            except yaml.YAMLError as exc:
                return {}, text, str(exc)

            if not isinstance(data, dict):
                return {}, body, "frontmatter is not a mapping"

            return data, body, None

        # No closing delimiter found
        return {}, text, None

    def _extract_title(self, frontmatter: dict[str, Any], body: str, file_path: Path) -> str:
        """Extract title from frontmatter, body, or filename."""
        title = frontmatter.get("title")
        if title is not None and str(title).strip():
            return str(title).strip()

        # Try to extract from first heading in body
        body_lines = body.strip().split("\n")
        for line in body_lines[:10]:  # Check first 10 lines
            stripped = line.strip()
            if stripped.startswith("# "):
                heading = stripped[2:].strip()
                if heading:
                    return heading

        # Fallback to filename
        return file_path.stem

    def _extract_tags(self, frontmatter: dict[str, Any]) -> list[str]:
        """Extract and normalize tags from frontmatter."""
        tags_value = frontmatter.get("tags", frontmatter.get("tag"))
        return self._normalize_list(tags_value)

    def _normalize_list(self, value: Any) -> list[str]:
        """Normalize a value to a list of strings."""
        if isinstance(value, str):
            # Split by common delimiters
            items = [item.strip() for item in value.replace(",", " ").replace(";", " ").split()]
        elif isinstance(value, list | tuple | set):
            items = [str(item).strip() for item in value]
        else:
            return []

        # Clean and deduplicate
        result: list[str] = []
        for item in items:
            clean = item.removeprefix("#").strip().lower() if item else ""
            if clean and clean not in result:
                result.append(clean)
        return result

    def _extract_datetime(
        self,
        frontmatter: dict[str, Any],
        keys: tuple[str, ...],
        *,
        fallback: datetime,
    ) -> datetime:
        """Extract datetime from frontmatter with fallback."""
        for key in keys:
            parsed = self._parse_datetime(frontmatter.get(key))
            if parsed is not None:
                return parsed
        return fallback

    def _parse_datetime(self, value: Any) -> datetime | None:
        """Parse a value into a datetime object."""
        if isinstance(value, datetime):
            parsed = value
        elif isinstance(value, date):
            parsed = datetime.combine(value, time.min)
        elif isinstance(value, str):
            raw = value.strip()
            if not raw:
                return None
            try:
                parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                return None
        else:
            return None

        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _source_id(self, relative_path: str) -> str:
        """Generate deterministic source ID from relative path."""
        digest = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:16]
        return f"yaml_frontmatter:{digest}"

    def _relative_path(self, path: Path, source_root: Path) -> str:
        """Get relative path from source root."""
        try:
            return path.relative_to(source_root).as_posix()
        except ValueError:
            return path.as_posix()

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

    def _jsonable(self, value: Any) -> Any:
        """Convert value to JSON-compatible format."""
        if isinstance(value, dict):
            return {str(key): self._jsonable(item) for key, item in value.items()}
        if isinstance(value, list | tuple | set):
            return [self._jsonable(item) for item in value]
        if isinstance(value, datetime | date):
            return value.isoformat()
        if isinstance(value, str | int | float | bool) or value is None:
            return value
        return str(value)
