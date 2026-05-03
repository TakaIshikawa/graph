"""Adapter for Markdown files with YAML-style frontmatter."""

from __future__ import annotations

import hashlib
import re
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any

import yaml

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


FRONTMATTER_DELIMITER = "---"
HEADING_RE = re.compile(r"^\s{0,3}#\s+(.+?)\s*#*\s*$", re.MULTILINE)
TAG_SPLIT_RE = re.compile(r"[,;|]")
MAPPED_FRONTMATTER_KEYS = {
    "title",
    "tags",
    "tag",
    "created",
    "created_at",
    "date",
    "updated",
    "updated_at",
    "modified",
    "lastmod",
    "source_url",
    "source-url",
    "url",
    "canonical_url",
    "aliases",
    "alias",
}


class MarkdownFrontmatterAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "markdown_frontmatter"

    @property
    def entity_types(self) -> list[str]:
        return ["markdown_frontmatter"]

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
        if entity_types and "markdown_frontmatter" not in entity_types:
            return result

        root = Path(self.path).expanduser()
        if not root.exists():
            return result

        files = self._markdown_files(root)
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
            source_url = self._first_text(
                frontmatter,
                ("source_url", "source-url", "url", "canonical_url"),
            )
            frontmatter_metadata = self._unmapped_frontmatter(frontmatter)
            metadata = {
                "source_file": relative_path,
                "frontmatter": self._jsonable(frontmatter),
                "frontmatter_metadata": self._jsonable(frontmatter_metadata),
                "aliases": self._normalize_aliases(
                    frontmatter.get("aliases", frontmatter.get("alias"))
                ),
            }
            if source_url:
                metadata["source_url"] = source_url
            if parse_error:
                metadata["frontmatter_parse_error"] = parse_error

            result.units.append(
                KnowledgeUnit(
                    source_project=SourceProject.MARKDOWN_FRONTMATTER,
                    source_id=self._source_id(relative_path),
                    source_entity_type="markdown_frontmatter",
                    title=self._title(frontmatter, body, file_path),
                    content=body,
                    content_type=ContentType.ARTIFACT,
                    metadata=metadata,
                    tags=self._normalize_tags(
                        frontmatter.get("tags", frontmatter.get("tag"))
                    ),
                    created_at=self._frontmatter_datetime(
                        frontmatter,
                        ("created", "created_at", "date"),
                        fallback=file_updated_at,
                    ),
                    updated_at=self._frontmatter_datetime(
                        frontmatter,
                        ("updated", "updated_at", "modified", "lastmod"),
                        fallback=file_updated_at,
                    ),
                )
            )

        return result

    def _markdown_files(self, root: Path) -> list[Path]:
        if root.is_file():
            return [root] if root.suffix.lower() in {".md", ".markdown"} else []
        if not root.is_dir():
            return []
        return sorted(
            path
            for pattern in ("*.md", "*.markdown")
            for path in root.rglob(pattern)
            if path.is_file()
        )

    def _split_frontmatter(self, text: str) -> tuple[dict[str, Any], str, str | None]:
        lines = text.splitlines()
        if not lines or lines[0].strip() != FRONTMATTER_DELIMITER:
            return {}, text, None

        for index, line in enumerate(lines[1:], start=1):
            if line.strip() != FRONTMATTER_DELIMITER:
                continue

            raw_frontmatter = "\n".join(lines[1:index])
            body = "\n".join(lines[index + 1 :])
            if text.endswith("\n"):
                body += "\n"
            try:
                data = yaml.safe_load(raw_frontmatter) or {}
            except yaml.YAMLError as exc:
                return {}, text, str(exc)
            if not isinstance(data, dict):
                return {}, body, "frontmatter is not a mapping"
            return data, body, None

        return {}, text, None

    def _title(self, frontmatter: dict[str, Any], body: str, file_path: Path) -> str:
        title = frontmatter.get("title")
        if title is not None and str(title).strip():
            return str(title).strip()

        heading = HEADING_RE.search(body)
        if heading:
            return heading.group(1).strip().strip("#").strip()
        return file_path.stem

    def _normalize_tags(self, value: Any) -> list[str]:
        raw_tags: list[Any]
        if isinstance(value, str):
            raw_tags = TAG_SPLIT_RE.split(value)
        elif isinstance(value, list | tuple | set):
            raw_tags = list(value)
        else:
            raw_tags = []

        tags: list[str] = []
        for raw_tag in raw_tags:
            if isinstance(raw_tag, str) and TAG_SPLIT_RE.search(raw_tag):
                candidates = TAG_SPLIT_RE.split(raw_tag)
            else:
                candidates = [raw_tag]
            for candidate in candidates:
                tag = str(candidate).strip().removeprefix("#").strip().lower()
                if tag and tag not in tags:
                    tags.append(tag)
        return tags

    def _normalize_aliases(self, value: Any) -> list[str]:
        if isinstance(value, str):
            aliases = [value]
        elif isinstance(value, list | tuple | set):
            aliases = list(value)
        else:
            aliases = []
        return [alias for alias in (str(item).strip() for item in aliases) if alias]

    def _frontmatter_datetime(
        self,
        frontmatter: dict[str, Any],
        keys: tuple[str, ...],
        *,
        fallback: datetime,
    ) -> datetime:
        for key in keys:
            parsed = self._parse_datetime(frontmatter.get(key))
            if parsed is not None:
                return parsed
        return fallback

    def _parse_datetime(self, value: Any) -> datetime | None:
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

    def _first_text(self, frontmatter: dict[str, Any], keys: tuple[str, ...]) -> str:
        for key in keys:
            value = frontmatter.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    def _unmapped_frontmatter(self, frontmatter: dict[str, Any]) -> dict[str, Any]:
        return {
            str(key): value
            for key, value in frontmatter.items()
            if str(key) not in MAPPED_FRONTMATTER_KEYS
        }

    def _source_id(self, relative_path: str) -> str:
        digest = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:16]
        return f"markdown_frontmatter:{digest}"

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

    def _jsonable(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._jsonable(item) for key, item in value.items()}
        if isinstance(value, list | tuple | set):
            return [self._jsonable(item) for item in value]
        if isinstance(value, datetime | date):
            return value.isoformat()
        if isinstance(value, str | int | float | bool) or value is None:
            return value
        return str(value)
