"""Adapter for Notion Markdown exports."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


FRONT_MATTER_DELIMITER = "---"
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
PROPERTY_RE = re.compile(r"^([A-Za-z][A-Za-z0-9 _/-]{0,80}):\s*(.*)$")
TAG_PROPERTIES = {"tag", "tags"}
STATUS_PROPERTIES = {"status"}


class NotionMarkdownAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "notion_markdown"

    @property
    def entity_types(self) -> list[str]:
        return ["notion_page"]

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
        if entity_types and "notion_page" not in entity_types:
            return result

        root = Path(self.path).expanduser() if self.path else None
        if root is None or not root.exists():
            return result

        source_root = Path(self.source_id_root).expanduser() if self.source_id_root else root
        if root.is_file() and not self.source_id_root:
            source_root = root.parent

        sync_at = self._sync_timestamp(since) if since else None
        for path in self._markdown_paths(root):
            stat = path.stat()
            if sync_at is not None and stat.st_mtime <= sync_at:
                continue
            result.units.append(self._unit(path, source_root, stat))

        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _markdown_paths(self, root: Path) -> list[Path]:
        if root.is_file():
            return [root] if root.suffix.lower() in {".md", ".markdown"} else []
        if not root.is_dir():
            return []
        return sorted(
            path
            for suffix in ("*.md", "*.markdown")
            for path in root.rglob(suffix)
            if path.is_file()
        )

    def _unit(self, path: Path, source_root: Path, stat: Any) -> KnowledgeUnit:
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Could not decode Notion Markdown file {path}") from exc
        except OSError as exc:
            raise ValueError(f"Could not read Notion Markdown file {path}") from exc

        properties, raw_property_block, body = self._extract_properties(text)
        title = self._first_heading(body) or self._property_text(properties, "title") or path.stem
        relative_path = self._relative_path(path, source_root)
        return KnowledgeUnit(
            source_project=SourceProject.NOTION_MARKDOWN,
            source_id=f"notion_markdown:{relative_path}",
            source_entity_type="notion_page",
            title=title,
            content=body,
            content_type=ContentType.ARTIFACT,
            metadata={
                "path": relative_path,
                "properties": self._jsonable(properties),
                "raw_property_block": raw_property_block,
                "has_property_block": bool(raw_property_block),
            },
            tags=self._tags(properties),
            created_at=datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc),
            updated_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
        )

    def _extract_properties(self, text: str) -> tuple[dict[str, Any], str, str]:
        lines = text.splitlines()
        if lines and lines[0].strip() == FRONT_MATTER_DELIMITER:
            for index, line in enumerate(lines[1:], start=1):
                if line.strip() == FRONT_MATTER_DELIMITER:
                    raw = "\n".join(lines[1:index])
                    body = "\n".join(lines[index + 1 :])
                    if text.endswith("\n"):
                        body += "\n"
                    parsed = yaml.safe_load(raw) or {}
                    return (parsed if isinstance(parsed, dict) else {}), raw, body

        heading_end = self._leading_heading_end(lines)
        block_start = self._first_nonblank(lines, heading_end)
        if block_start is None or not PROPERTY_RE.match(lines[block_start]):
            return {}, "", text

        properties: dict[str, Any] = {}
        block_end = block_start
        while block_end < len(lines):
            line = lines[block_end]
            if not line.strip():
                break
            match = PROPERTY_RE.match(line)
            if match is None:
                break
            properties[match.group(1).strip()] = match.group(2).strip()
            block_end += 1

        raw = "\n".join(lines[block_start:block_end])
        body_lines = lines[:block_start] + lines[block_end:]
        while body_lines and not body_lines[0].strip():
            body_lines.pop(0)
        body = "\n".join(body_lines)
        if text.endswith("\n"):
            body += "\n"
        return properties, raw, body

    def _leading_heading_end(self, lines: list[str]) -> int:
        first = self._first_nonblank(lines, 0)
        if first is None or not HEADING_RE.match(lines[first]):
            return 0
        return first + 1

    def _first_nonblank(self, lines: list[str], start: int) -> int | None:
        for index in range(start, len(lines)):
            if lines[index].strip():
                return index
        return None

    def _first_heading(self, text: str) -> str:
        for line in text.splitlines():
            match = HEADING_RE.match(line)
            if match:
                return self._clean_heading(match.group(1))
        return ""

    def _clean_heading(self, value: str) -> str:
        return value.strip().strip("#").strip()

    def _property_text(self, properties: dict[str, Any], name: str) -> str:
        for key, value in properties.items():
            if str(key).strip().lower() == name:
                return str(value).strip()
        return ""

    def _tags(self, properties: dict[str, Any]) -> list[str]:
        tags: list[str] = []
        for key, value in properties.items():
            normalized_key = str(key).strip().lower()
            if normalized_key in TAG_PROPERTIES:
                for tag in self._split_tag_values(value):
                    self._append_tag(tags, tag)
            elif normalized_key in STATUS_PROPERTIES:
                status = self._slug(str(value))
                if status:
                    self._append_tag(tags, f"status/{status}")
        return tags

    def _split_tag_values(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item) for item in value]
        if value is None:
            return []
        return [
            item.strip()
            for item in re.split(r"[,;]", str(value))
            if item.strip()
        ]

    def _append_tag(self, tags: list[str], tag: str) -> None:
        normalized = self._normalize_tag(tag)
        if normalized and normalized not in tags:
            tags.append(normalized)

    def _normalize_tag(self, tag: str) -> str:
        normalized = tag.strip().removeprefix("#").strip()
        if normalized.startswith("[[") and normalized.endswith("]]"):
            normalized = normalized[2:-2].strip()
        return normalized

    def _slug(self, value: str) -> str:
        return re.sub(r"[^a-z0-9_-]+", "-", value.strip().lower()).strip("-")

    def _relative_path(self, path: Path, source_root: Path) -> str:
        try:
            return path.relative_to(source_root).as_posix()
        except ValueError:
            return path.resolve().as_posix()

    def _sync_timestamp(self, since: SyncState) -> float:
        if isinstance(since.last_sync_at, datetime):
            return since.last_sync_at.timestamp()
        return datetime.fromisoformat(str(since.last_sync_at)).timestamp()

    def _jsonable(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._jsonable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._jsonable(item) for item in value]
        if isinstance(value, str | int | float | bool) or value is None:
            return value
        return str(value)
