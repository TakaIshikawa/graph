"""Adapter for Notion Markdown export folders."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


FRONT_MATTER_DELIMITER = "---"
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
PROPERTY_RE = re.compile(r"^([A-Za-z][A-Za-z0-9 _/-]{0,80}):\s*(.*)$")
TAG_KEYS = {"tag", "tags", "select", "multi-select", "multi select"}
CREATED_KEYS = {"created", "created time", "created_time", "created at"}
UPDATED_KEYS = {"updated", "last edited", "last edited time", "updated time", "last_edited_time"}
URL_KEYS = {"url", "source", "source url", "source_url", "link"}
ID_KEYS = {"id", "page id", "page_id", "notion id", "notion_id"}


class NotionMarkdownExportAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "notion_markdown_export"

    @property
    def entity_types(self) -> list[str]:
        return ["page"]

    def __init__(self, path: str = "", *, source_id_root: str | None = None) -> None:
        self.path = path
        self.source_id_root = source_id_root

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if "page" not in set(entity_types or self.entity_types):
            return result

        root = Path(self.path).expanduser() if self.path else None
        if root is None or not root.exists():
            return result
        source_root = Path(self.source_id_root).expanduser() if self.source_id_root else (root.parent if root.is_file() else root)
        sync_at = self._ensure_utc(since.last_sync_at) if since else None

        for path in self._markdown_paths(root):
            try:
                stat = path.stat()
                unit = self._unit(path, source_root, stat)
            except (OSError, UnicodeDecodeError, yaml.YAMLError):
                continue
            if sync_at and unit.updated_at <= sync_at:
                continue
            result.units.append(unit)

        result.units.sort(key=lambda unit: (str(unit.metadata.get("path") or ""), unit.source_id))
        return result

    def _markdown_paths(self, root: Path) -> list[Path]:
        if root.is_file():
            return [root] if root.suffix.lower() in {".md", ".markdown"} else []
        return sorted(path for suffix in ("*.md", "*.markdown") for path in root.rglob(suffix) if path.is_file())

    def _unit(self, path: Path, source_root: Path, stat: Any) -> KnowledgeUnit:
        text = path.read_text(encoding="utf-8-sig")
        properties, raw_property_block, body = self._extract_properties(text)
        relative_path = self._relative_path(path, source_root)
        title = self._first_heading(body) or self._property_text(properties, "title", "name") or path.stem
        created = self._property_datetime(properties, CREATED_KEYS) or datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc)
        updated = self._property_datetime(properties, UPDATED_KEYS) or datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        property_id = self._property_text(properties, *ID_KEYS)
        source_url = self._property_text(properties, *URL_KEYS)
        backlinks = self._backlinks(properties, body)
        metadata = {
            "path": relative_path,
            "properties": self._jsonable(properties),
            "raw_property_block": raw_property_block,
            "has_property_block": bool(raw_property_block),
            "property_id": property_id,
            "source_url": source_url,
            "backlinks": backlinks,
            "created_time": created.isoformat(),
            "updated_time": updated.isoformat(),
        }
        return KnowledgeUnit(
            source_project=self.name,
            source_id=self._source_id(relative_path, property_id),
            source_entity_type="page",
            title=title,
            content=body,
            content_type=ContentType.ARTIFACT,
            metadata={key: value for key, value in metadata.items() if value not in ("", None, [])},
            tags=self._tags(properties),
            created_at=created,
            updated_at=updated,
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
        if block_start is None or PROPERTY_RE.match(lines[block_start]) is None:
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
        body = "\n".join(lines[:block_start] + lines[block_end:])
        if text.endswith("\n"):
            body += "\n"
        return properties, raw, body

    def _leading_heading_end(self, lines: list[str]) -> int:
        first = self._first_nonblank(lines, 0)
        if first is None or HEADING_RE.match(lines[first]) is None:
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
                return match.group(1).strip().strip("#").strip()
        return ""

    def _property_text(self, properties: dict[str, Any], *names: str) -> str:
        wanted = {self._normalize_key(name) for name in names}
        for key, value in properties.items():
            if self._normalize_key(str(key)) in wanted and value is not None:
                if isinstance(value, list):
                    return ", ".join(str(item) for item in value)
                return str(value).strip()
        return ""

    def _property_datetime(self, properties: dict[str, Any], names: set[str]) -> datetime | None:
        value = self._property_text(properties, *names)
        return self._parse_datetime(value)

    def _tags(self, properties: dict[str, Any]) -> list[str]:
        tags: list[str] = []
        for key, value in properties.items():
            if self._normalize_key(str(key)) not in {self._normalize_key(item) for item in TAG_KEYS}:
                continue
            for tag in self._split_tag_values(value):
                normalized = tag.strip().removeprefix("#").strip()
                if normalized.startswith("[[") and normalized.endswith("]]"):
                    normalized = normalized[2:-2].strip()
                if normalized and normalized not in tags:
                    tags.append(normalized)
        return tags

    def _split_tag_values(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item) for item in value]
        if value is None:
            return []
        return [item.strip() for item in re.split(r"[,;]", str(value)) if item.strip()]

    def _backlinks(self, properties: dict[str, Any], body: str) -> list[str]:
        values: list[str] = []
        explicit = self._property_text(properties, "backlinks", "backlink", "links")
        values.extend(self._split_tag_values(explicit))
        values.extend(match.group(1).strip() for match in re.finditer(r"\[\[([^\]]+)\]\]", body))
        seen: list[str] = []
        for value in values:
            if value and value not in seen:
                seen.append(value)
        return seen

    def _source_id(self, relative_path: str, property_id: str) -> str:
        if property_id:
            digest = hashlib.sha256(property_id.encode("utf-8")).hexdigest()[:24]
            return f"notion_markdown_export:{digest}"
        digest = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:24]
        return f"notion_markdown_export:{digest}"

    def _relative_path(self, path: Path, source_root: Path) -> str:
        try:
            return path.relative_to(source_root).as_posix()
        except ValueError:
            return path.resolve().as_posix()

    def _normalize_key(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.casefold())

    def _parse_datetime(self, value: str) -> datetime | None:
        if not value:
            return None
        for candidate in (value.strip(), f"{value.strip()}T00:00:00"):
            try:
                return self._ensure_utc(datetime.fromisoformat(candidate.replace("Z", "+00:00")))
            except ValueError:
                continue
        return None

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _jsonable(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._jsonable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._jsonable(item) for item in value]
        if isinstance(value, str | int | float | bool) or value is None:
            return value
        return str(value)
