"""Adapter for local YAML documents."""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


MAPPED_KEYS = {"title", "name", "tags"}
YAML_SUFFIXES = {".yaml", ".yml"}


class YamlAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "yaml"

    @property
    def entity_types(self) -> list[str]:
        return ["yaml_document"]

    def __init__(self, root_path: str = "") -> None:
        self.root_path = root_path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "yaml_document" not in entity_types:
            return result

        root = Path(self.root_path).expanduser()
        if not root.exists() or not root.is_dir():
            return result

        sync_at = self._sync_timestamp(since) if since else None
        malformed_files = 0
        for path in sorted(
            item
            for item in root.rglob("*")
            if item.is_file() and item.suffix.lower() in YAML_SUFFIXES
        ):
            stat = path.stat()
            if sync_at is not None and stat.st_mtime <= sync_at:
                continue

            unit = self._read_yaml(root, path, stat.st_size, stat.st_ctime)
            if unit is None:
                malformed_files += 1
                continue
            result.units.append(unit)

        if malformed_files:
            warnings.warn(
                f"Skipped {malformed_files} malformed YAML file(s).",
                stacklevel=2,
            )

        return result

    def _read_yaml(
        self,
        root: Path,
        path: Path,
        file_size: int,
        created_timestamp: float,
    ) -> KnowledgeUnit | None:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError):
            return None

        source_id = path.relative_to(root).as_posix()
        title = self._title(data, path)
        content = self._content(data, title)
        top_level_keys = [str(key) for key in data] if isinstance(data, dict) else []

        return KnowledgeUnit(
            source_project=SourceProject.YAML,
            source_id=source_id,
            source_entity_type="yaml_document",
            title=title,
            content=content,
            content_type=ContentType.INSIGHT,
            metadata={
                "path": source_id,
                "file_size": file_size,
                "top_level_keys": top_level_keys,
            },
            tags=self._tags(data),
            created_at=datetime.fromtimestamp(created_timestamp, tz=timezone.utc),
        )

    def _title(self, data: Any, path: Path) -> str:
        if isinstance(data, dict):
            for key in ("title", "name"):
                value = data.get(key)
                if value is not None and not isinstance(value, (dict, list)):
                    title = str(value).strip()
                    if title:
                        return title
        return path.stem

    def _content(self, data: Any, fallback: str) -> str:
        if isinstance(data, dict):
            remaining = {
                key: value
                for key, value in data.items()
                if str(key) not in MAPPED_KEYS
            }
            if not remaining:
                return fallback
            return self._dump_yaml(remaining)
        if data is None:
            return fallback
        return self._dump_yaml(data)

    def _dump_yaml(self, value: Any) -> str:
        try:
            return yaml.safe_dump(
                value,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            ).strip()
        except yaml.YAMLError:
            return str(value).strip()

    def _tags(self, data: Any) -> list[str]:
        if not isinstance(data, dict) or not isinstance(data.get("tags"), list):
            return []

        tags: list[str] = []
        for tag in data["tags"]:
            normalized = str(tag).strip().removeprefix("#").strip()
            if normalized and normalized not in tags:
                tags.append(normalized)
        return tags

    def _sync_timestamp(self, since: SyncState) -> float:
        if isinstance(since.last_sync_at, datetime):
            return since.last_sync_at.timestamp()
        return datetime.fromisoformat(str(since.last_sync_at)).timestamp()
