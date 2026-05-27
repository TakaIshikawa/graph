"""Adapter for Microsoft Edge bookmark JSON exports."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class MicrosoftEdgeBookmarksJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "microsoft_edge_bookmarks_json"

    @property
    def entity_types(self) -> list[str]:
        return ["bookmark"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "bookmark" not in entity_types:
            return result
        sync_at = since.last_sync_at if since else None
        for path in self._paths():
            try:
                parsed = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for record in self._bookmarks(parsed):
                unit = self._unit(record, path.name)
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)
        result.units = sorted({unit.source_id: unit for unit in result.units}.values(), key=lambda unit: unit.source_id)
        return result

    def _paths(self) -> list[Path]:
        root = Path(self.path).expanduser()
        if not self.path:
            return []
        if root.is_file() and root.suffix.lower() == ".json":
            return [root]
        if root.is_dir():
            return sorted(child for child in root.rglob("*.json") if child.is_file())
        return []

    def _bookmarks(self, parsed: Any) -> list[dict[str, Any]]:
        roots = parsed.get("roots") if isinstance(parsed, dict) else None
        if not isinstance(roots, dict) and isinstance(parsed, dict):
            roots = parsed
        records: list[dict[str, Any]] = []
        for node in (roots or {}).values():
            if isinstance(node, dict):
                self._walk(node, (), records)
        return records

    def _walk(self, node: dict[str, Any], folder_path: tuple[str, ...], records: list[dict[str, Any]]) -> None:
        if node.get("type") == "url" or node.get("url"):
            url = str(node.get("url") or "").strip()
            if url:
                records.append({"title": str(node.get("name") or url).strip(), "url": url, "folder_path": folder_path, "date_added": node.get("date_added"), "guid": str(node.get("guid") or "").strip()})
            return
        name = str(node.get("name") or "").strip()
        next_path = (*folder_path, name) if name else folder_path
        children = node.get("children")
        if isinstance(children, list):
            for child in children:
                if isinstance(child, dict):
                    self._walk(child, next_path, records)

    def _unit(self, record: dict[str, Any], source_file: str) -> KnowledgeUnit:
        created_at = self._edge_datetime(record.get("date_added"))
        now = datetime.now(timezone.utc)
        folder_path = "/".join(record["folder_path"])
        metadata = clean_metadata({"url": record["url"], "folder_path": folder_path, "date_added": created_at.isoformat() if created_at else "", "guid": record["guid"], "source_file": source_file})
        return KnowledgeUnit(
            source_project="microsoft_edge_bookmarks_json",
            source_id=digest_source_id("microsoft_edge_bookmarks_json", record["guid"] or record["url"]),
            source_entity_type="bookmark",
            title=record["title"],
            content="\n".join(part for part in [record["title"], record["url"], f"Folder: {folder_path}" if folder_path else ""] if part),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            created_at=created_at or now,
            updated_at=created_at or now,
        )

    def _edge_datetime(self, value: Any) -> datetime | None:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            micros = int(text)
        except ValueError:
            return None
        return datetime(1601, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=micros)
