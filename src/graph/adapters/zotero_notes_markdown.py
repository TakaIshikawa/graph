"""Adapter for Zotero notes exported as Markdown."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class ZoteroNotesMarkdownAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "zotero_notes_markdown"

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
        for path in self._paths():
            try:
                raw = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            unit = self._unit(path, raw)
            if unit and (sync_at is None or unit.updated_at > sync_at):
                result.units.append(unit)
        result.units = sorted({unit.source_id: unit for unit in result.units}.values(), key=lambda unit: unit.source_id)
        return result

    def _paths(self) -> list[Path]:
        if not self.path:
            return []
        root = Path(self.path).expanduser()
        if root.is_file() and root.suffix.lower() in {".md", ".markdown"}:
            return [root]
        if root.is_dir():
            return sorted(child for child in root.rglob("*") if child.is_file() and child.suffix.lower() in {".md", ".markdown"})
        return []

    def _unit(self, path: Path, raw: str) -> KnowledgeUnit | None:
        meta, body = self._frontmatter(raw)
        body = body.strip()
        if not body:
            return None
        item_key = self._first(meta, "item_key", "item key", "zotero_item_key", "key")
        citation_key = self._first(meta, "citation_key", "citation key", "citekey")
        tags = self._list(meta.get("tags"))
        collections = self._list(meta.get("collections"))
        now = datetime.now(timezone.utc)
        title = self._first(meta, "title") or path.stem
        metadata = clean_metadata({"item_key": item_key, "citation_key": citation_key, "tags": tags, "collections": collections, "path": str(path), "frontmatter": meta})
        return KnowledgeUnit(
            source_project="zotero_notes_markdown",
            source_id=digest_source_id("zotero_notes_markdown", str(path), item_key or citation_key),
            source_entity_type="note",
            title=title,
            content=body,
            content_type=ContentType.INSIGHT,
            metadata=metadata,
            tags=tags,
            created_at=now,
            updated_at=now,
        )

    def _frontmatter(self, raw: str) -> tuple[dict[str, Any], str]:
        if not raw.startswith("---\n"):
            return {}, raw
        end = raw.find("\n---", 4)
        if end == -1:
            return {}, raw
        meta: dict[str, Any] = {}
        for line in raw[4:end].splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                meta[key.strip()] = value.strip().strip('"').strip("'")
        return meta, raw[end + 4 :]

    def _first(self, meta: dict[str, Any], *keys: str) -> str:
        compact = {key.casefold().replace("-", "_"): value for key, value in meta.items()}
        for key in keys:
            value = compact.get(key.casefold().replace("-", "_"))
            if value:
                return str(value).strip()
        return ""

    def _list(self, value: Any) -> list[str]:
        text = "" if value is None else str(value).strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        return split_values(text)
