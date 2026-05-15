"""Adapter for Confluence page JSON exports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, iter_paths, parse_datetime
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class ConfluencePagesJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "confluence_pages_json"

    @property
    def entity_types(self) -> list[str]:
        return ["page"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "page" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None

        for path in iter_paths(self.path, {".json"}):
            try:
                records = self._read_records(path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for record in records:
                unit = self._unit(record, path.name)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)

        result.units.sort(key=lambda unit: (unit.updated_at, unit.source_id))
        return result

    def _read_records(self, path: Path) -> list[dict[str, Any]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            for key in ("results", "pages", "data"):
                if isinstance(parsed.get(key), list):
                    return [item for item in parsed[key] if isinstance(item, dict)]
            return [parsed]
        return []

    def _unit(self, record: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        page_id = self._text(record.get("id"))
        title = self._text(record.get("title"))
        body = self._body(record)
        url = self._url(record)
        if not any([page_id, title, body, url]):
            return None

        created = parse_datetime(record.get("createdAt") or record.get("created_at") or self._nested(record, "history", "createdDate"))
        updated = parse_datetime(record.get("updatedAt") or record.get("updated_at") or self._nested(record, "version", "when")) or created
        space = record.get("space") if isinstance(record.get("space"), dict) else {}
        creator = self._person(record.get("creator") or self._nested(record, "history", "createdBy"))
        labels = self._labels(record)
        version = record.get("version") if isinstance(record.get("version"), dict) else {}
        metadata = clean_metadata(
            {
                "page_id": page_id,
                "title": title,
                "body": body,
                "url": url,
                "source_url": url,
                "space_key": self._text(space.get("key")),
                "space_name": self._text(space.get("name")),
                "creator": creator,
                "version_number": version.get("number"),
                "version_message": self._text(version.get("message")),
                "labels": labels,
                "created_at": created.isoformat() if created else self._text(record.get("createdAt")),
                "updated_at": updated.isoformat() if updated else self._text(record.get("updatedAt")),
                "source_file": source_file,
            }
        )
        now = datetime.now(timezone.utc)
        return KnowledgeUnit(
            source_project=SourceProject.CONFLUENCE_PAGES_JSON,
            source_id=f"confluence_pages_json:{page_id}" if page_id else digest_source_id("confluence_pages_json", title, url),
            source_entity_type="page",
            title=title or page_id or "Confluence page",
            content=self._content(title, body, metadata),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["confluence", "page", metadata.get("space_key"), *labels] if tag)),
            created_at=created or updated or now,
            updated_at=updated or created or now,
        )

    def _body(self, record: dict[str, Any]) -> str:
        body = record.get("body") if isinstance(record.get("body"), dict) else {}
        for representation in ("storage", "view"):
            value = body.get(representation)
            if isinstance(value, dict) and self._text(value.get("value")):
                return self._text(value.get("value"))
        return self._text(record.get("body") or record.get("content"))

    def _url(self, record: dict[str, Any]) -> str:
        direct = self._text(record.get("url") or record.get("webui") or record.get("tinyui"))
        if direct:
            return direct
        links = record.get("_links") if isinstance(record.get("_links"), dict) else {}
        base = self._text(links.get("base") or links.get("context"))
        webui = self._text(links.get("webui") or links.get("tinyui") or links.get("self"))
        if base and webui.startswith("/"):
            return f"{base}{webui}"
        return webui

    def _labels(self, record: dict[str, Any]) -> list[str]:
        labels = record.get("labels")
        if isinstance(labels, dict):
            labels = labels.get("results")
        if not isinstance(labels, list):
            return []
        values: list[str] = []
        for label in labels:
            value = self._text(label.get("name") if isinstance(label, dict) else label)
            if value and value not in values:
                values.append(value)
        return values

    def _person(self, value: Any) -> str:
        if isinstance(value, dict):
            return self._text(value.get("displayName") or value.get("display_name") or value.get("username") or value.get("accountId"))
        return self._text(value)

    def _nested(self, record: dict[str, Any], *keys: str) -> Any:
        value: Any = record
        for key in keys:
            if not isinstance(value, dict):
                return None
            value = value.get(key)
        return value

    def _content(self, title: str, body: str, metadata: dict[str, Any]) -> str:
        parts = [title, body]
        if metadata.get("url"):
            parts.append(f"URL: {metadata['url']}")
        if metadata.get("space_key"):
            parts.append(f"Space: {metadata['space_key']}")
        return "\n".join(part for part in parts if part)

    def _text(self, value: Any) -> str:
        return "" if value is None else str(value).strip()
