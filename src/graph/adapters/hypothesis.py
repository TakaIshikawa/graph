"""Adapter for local Hypothesis annotation exports."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class HypothesisAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "hypothesis"

    @property
    def entity_types(self) -> list[str]:
        return ["annotation"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "annotation" not in entity_types:
            return result

        path = Path(self.path).expanduser() if self.path else None
        if path is None or not path.exists() or not path.is_file():
            return result

        try:
            annotations = self._read_annotations(path)
        except (OSError, UnicodeDecodeError):
            return result

        sync_at = self._sync_datetime(since) if since else None
        for annotation in annotations:
            unit = self._unit_from_annotation(annotation)
            if unit is None:
                continue
            comparable_at = unit.updated_at or unit.created_at
            if sync_at and comparable_at and comparable_at <= sync_at:
                continue
            result.units.append(unit)

        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _read_annotations(self, path: Path) -> list[dict[str, Any]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if not isinstance(parsed, dict):
            return []

        for key in ("rows", "annotations", "items"):
            value = parsed.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, dict):
                return [item for item in value.values() if isinstance(item, dict)]
        if self._looks_like_annotation(parsed):
            return [parsed]
        return []

    def _unit_from_annotation(self, annotation: dict[str, Any]) -> KnowledgeUnit | None:
        uri = self._field(annotation, "uri")
        text = self._field(annotation, "text")
        quote = self._quote(annotation)
        created_text = self._field(annotation, "created")
        updated_text = self._field(annotation, "updated")
        created_at = self._parse_datetime(created_text)
        updated_at = self._parse_datetime(updated_text)

        if not any((uri, text, quote, self._field(annotation, "id"))):
            return None

        tags = self._tags(annotation.get("tags"))
        title = self._title(annotation, uri)
        metadata = {
            "id": self._field(annotation, "id"),
            "uri": uri,
            "group": self._field(annotation, "group"),
            "user": self._field(annotation, "user"),
            "created": created_text,
            "updated": updated_text,
            "tags": tags,
            "quote": quote,
            "text": text,
            "document": self._jsonable(annotation.get("document")),
            "target": self._jsonable(annotation.get("target")),
        }

        return KnowledgeUnit(
            source_project=SourceProject.HYPOTHESIS,
            source_id=self._source_id(annotation, uri, quote, text),
            source_entity_type="annotation",
            title=title,
            content=self._content(text, quote, uri, tags),
            content_type=ContentType.INSIGHT,
            metadata=metadata,
            tags=tags,
            created_at=created_at or updated_at or datetime.now(timezone.utc),
            updated_at=updated_at or created_at or datetime.now(timezone.utc),
        )

    def _looks_like_annotation(self, value: dict[str, Any]) -> bool:
        return any(key in value for key in ("id", "uri", "text", "target", "created", "updated"))

    def _title(self, annotation: dict[str, Any], uri: str) -> str:
        document = annotation.get("document")
        if isinstance(document, dict):
            title = document.get("title")
            if isinstance(title, list):
                for item in title:
                    text = str(item).strip()
                    if text:
                        return text
            elif title is not None:
                text = str(title).strip()
                if text:
                    return text

        if uri:
            return uri
        return "Hypothesis annotation"

    def _quote(self, annotation: dict[str, Any]) -> str:
        target = annotation.get("target")
        targets = target if isinstance(target, list) else [target]
        for item in targets:
            if not isinstance(item, dict):
                continue
            selector = item.get("selector")
            selectors = selector if isinstance(selector, list) else [selector]
            for candidate in selectors:
                if not isinstance(candidate, dict):
                    continue
                if candidate.get("type") == "TextQuoteSelector":
                    exact = self._scalar(candidate.get("exact"))
                    if exact:
                        return exact
            for candidate in selectors:
                if isinstance(candidate, dict):
                    exact = self._scalar(candidate.get("exact"))
                    if exact:
                        return exact
        return ""

    def _content(self, text: str, quote: str, uri: str, tags: list[str]) -> str:
        parts: list[str] = []
        if text:
            parts.append(text)
        if quote:
            parts.append(f"Quote: {quote}")
        if uri:
            parts.append(f"URL: {uri}")
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")
        return "\n".join(parts)

    def _source_id(self, annotation: dict[str, Any], uri: str, quote: str, text: str) -> str:
        annotation_id = self._field(annotation, "id")
        if annotation_id:
            return f"hypothesis:{annotation_id}"

        created = self._field(annotation, "created")
        digest = hashlib.sha256(
            "\n".join([uri, created, quote, text]).encode("utf-8")
        ).hexdigest()
        return f"hypothesis:{digest[:24]}"

    def _tags(self, value: Any) -> list[str]:
        raw_tags = value if isinstance(value, list) else []
        tags: list[str] = []
        for tag in raw_tags:
            normalized = re.sub(r"\s+", " ", str(tag).strip().removeprefix("#")).strip()
            if normalized and normalized not in tags:
                tags.append(normalized)
        return tags

    def _field(self, annotation: dict[str, Any], key: str) -> str:
        return self._scalar(annotation.get(key))

    def _scalar(self, value: Any) -> str:
        if value is None or isinstance(value, (dict, list)):
            return ""
        return str(value).strip()

    def _jsonable(self, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, list):
            return [self._jsonable(item) for item in value]
        if isinstance(value, dict):
            return {str(key): self._jsonable(item) for key, item in value.items()}
        return str(value)

    def _parse_datetime(self, value: str) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _sync_datetime(self, since: SyncState) -> datetime:
        value = since.last_sync_at
        if isinstance(value, datetime):
            parsed = value
        else:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
