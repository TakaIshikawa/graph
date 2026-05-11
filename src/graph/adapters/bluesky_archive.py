"""Adapter for Bluesky/PDS personal archive JSON records."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


POST_TYPE = "app.bsky.feed.post"
LIKE_TYPE = "app.bsky.feed.like"
REPOST_TYPE = "app.bsky.feed.repost"


class BlueskyArchiveAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "bluesky_archive"

    @property
    def entity_types(self) -> list[str]:
        return ["post", "like", "repost"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        requested = set(entity_types or self.entity_types)
        if not requested.intersection(self.entity_types):
            return result

        root = Path(self.path).expanduser() if self.path else None
        if root is None or not root.exists():
            return result

        sync_at = self._sync_datetime(since) if since else None
        references_by_source_id: dict[str, dict[str, str]] = {}
        post_index: dict[str, str] = {}

        for path in self._iter_paths(root):
            for index, record in enumerate(self._read_records(path)):
                normalized = self._normalize_record(record, path, root, index)
                if normalized is None:
                    continue
                unit = self._unit_from_record(normalized)
                if unit is None or unit.source_entity_type not in requested:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)
                if unit.source_entity_type == "post":
                    self._index_post(post_index, unit)
                else:
                    reference = self._reference(normalized)
                    if reference:
                        references_by_source_id[unit.source_id] = reference

        included_source_ids = {unit.source_id for unit in result.units}
        emitted_edges: set[tuple[str, str, str]] = set()
        for source_id in sorted(references_by_source_id):
            if source_id not in included_source_ids:
                continue
            reference = references_by_source_id[source_id]
            target_id = self._target_source_id(post_index, reference)
            if not target_id or target_id == source_id or target_id not in included_source_ids:
                continue
            edge_key = (source_id, target_id, reference["relation_type"])
            if edge_key in emitted_edges:
                continue
            emitted_edges.add(edge_key)
            result.edges.append(
                KnowledgeEdge(
                    id=self._edge_id(source_id, target_id, reference["relation_type"]),
                    from_unit_id=source_id,
                    to_unit_id=target_id,
                    relation=EdgeRelation.REFERENCES,
                    source=EdgeSource.SOURCE,
                    metadata={
                        "source_project": SourceProject.BLUESKY_ARCHIVE.value,
                        "from_entity_type": reference["from_entity_type"],
                        "to_entity_type": "post",
                        **reference,
                    },
                )
            )

        result.units.sort(key=lambda unit: (unit.created_at, unit.source_id))
        result.edges.sort(key=lambda edge: (edge.from_unit_id, edge.to_unit_id, edge.id))
        return result

    def _iter_paths(self, root: Path) -> list[Path]:
        if root.is_file():
            return [root] if root.suffix.lower() == ".json" else []
        if not root.is_dir():
            return []
        return sorted(child for child in root.rglob("*.json") if child.is_file())

    def _read_records(self, path: Path) -> list[dict[str, Any]]:
        try:
            parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return []
        return self._records(parsed)

    def _records(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [record for item in value for record in self._records(item)]
        if not isinstance(value, dict):
            return []
        for key in ("records", "items", "data", "children"):
            nested = value.get(key)
            if isinstance(nested, (list, dict)):
                records = self._records(nested)
                if records:
                    return records
        nested_value = value.get("value") if isinstance(value.get("value"), dict) else value.get("record")
        if self._record_type(value) or (isinstance(nested_value, dict) and self._record_type(nested_value)):
            return [value]
        return []

    def _normalize_record(self, record: dict[str, Any], path: Path, root: Path, index: int) -> dict[str, Any] | None:
        value = record.get("value") if isinstance(record.get("value"), dict) else record.get("record")
        if not isinstance(value, dict):
            value = record
        record_type = self._record_type(record) or self._record_type(value) or self._type_from_path(path)
        entity_type = self._entity_type(record_type)
        if not entity_type:
            return None

        source_path = self._relative_path(path, root)
        normalized = {
            "record_type": record_type,
            "entity_type": entity_type,
            "record": value,
            "wrapper": record,
            "uri": self._first(record, "uri", "at_uri", "atUri") or self._first(value, "uri"),
            "cid": self._first(record, "cid") or self._first(value, "cid"),
            "collection": self._first(record, "collection") or record_type,
            "rkey": self._first(record, "rkey", "recordKey") or path.stem,
            "did": self._did(record, value),
            "source_path": source_path,
            "index": str(index),
        }
        if not normalized["uri"] and normalized["did"] and normalized["collection"] and normalized["rkey"]:
            normalized["uri"] = f"at://{normalized['did']}/{normalized['collection']}/{normalized['rkey']}"
        return normalized

    def _unit_from_record(self, normalized: dict[str, Any]) -> KnowledgeUnit | None:
        entity_type = normalized["entity_type"]
        record = normalized["record"]
        created_text = self._created_at(record)
        created_at = self._parse_datetime(created_text) or datetime.now(timezone.utc)
        metadata = self._metadata(normalized, created_text)
        source_id = self._source_id(normalized)

        if entity_type == "post":
            text = self._first(record, "text")
            if not text:
                text = normalized["uri"] or normalized["source_path"]
            return KnowledgeUnit(
                source_project=SourceProject.BLUESKY_ARCHIVE,
                source_id=source_id,
                source_entity_type="post",
                title=self._title(text, created_at, "Bluesky post"),
                content=text,
                content_type=ContentType.ARTIFACT,
                metadata=metadata,
                tags=["bluesky", "post"],
                created_at=created_at,
                updated_at=created_at,
            )

        reference = self._reference(normalized)
        if reference is None:
            return None
        label = "like" if entity_type == "like" else "repost"
        target = reference.get("referenced_uri") or reference.get("referenced_cid") or "unknown target"
        return KnowledgeUnit(
            source_project=SourceProject.BLUESKY_ARCHIVE,
            source_id=source_id,
            source_entity_type=entity_type,
            title=f"Bluesky {label}: {target}",
            content=f"Bluesky {label}: {target}",
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=["bluesky", entity_type],
            created_at=created_at,
            updated_at=created_at,
        )

    def _metadata(self, normalized: dict[str, Any], created_text: str) -> dict[str, Any]:
        record = normalized["record"]
        subject = self._subject(record)
        metadata: dict[str, Any] = {
            "record_type": normalized["record_type"],
            "uri": normalized["uri"],
            "cid": normalized["cid"],
            "collection": normalized["collection"],
            "rkey": normalized["rkey"],
            "did": normalized["did"],
            "author_did": normalized["did"],
            "created_at": created_text,
            "source_path": normalized["source_path"],
            "path": normalized["source_path"],
            "text": self._first(record, "text"),
            "subject_uri": self._first(subject, "uri"),
            "subject_cid": self._first(subject, "cid"),
            "record": record,
        }
        return {key: value for key, value in metadata.items() if value not in ("", None)}

    def _reference(self, normalized: dict[str, Any]) -> dict[str, str] | None:
        subject = self._subject(normalized["record"])
        referenced_uri = self._first(subject, "uri")
        referenced_cid = self._first(subject, "cid")
        if not referenced_uri and not referenced_cid:
            return None
        entity_type = normalized["entity_type"]
        return {
            "relation_type": f"bluesky_{entity_type}",
            "from_entity_type": entity_type,
            "referenced_uri": referenced_uri,
            "referenced_cid": referenced_cid,
        }

    def _index_post(self, index: dict[str, str], unit: KnowledgeUnit) -> None:
        for key in ("uri", "cid"):
            value = self._string(unit.metadata.get(key))
            if value:
                index[value] = unit.source_id

    def _target_source_id(self, post_index: dict[str, str], reference: dict[str, str]) -> str:
        for key in ("referenced_uri", "referenced_cid"):
            value = reference.get(key, "")
            if value and value in post_index:
                return post_index[value]
        return ""

    def _source_id(self, normalized: dict[str, Any]) -> str:
        raw = (
            normalized["uri"]
            or normalized["cid"]
            or "|".join(
                [
                    normalized["record_type"],
                    normalized["source_path"],
                    normalized["rkey"],
                    normalized["index"],
                    json.dumps(normalized["record"], sort_keys=True, ensure_ascii=False),
                ]
            )
        )
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"bluesky_archive:{digest}"

    def _edge_id(self, source_id: str, target_id: str, relation_type: str) -> str:
        raw = "|".join([SourceProject.BLUESKY_ARCHIVE.value, relation_type, source_id, target_id])
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
        return f"bluesky-archive-references-{digest}"

    def _record_type(self, value: dict[str, Any]) -> str:
        return self._first(value, "$type", "type", "collection")

    def _entity_type(self, record_type: str) -> str:
        return {
            POST_TYPE: "post",
            LIKE_TYPE: "like",
            REPOST_TYPE: "repost",
        }.get(record_type, "")

    def _type_from_path(self, path: Path) -> str:
        parts = set(path.parts)
        if POST_TYPE in parts:
            return POST_TYPE
        if LIKE_TYPE in parts:
            return LIKE_TYPE
        if REPOST_TYPE in parts:
            return REPOST_TYPE
        return ""

    def _subject(self, record: dict[str, Any]) -> dict[str, Any]:
        subject = record.get("subject")
        return subject if isinstance(subject, dict) else {}

    def _created_at(self, record: dict[str, Any]) -> str:
        return self._first(record, "createdAt", "created_at", "indexedAt", "timestamp")

    def _did(self, record: dict[str, Any], value: dict[str, Any]) -> str:
        did = self._first(record, "did", "author_did", "repo") or self._first(value, "did", "author_did", "repo")
        author = record.get("author") or value.get("author")
        if not did and isinstance(author, dict):
            did = self._first(author, "did")
        uri = self._first(record, "uri", "at_uri", "atUri")
        if not did and uri.startswith("at://"):
            parts = uri.removeprefix("at://").split("/", 1)
            did = parts[0] if parts else ""
        return did

    def _title(self, text: str, created_at: datetime, fallback: str) -> str:
        compact = " ".join(text.split()) or fallback
        if len(compact) > 64:
            compact = f"{compact[:61].rstrip()}..."
        return f"{created_at.date().isoformat()} {compact}"

    def _relative_path(self, path: Path, root: Path) -> str:
        source_root = root.parent if root.is_file() else root
        try:
            return path.relative_to(source_root).as_posix()
        except ValueError:
            return path.as_posix()

    def _parse_datetime(self, value: Any) -> datetime | None:
        text = self._string(value)
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            try:
                parsed = datetime.fromtimestamp(float(text), tz=timezone.utc)
            except (OSError, OverflowError, TypeError, ValueError):
                return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _sync_datetime(self, since: SyncState) -> datetime:
        value = since.last_sync_at
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)

    def _first(self, mapping: dict[str, Any], *keys: str) -> str:
        for key in keys:
            text = self._string(mapping.get(key))
            if text:
                return text
        return ""

    def _string(self, value: Any) -> str:
        if value is None or isinstance(value, (dict, list)):
            return ""
        return str(value).strip()
