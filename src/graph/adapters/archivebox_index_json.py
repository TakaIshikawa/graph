"""Adapter for ArchiveBox index JSON exports."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState

_ARTIFACT_EXTRACTORS = frozenset(
    {"title", "readability", "media", "screenshot", "pdf", "wget", "dom"}
)


class ArchiveBoxIndexJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "archivebox_index_json"

    @property
    def entity_types(self) -> list[str]:
        return ["archive", "artifact", "url_reference"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        allowed_types = set(entity_types) if entity_types else {"archive"}

        sync_at = self._ensure_utc(since.last_sync_at) if since else None
        units: list[KnowledgeUnit] = []
        edges: list[KnowledgeEdge] = []
        for path in self._iter_paths():
            try:
                entries = self._read_entries(path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for entry in entries:
                archive_unit = self._unit_from_entry(entry, path.name)
                if archive_unit is None:
                    continue
                if sync_at and archive_unit.created_at <= sync_at:
                    continue
                archive_emitted = allowed_types is None or "archive" in allowed_types
                artifact_emitted = allowed_types is None or "artifact" in allowed_types
                reference_emitted = allowed_types is None or "url_reference" in allowed_types

                if archive_emitted:
                    units.append(archive_unit)

                artifacts = self._artifact_units(entry, archive_unit, path.name)
                if artifact_emitted:
                    units.extend(artifacts)

                if archive_emitted and artifact_emitted:
                    edges.extend(self._artifact_edges(archive_unit, artifacts))

                references = self._url_reference_units(entry, archive_unit, path.name)
                if reference_emitted:
                    units.extend(references)
                if archive_emitted and reference_emitted:
                    edges.extend(self._url_reference_edges(archive_unit, references))

        result.units.extend(sorted(units, key=lambda unit: (unit.created_at, unit.source_id)))
        result.edges.extend(sorted(edges, key=lambda edge: edge.id))
        return result

    def _iter_paths(self) -> list[Path]:
        if not self.path:
            return []
        root = Path(self.path).expanduser()
        if root.is_file() and root.suffix.lower() == ".json":
            return [root]
        if not root.is_dir():
            return []
        return sorted(child for child in root.rglob("*.json") if child.is_file())

    def _read_entries(self, path: Path) -> list[dict[str, Any]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        return self._entries(parsed)

    def _entries(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if not isinstance(value, dict):
            return []
        for key in ("entries", "results", "items", "snapshots", "archive"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
            if isinstance(nested, dict):
                if self._looks_like_entry(nested):
                    return [nested]
                return [item for item in nested.values() if isinstance(item, dict)]
        if self._looks_like_entry(value):
            return [value]
        return [item for item in value.values() if isinstance(item, dict) and self._looks_like_entry(item)]

    def _looks_like_entry(self, entry: dict[str, Any]) -> bool:
        return any(key in entry for key in ("url", "base_url", "timestamp", "title"))

    def _unit_from_entry(self, entry: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        url = self._first(entry, "url", "base_url", "original_url")
        if not url:
            return None
        title = self._first(entry, "title") or url
        timestamp_text = self._first(entry, "timestamp", "bookmarked_at", "created_at", "added")
        timestamp = self._parse_datetime(timestamp_text) or datetime.now(timezone.utc)
        tags = self._tags(entry.get("tags"))
        extractor_outputs = self._extractor_outputs(entry)
        archive_paths = self._archive_paths(entry)
        status = self._first(entry, "status", "downloaded", "is_archived")

        metadata = {
            "url": url,
            "title": title,
            "timestamp": timestamp.isoformat(),
            "tags": tags,
            "status": status,
            "extractor_outputs": extractor_outputs,
            "archive_paths": archive_paths,
            "source_file": source_file,
            "entry": entry,
        }
        unit_tags = ["archivebox", *tags]
        return KnowledgeUnit(
            source_project=SourceProject.ARCHIVEBOX_INDEX_JSON,
            source_id=self._source_id(entry, url, timestamp),
            source_entity_type="archive",
            title=title,
            content=self._content(title, url, timestamp, tags, status, archive_paths),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=unit_tags,
            created_at=timestamp,
            updated_at=timestamp,
        )

    def _source_id(self, entry: dict[str, Any], url: str, timestamp: datetime) -> str:
        explicit = self._first(entry, "id", "uuid", "timestamp")
        raw = explicit or f"{url}|{timestamp.isoformat()}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"archivebox_index_json:{digest}"

    def _artifact_units(
        self,
        entry: dict[str, Any],
        archive_unit: KnowledgeUnit,
        source_file: str,
    ) -> list[KnowledgeUnit]:
        artifacts: list[KnowledgeUnit] = []
        url = str(archive_unit.metadata.get("url") or "")
        for extractor, output in self._artifact_outputs(entry):
            output_value = self._artifact_output_value(output)
            source_id = self._artifact_source_id(archive_unit.source_id, extractor, output_value)
            metadata = {
                "extractor": extractor,
                "output": output,
                "output_path": output_value,
                "parent_archive_source_id": archive_unit.source_id,
                "source_file": source_file,
                "original_url": url,
            }
            title = f"{archive_unit.title} [{extractor}]"
            artifacts.append(
                KnowledgeUnit(
                    source_project=SourceProject.ARCHIVEBOX_INDEX_JSON,
                    source_id=source_id,
                    source_entity_type="artifact",
                    title=title,
                    content=self._artifact_content(title, extractor, output_value, url),
                    content_type=ContentType.ARTIFACT,
                    metadata=metadata,
                    tags=sorted({"archivebox", "artifact", extractor}),
                    created_at=archive_unit.created_at,
                    updated_at=archive_unit.updated_at,
                )
            )
        return sorted(artifacts, key=lambda unit: unit.source_id)

    def _artifact_outputs(self, entry: dict[str, Any]) -> list[tuple[str, Any]]:
        outputs: list[tuple[str, Any]] = []
        for key, value in entry.items():
            extractor = str(key).lower()
            if extractor in _ARTIFACT_EXTRACTORS and self._has_artifact_value(value):
                outputs.append((extractor, value))
        for container_key in ("history", "extractors", "outputs"):
            container = entry.get(container_key)
            if not isinstance(container, dict):
                continue
            for key, value in container.items():
                extractor = str(key).lower()
                if extractor in _ARTIFACT_EXTRACTORS and self._has_artifact_value(value):
                    outputs.append((extractor, value))
        deduped: dict[tuple[str, str], tuple[str, Any]] = {}
        for extractor, value in outputs:
            deduped[(extractor, self._artifact_output_value(value))] = (extractor, value)
        return [deduped[key] for key in sorted(deduped)]

    def _has_artifact_value(self, value: Any) -> bool:
        if value is None or value is False:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, list | tuple | dict):
            return bool(value)
        return True

    def _artifact_output_value(self, value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            for key in ("path", "output", "url", "href", "cmd", "filename"):
                found = self._first(value, key)
                if found:
                    return found
            if "result" in value:
                return self._artifact_output_value(value["result"])
            return json.dumps(value, sort_keys=True, default=str)
        if isinstance(value, list | tuple):
            parts = [self._artifact_output_value(item) for item in value]
            return ", ".join(part for part in parts if part)
        return str(value)

    def _artifact_source_id(self, archive_source_id: str, extractor: str, output_value: str) -> str:
        digest = hashlib.sha256(
            f"{archive_source_id}|{extractor}|{output_value}".encode("utf-8")
        ).hexdigest()[:24]
        return f"{archive_source_id}:artifact:{extractor}:{digest}"

    def _artifact_edges(
        self,
        archive_unit: KnowledgeUnit,
        artifacts: list[KnowledgeUnit],
    ) -> list[KnowledgeEdge]:
        return [
            KnowledgeEdge(
                id=self._edge_id(archive_unit.source_id, artifact.source_id),
                from_unit_id=archive_unit.source_id,
                to_unit_id=artifact.source_id,
                relation=EdgeRelation.CONTAINS,
                source=EdgeSource.SOURCE,
                metadata={
                    "source_project": SourceProject.ARCHIVEBOX_INDEX_JSON.value,
                    "relation_type": "archive_contains_artifact",
                    "extractor": artifact.metadata["extractor"],
                    "original_url": archive_unit.metadata.get("url"),
                },
            )
            for artifact in artifacts
        ]

    def _url_reference_units(
        self,
        entry: dict[str, Any],
        archive_unit: KnowledgeUnit,
        source_file: str,
    ) -> list[KnowledgeUnit]:
        references: list[KnowledgeUnit] = []
        for link in self._outbound_links(entry):
            url = link["url"]
            title = link.get("title") or link.get("text") or url
            metadata = {
                "url": url,
                "title": link.get("title"),
                "text": link.get("text"),
                "parent_archive_source_id": archive_unit.source_id,
                "source_file": source_file,
                "original_url": archive_unit.metadata.get("url"),
            }
            references.append(
                KnowledgeUnit(
                    source_project=SourceProject.ARCHIVEBOX_INDEX_JSON,
                    source_id=self._url_reference_source_id(archive_unit.source_id, url),
                    source_entity_type="url_reference",
                    title=title,
                    content=f"{title}\nURL: {url}",
                    content_type=ContentType.METADATA,
                    metadata={key: value for key, value in metadata.items() if value not in ("", None)},
                    tags=["archivebox", "url_reference"],
                    created_at=archive_unit.created_at,
                    updated_at=archive_unit.updated_at,
                )
            )
        return references

    def _outbound_links(self, entry: dict[str, Any]) -> list[dict[str, str]]:
        raw_values: list[Any] = []
        for key in ("outlinks", "links", "extracted_links"):
            raw_values.append(entry.get(key))
        metadata = entry.get("metadata")
        if isinstance(metadata, dict):
            raw_values.append(metadata.get("outlinks"))

        by_url: dict[str, dict[str, str]] = {}
        for value in raw_values:
            for link in self._coerce_links(value):
                url = link.get("url", "").strip()
                if not url:
                    continue
                by_url.setdefault(url, link)
        return [by_url[url] for url in sorted(by_url)]

    def _coerce_links(self, value: Any) -> list[dict[str, str]]:
        if value is None:
            return []
        if isinstance(value, str):
            return [{"url": item.strip()} for item in value.replace("\n", ",").split(",") if item.strip()]
        if isinstance(value, list | tuple | set):
            links: list[dict[str, str]] = []
            for item in value:
                links.extend(self._coerce_links(item))
            return links
        if isinstance(value, dict):
            url = self._first(value, "url", "href", "link")
            if not url and len(value) == 1:
                only_key, only_value = next(iter(value.items()))
                if str(only_key).startswith("http"):
                    url = str(only_key)
                    value = only_value if isinstance(only_value, dict) else {"text": only_value}
            if not url:
                return []
            return [
                {
                    "url": url,
                    "title": self._first(value, "title", "name") if isinstance(value, dict) else "",
                    "text": self._first(value, "text", "label", "anchor") if isinstance(value, dict) else "",
                }
            ]
        return []

    def _url_reference_source_id(self, archive_source_id: str, url: str) -> str:
        digest = hashlib.sha256(f"{archive_source_id}|url_reference|{url}".encode("utf-8")).hexdigest()[:24]
        return f"{archive_source_id}:url_reference:{digest}"

    def _url_reference_edges(
        self,
        archive_unit: KnowledgeUnit,
        references: list[KnowledgeUnit],
    ) -> list[KnowledgeEdge]:
        return [
            KnowledgeEdge(
                id=self._reference_edge_id(archive_unit.source_id, reference.source_id),
                from_unit_id=archive_unit.source_id,
                to_unit_id=reference.source_id,
                relation=EdgeRelation.REFERENCES,
                source=EdgeSource.SOURCE,
                metadata={
                    "source_project": SourceProject.ARCHIVEBOX_INDEX_JSON.value,
                    "relation_type": "archive_references_url",
                    "original_url": archive_unit.metadata.get("url"),
                    "url": reference.metadata.get("url"),
                },
            )
            for reference in references
        ]

    def _reference_edge_id(self, archive_source_id: str, reference_source_id: str) -> str:
        digest = hashlib.sha256(
            f"{archive_source_id}|{reference_source_id}|references".encode("utf-8")
        ).hexdigest()[:24]
        return f"archivebox-index-json-references-{digest}"

    def _edge_id(self, archive_source_id: str, artifact_source_id: str) -> str:
        digest = hashlib.sha256(
            f"{archive_source_id}|{artifact_source_id}|contains".encode("utf-8")
        ).hexdigest()[:24]
        return f"archivebox-index-json-contains-{digest}"

    def _artifact_content(self, title: str, extractor: str, output_value: str, url: str) -> str:
        parts = [title, f"Extractor: {extractor}"]
        if output_value:
            parts.append(f"Output: {output_value}")
        if url:
            parts.append(f"Original URL: {url}")
        return "\n".join(parts)

    def _extractor_outputs(self, entry: dict[str, Any]) -> dict[str, Any]:
        outputs: dict[str, Any] = {}
        for key in ("history", "extractors", "outputs"):
            value = entry.get(key)
            if isinstance(value, dict):
                outputs[key] = value
        return outputs

    def _archive_paths(self, entry: dict[str, Any]) -> list[str]:
        paths: list[str] = []
        for key, value in entry.items():
            lowered = str(key).lower()
            if any(token in lowered for token in ("path", "archive", "index")):
                self._append_paths(paths, value)
        return paths

    def _append_paths(self, paths: list[str], value: Any) -> None:
        if isinstance(value, str):
            if value.strip() and value.strip() not in paths:
                paths.append(value.strip())
            return
        if isinstance(value, list):
            for item in value:
                self._append_paths(paths, item)
            return
        if isinstance(value, dict):
            for item in value.values():
                self._append_paths(paths, item)

    def _tags(self, value: Any) -> list[str]:
        raw: list[Any]
        if isinstance(value, list):
            raw = value
        elif isinstance(value, str):
            raw = value.replace(";", ",").split(",")
        else:
            raw = []
        tags: list[str] = []
        for item in raw:
            tag = str(item).strip()
            if tag and tag not in tags:
                tags.append(tag)
        return tags

    def _content(self, title: str, url: str, timestamp: datetime, tags: list[str], status: str, archive_paths: list[str]) -> str:
        parts = [title, f"URL: {url}", f"Timestamp: {timestamp.isoformat()}"]
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")
        if status:
            parts.append(f"Status: {status}")
        if archive_paths:
            parts.append("Archive paths: " + ", ".join(archive_paths))
        return "\n".join(parts)

    def _first(self, item: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = item.get(key)
            if value is not None and not isinstance(value, (dict, list)) and str(value).strip():
                return str(value).strip()
        return ""

    def _parse_datetime(self, value: Any) -> datetime | None:
        if value is None or value == "":
            return None
        text = str(value).strip()
        if text.replace(".", "", 1).isdigit():
            try:
                number = float(text)
                if number > 10_000_000_000:
                    number = number / 1000
                return datetime.fromtimestamp(number, tz=timezone.utc)
            except (OSError, OverflowError, ValueError):
                return None
        for candidate in (text, f"{text}T00:00:00"):
            try:
                return self._ensure_utc(datetime.fromisoformat(candidate.replace("Z", "+00:00")))
            except ValueError:
                pass
        return None

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
