"""Adapter for Roam Research JSON exports."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


PAGE_LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
HASHTAG_RE = re.compile(r"(?<![\w/])#([A-Za-z0-9_/-]*[A-Za-z0-9_])")
BLOCK_REF_RE = re.compile(r"\(\(([A-Za-z0-9_-]+)\)\)")


@dataclass(frozen=True)
class _RoamItem:
    source_id: str
    entity_type: str
    title: str
    content: str
    metadata: dict[str, Any]
    tags: list[str] = field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class RoamAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "roam"

    @property
    def entity_types(self) -> list[str]:
        return ["page", "block"]

    def __init__(
        self,
        file_path: str = "",
        root_path: str = "",
        path: str = "",
    ) -> None:
        self.file_path = file_path or path
        self.root_path = root_path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        allowed_types = set(entity_types or self.entity_types)
        if not allowed_types.intersection(self.entity_types):
            return result

        paths = self._json_paths()
        sync_at = self._sync_datetime(since) if since else None
        all_items: list[_RoamItem] = []
        for path in paths:
            pages = self._read_pages(path)
            all_items.extend(self._items_from_pages(path, pages))

        title_index = self._title_index(all_items)
        uid_index = self._uid_index(all_items)
        included_source_ids: set[str] = set()

        for item in all_items:
            if item.entity_type not in allowed_types:
                continue
            sync_candidate = item.updated_at or item.created_at
            if sync_at and sync_candidate and sync_candidate <= sync_at:
                continue
            included_source_ids.add(item.source_id)
            result.units.append(self._unit(item))

        result.units.sort(key=lambda unit: (unit.source_entity_type, unit.source_id))

        emitted_edges: set[tuple[str, str, EdgeRelation]] = set()
        for item in all_items:
            if item.entity_type != "block" or item.source_id not in included_source_ids:
                continue
            for target in self._resolved_reference_targets(item.content, title_index, uid_index):
                if target == item.source_id or target not in included_source_ids:
                    continue
                edge_key = (item.source_id, target, EdgeRelation.REFERENCES)
                if edge_key in emitted_edges:
                    continue
                emitted_edges.add(edge_key)
                result.edges.append(
                    KnowledgeEdge(
                        id=self._edge_id(item.source_id, target),
                        from_unit_id=item.source_id,
                        to_unit_id=target,
                        relation=EdgeRelation.REFERENCES,
                        source=EdgeSource.SOURCE,
                        metadata={
                            "source_project": SourceProject.ROAM.value,
                            "from_entity_type": "block",
                            "relation_type": "roam_reference",
                        },
                    )
                )

        result.edges.sort(key=lambda edge: (edge.from_unit_id, edge.to_unit_id, edge.id))
        return result

    def _json_paths(self) -> list[Path]:
        if self.file_path:
            path = Path(self.file_path).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Roam JSON export path does not exist: {path}")
            if not path.is_file():
                raise ValueError(f"Roam file_path must be a JSON file: {path}")
            return [path]

        if self.root_path:
            root = Path(self.root_path).expanduser()
            if not root.exists():
                raise FileNotFoundError(f"Roam JSON export root does not exist: {root}")
            if not root.is_dir():
                raise ValueError(f"Roam root_path must be a directory: {root}")
            paths = sorted(path for path in root.rglob("*.json") if path.is_file())
            if not paths:
                raise FileNotFoundError(f"No Roam JSON export files found under: {root}")
            return paths

        raise ValueError("RoamAdapter requires file_path or root_path")

    def _read_pages(self, path: Path) -> list[dict[str, Any]]:
        try:
            parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed Roam JSON export at {path}: {exc.msg}") from exc
        except (OSError, UnicodeDecodeError) as exc:
            raise ValueError(f"Unable to read Roam JSON export at {path}: {exc}") from exc

        if isinstance(parsed, list):
            pages = parsed
        elif isinstance(parsed, dict):
            value = parsed.get("pages") or parsed.get("data")
            if not isinstance(value, list):
                raise ValueError(f"Roam JSON export must be a list of pages or contain a pages list: {path}")
            pages = value
        else:
            raise ValueError(f"Roam JSON export must be a list or object: {path}")

        invalid = next((page for page in pages if not isinstance(page, dict)), None)
        if invalid is not None:
            raise ValueError(f"Roam JSON export contains a non-object page in: {path}")
        return pages

    def _items_from_pages(self, path: Path, pages: list[dict[str, Any]]) -> list[_RoamItem]:
        items: list[_RoamItem] = []
        for page_index, page in enumerate(pages, start=1):
            title = self._text(page.get("title")) or "Untitled Roam page"
            uid = self._text(page.get("uid"))
            source_id = self._source_id("page", uid, title, f"{path}:{page_index}")
            created_at = self._parse_datetime(self._first(page, "create-time", "created-time", "created", "created_at"))
            updated_at = self._parse_datetime(self._first(page, "edit-time", "edited-time", "edited", "updated_at"))
            children = self._children(page)
            content = "\n".join(self._block_strings(children))
            page_tags = self._tags(f"{title}\n{content}")
            items.append(
                _RoamItem(
                    source_id=source_id,
                    entity_type="page",
                    title=title,
                    content=content or title,
                    metadata={
                        "uid": uid,
                        "title": title,
                        "path": str(path),
                        "page_index": page_index,
                    },
                    tags=page_tags,
                    created_at=created_at,
                    updated_at=updated_at or created_at,
                )
            )
            self._append_blocks(
                items,
                path=path,
                blocks=children,
                page_title=title,
                page_source_id=source_id,
                parent_source_id=source_id,
                position=(page_index,),
            )
        return items

    def _append_blocks(
        self,
        items: list[_RoamItem],
        *,
        path: Path,
        blocks: list[dict[str, Any]],
        page_title: str,
        page_source_id: str,
        parent_source_id: str,
        position: tuple[int, ...],
    ) -> None:
        for index, block in enumerate(blocks, start=1):
            block_position = (*position, index)
            text = self._text(block.get("string"))
            uid = self._text(block.get("uid"))
            if text:
                source_id = self._source_id("block", uid, text, f"{path}:{'.'.join(map(str, block_position))}")
                created_at = self._parse_datetime(self._first(block, "create-time", "created-time", "created", "created_at"))
                updated_at = self._parse_datetime(self._first(block, "edit-time", "edited-time", "edited", "updated_at"))
                items.append(
                    _RoamItem(
                        source_id=source_id,
                        entity_type="block",
                        title=self._block_title(text),
                        content=text,
                        metadata={
                            "uid": uid,
                            "path": str(path),
                            "page_title": page_title,
                            "page_source_id": page_source_id,
                            "parent_source_id": parent_source_id,
                            "position": list(block_position),
                        },
                        tags=self._tags(text),
                        created_at=created_at,
                        updated_at=updated_at or created_at,
                    )
                )
                next_parent = source_id
            else:
                next_parent = parent_source_id

            self._append_blocks(
                items,
                path=path,
                blocks=self._children(block),
                page_title=page_title,
                page_source_id=page_source_id,
                parent_source_id=next_parent,
                position=block_position,
            )

    def _unit(self, item: _RoamItem) -> KnowledgeUnit:
        now = datetime.now(timezone.utc)
        return KnowledgeUnit(
            source_project=SourceProject.ROAM,
            source_id=item.source_id,
            source_entity_type=item.entity_type,
            title=item.title,
            content=item.content,
            content_type=ContentType.ARTIFACT,
            metadata=item.metadata,
            tags=item.tags,
            created_at=item.created_at or now,
            updated_at=item.updated_at or item.created_at or now,
        )

    def _resolved_reference_targets(
        self,
        text: str,
        title_index: dict[str, str],
        uid_index: dict[str, str],
    ) -> list[str]:
        targets: list[str] = []
        for title in PAGE_LINK_RE.findall(text):
            target = title_index.get(self._normalize_title(title))
            if target and target not in targets:
                targets.append(target)
        for tag in HASHTAG_RE.findall(text):
            target = title_index.get(self._normalize_title(tag))
            if target and target not in targets:
                targets.append(target)
        for uid in BLOCK_REF_RE.findall(text):
            target = uid_index.get(uid)
            if target and target not in targets:
                targets.append(target)
        return targets

    def _title_index(self, items: list[_RoamItem]) -> dict[str, str]:
        index: dict[str, str] = {}
        for item in items:
            if item.entity_type == "page":
                index.setdefault(self._normalize_title(item.title), item.source_id)
        return index

    def _uid_index(self, items: list[_RoamItem]) -> dict[str, str]:
        index: dict[str, str] = {}
        for item in items:
            uid = self._text(item.metadata.get("uid"))
            if uid:
                index.setdefault(uid, item.source_id)
        return index

    def _source_id(self, entity_type: str, uid: str, text: str, fallback: str) -> str:
        if uid:
            return f"roam:{entity_type}:{uid}"
        digest = hashlib.sha1(f"{entity_type}|{text}|{fallback}".encode("utf-8")).hexdigest()[:16]
        return f"roam:{entity_type}:{digest}"

    def _edge_id(self, from_source_id: str, to_source_id: str) -> str:
        raw = "|".join([SourceProject.ROAM.value, EdgeRelation.REFERENCES.value, from_source_id, to_source_id])
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
        return f"roam-references-{digest}"

    def _children(self, value: dict[str, Any]) -> list[dict[str, Any]]:
        children = value.get("children") or []
        if not isinstance(children, list):
            return []
        return [child for child in children if isinstance(child, dict)]

    def _block_strings(self, blocks: list[dict[str, Any]]) -> list[str]:
        strings: list[str] = []
        for block in blocks:
            text = self._text(block.get("string"))
            if text:
                strings.append(text)
            strings.extend(self._block_strings(self._children(block)))
        return strings

    def _tags(self, text: str) -> list[str]:
        tags: list[str] = []
        for value in [*PAGE_LINK_RE.findall(text), *HASHTAG_RE.findall(text)]:
            normalized = self._normalize_tag(value)
            if normalized and normalized not in tags:
                tags.append(normalized)
        return tags

    def _normalize_title(self, value: str) -> str:
        return re.sub(r"\s+", " ", value.strip()).casefold()

    def _normalize_tag(self, value: str) -> str:
        return re.sub(r"\s+", " ", value.strip().removeprefix("#")).strip().lower()

    def _block_title(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        return text if len(text) <= 80 else f"{text[:77]}..."

    def _first(self, item: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in item and item[key] not in (None, ""):
                return item[key]
        return None

    def _text(self, value: Any) -> str:
        if value is None or isinstance(value, (dict, list)):
            return ""
        return str(value).strip()

    def _parse_datetime(self, value: Any) -> datetime | None:
        if value in (None, ""):
            return None
        if isinstance(value, (int, float)):
            try:
                timestamp = float(value)
                if timestamp > 10_000_000_000:
                    timestamp /= 1000
                return datetime.fromtimestamp(timestamp, tz=timezone.utc)
            except (OSError, OverflowError, ValueError):
                return None
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
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
