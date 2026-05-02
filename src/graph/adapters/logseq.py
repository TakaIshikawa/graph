"""Adapter for Logseq EDN-like graph exports."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


PAGE_LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
HASHTAG_RE = re.compile(r"(?<![\w/])#(?!\{|\w+\\?[\"])([A-Za-z0-9_/-]*[A-Za-z0-9_])")
BLOCK_REF_RE = re.compile(r"\(\(([A-Za-z0-9_-]+)\)\)")


@dataclass(frozen=True)
class _LogseqItem:
    source_id: str
    entity_type: str
    title: str
    content: str
    metadata: dict[str, Any]
    tags: list[str] = field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class LogseqAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "logseq"

    @property
    def entity_types(self) -> list[str]:
        return ["page", "block"]

    def __init__(self, file_path: str = "", path: str = "") -> None:
        self.file_path = file_path or path

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

        path = self._export_path()
        parsed = self._read_export(path)
        pages, loose_blocks = self._split_export(parsed)
        sync_at = self._sync_datetime(since) if since else None
        items, contains = self._items_from_export(path, pages, loose_blocks)
        included_source_ids: set[str] = set()

        for item in items:
            if item.entity_type not in allowed_types:
                continue
            sync_candidate = item.updated_at or item.created_at
            if sync_at and sync_candidate and sync_candidate <= sync_at:
                continue
            included_source_ids.add(item.source_id)
            result.units.append(self._unit(item))

        result.units.sort(key=lambda unit: (unit.source_entity_type, unit.source_id))

        emitted_edges: set[tuple[str, str]] = set()
        for parent_id, child_id in contains:
            if parent_id not in included_source_ids or child_id not in included_source_ids:
                continue
            edge_key = (parent_id, child_id)
            if edge_key in emitted_edges:
                continue
            emitted_edges.add(edge_key)
            result.edges.append(
                KnowledgeEdge(
                    id=self._edge_id(parent_id, child_id),
                    from_unit_id=parent_id,
                    to_unit_id=child_id,
                    relation=EdgeRelation.CONTAINS,
                    source=EdgeSource.SOURCE,
                    metadata={
                        "source_project": SourceProject.LOGSEQ.value,
                        "relation_type": "logseq_contains",
                    },
                )
            )

        result.edges.sort(key=lambda edge: (edge.from_unit_id, edge.to_unit_id, edge.id))
        return result

    def _export_path(self) -> Path:
        if not self.file_path:
            raise ValueError("LogseqAdapter requires file_path")
        path = Path(self.file_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Logseq EDN export path does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"Logseq file_path must be an EDN export file: {path}")
        return path

    def _read_export(self, path: Path) -> Any:
        try:
            return _EdnParser(path.read_text(encoding="utf-8-sig")).parse()
        except OSError as exc:
            raise ValueError(f"Unable to read Logseq EDN export at {path}: {exc}") from exc
        except _EdnParseError as exc:
            raise ValueError(f"Malformed Logseq EDN export at {path}: {exc}") from exc

    def _split_export(self, parsed: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if isinstance(parsed, dict):
            pages = self._map_list(parsed, "pages", "logseq/pages", "block/pages")
            blocks = self._map_list(parsed, "blocks", "logseq/blocks", "block/blocks")
            if pages or blocks:
                return pages, blocks
        values = parsed if isinstance(parsed, list) else [parsed]
        maps = [value for value in values if isinstance(value, dict)]
        pages = [value for value in maps if self._looks_like_page(value)]
        blocks = [value for value in maps if self._looks_like_block(value)]
        return pages, blocks

    def _items_from_export(
        self,
        path: Path,
        pages: list[dict[str, Any]],
        loose_blocks: list[dict[str, Any]],
    ) -> tuple[list[_LogseqItem], list[tuple[str, str]]]:
        items: list[_LogseqItem] = []
        contains: list[tuple[str, str]] = []
        page_index: dict[str, str] = {}
        block_index: dict[str, str] = {}

        for index, page in enumerate(pages, start=1):
            page_name = self._page_name(page) or "Untitled Logseq page"
            uuid = self._uuid(page)
            source_id = self._source_id("page", uuid or page_name, f"{path}:{index}")
            page_index[page_name] = source_id
            if uuid:
                page_index[uuid] = source_id
            children = self._children(page)
            content = "\n".join(self._block_contents(children)) or page_name
            tags = self._item_tags(page, content)
            refs = self._item_refs(page, content)
            items.append(
                _LogseqItem(
                    source_id=source_id,
                    entity_type="page",
                    title=page_name,
                    content=content,
                    metadata={
                        "page_name": page_name,
                        "uuid": uuid,
                        "path": str(path),
                        "page_index": index,
                        "tags": tags,
                        "refs": refs,
                    },
                    tags=tags,
                    created_at=self._parse_datetime(self._first(page, "created-at", "block/created-at", "created_at")),
                    updated_at=self._parse_datetime(self._first(page, "updated-at", "block/updated-at", "updated_at")),
                )
            )
            self._append_nested_blocks(
                items,
                contains,
                block_index,
                path=path,
                blocks=children,
                page_name=page_name,
                page_source_id=source_id,
                parent_source_id=source_id,
                position=(index,),
            )

        deferred_blocks: list[tuple[dict[str, Any], str, str | None, int]] = []
        for index, block in enumerate(loose_blocks, start=1):
            content = self._block_content(block)
            uuid = self._uuid(block)
            source_id = self._source_id("block", uuid or content, f"{path}:loose:{index}")
            if uuid:
                block_index[uuid] = source_id
            parent_ref = self._ref_value(self._first(block, "parent", "block/parent"))
            page_ref = self._ref_value(self._first(block, "page", "block/page"))
            deferred_blocks.append((block, source_id, parent_ref or page_ref, index))

        for block, source_id, parent_ref, index in deferred_blocks:
            page_ref = self._ref_value(self._first(block, "page", "block/page"))
            page_source_id = page_index.get(page_ref or "", "")
            parent_source_id = block_index.get(parent_ref or "") or page_index.get(parent_ref or "") or page_source_id
            item = self._block_item(
                path=path,
                block=block,
                source_id=source_id,
                page_name=self._text(page_ref),
                page_source_id=page_source_id,
                parent_source_id=parent_source_id,
                position=(index,),
            )
            items.append(item)
            if parent_source_id:
                contains.append((parent_source_id, source_id))

        return items, contains

    def _append_nested_blocks(
        self,
        items: list[_LogseqItem],
        contains: list[tuple[str, str]],
        block_index: dict[str, str],
        *,
        path: Path,
        blocks: list[dict[str, Any]],
        page_name: str,
        page_source_id: str,
        parent_source_id: str,
        position: tuple[int, ...],
    ) -> None:
        for index, block in enumerate(blocks, start=1):
            block_position = (*position, index)
            content = self._block_content(block)
            uuid = self._uuid(block)
            if content:
                source_id = self._source_id("block", uuid or content, f"{path}:{'.'.join(map(str, block_position))}")
                if uuid:
                    block_index[uuid] = source_id
                items.append(
                    self._block_item(
                        path=path,
                        block=block,
                        source_id=source_id,
                        page_name=page_name,
                        page_source_id=page_source_id,
                        parent_source_id=parent_source_id,
                        position=block_position,
                    )
                )
                contains.append((parent_source_id, source_id))
                next_parent = source_id
            else:
                next_parent = parent_source_id
            self._append_nested_blocks(
                items,
                contains,
                block_index,
                path=path,
                blocks=self._children(block),
                page_name=page_name,
                page_source_id=page_source_id,
                parent_source_id=next_parent,
                position=block_position,
            )

    def _block_item(
        self,
        *,
        path: Path,
        block: dict[str, Any],
        source_id: str,
        page_name: str,
        page_source_id: str,
        parent_source_id: str,
        position: tuple[int, ...],
    ) -> _LogseqItem:
        content = self._block_content(block)
        uuid = self._uuid(block)
        tags = self._item_tags(block, content)
        refs = self._item_refs(block, content)
        return _LogseqItem(
            source_id=source_id,
            entity_type="block",
            title=self._block_title(content),
            content=content,
            metadata={
                "uuid": uuid,
                "path": str(path),
                "page_name": page_name,
                "page_source_id": page_source_id,
                "parent_source_id": parent_source_id,
                "position": list(position),
                "tags": tags,
                "refs": refs,
            },
            tags=tags,
            created_at=self._parse_datetime(self._first(block, "created-at", "block/created-at", "created_at")),
            updated_at=self._parse_datetime(self._first(block, "updated-at", "block/updated-at", "updated_at")),
        )

    def _unit(self, item: _LogseqItem) -> KnowledgeUnit:
        now = datetime.now(timezone.utc)
        return KnowledgeUnit(
            source_project=SourceProject.LOGSEQ,
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

    def _map_list(self, item: dict[str, Any], *keys: str) -> list[dict[str, Any]]:
        for key in keys:
            value = item.get(key)
            if isinstance(value, list):
                return [entry for entry in value if isinstance(entry, dict)]
        return []

    def _looks_like_page(self, item: dict[str, Any]) -> bool:
        return bool(self._page_name(item)) and not self._block_content(item)

    def _looks_like_block(self, item: dict[str, Any]) -> bool:
        return bool(self._block_content(item))

    def _page_name(self, item: dict[str, Any]) -> str:
        return self._text(self._first(item, "name", "title", "original-name", "block/name", "block/original-name"))

    def _block_content(self, item: dict[str, Any]) -> str:
        return self._text(self._first(item, "content", "string", "body", "block/content"))

    def _uuid(self, item: dict[str, Any]) -> str:
        return self._text(self._first(item, "uuid", "id", "block/uuid"))

    def _children(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        for key in ("blocks", "children", "block/children"):
            value = item.get(key)
            if isinstance(value, list):
                return [child for child in value if isinstance(child, dict)]
        return []

    def _block_contents(self, blocks: list[dict[str, Any]]) -> list[str]:
        contents: list[str] = []
        for block in blocks:
            content = self._block_content(block)
            if content:
                contents.append(content)
            contents.extend(self._block_contents(self._children(block)))
        return contents

    def _item_tags(self, item: dict[str, Any], content: str) -> list[str]:
        tags = self._values_from_refs(self._first(item, "tags", "block/tags"))
        tags.extend(self._normalize_tag(value) for value in [*PAGE_LINK_RE.findall(content), *HASHTAG_RE.findall(content)])
        return self._dedupe(value for value in tags if value)

    def _item_refs(self, item: dict[str, Any], content: str) -> list[str]:
        refs = self._values_from_refs(self._first(item, "refs", "block/refs"))
        refs.extend(self._normalize_ref(value) for value in PAGE_LINK_RE.findall(content))
        refs.extend(BLOCK_REF_RE.findall(content))
        return self._dedupe(value for value in refs if value)

    def _values_from_refs(self, value: Any) -> list[str]:
        if value is None:
            return []
        values = value if isinstance(value, list) else [value]
        refs: list[str] = []
        for entry in values:
            if isinstance(entry, dict):
                refs.append(self._text(self._first(entry, "name", "title", "original-name", "block/name", "block/uuid", "uuid", "id")))
            else:
                refs.append(self._text(entry))
        return [self._normalize_ref(ref) for ref in refs if ref]

    def _ref_value(self, value: Any) -> str:
        if isinstance(value, dict):
            return self._text(self._first(value, "uuid", "id", "name", "title", "block/uuid", "block/name"))
        return self._text(value)

    def _source_id(self, entity_type: str, key: str, fallback: str) -> str:
        if key:
            return f"logseq:{entity_type}:{key}"
        digest = hashlib.sha1(f"{entity_type}|{fallback}".encode("utf-8")).hexdigest()[:16]
        return f"logseq:{entity_type}:{digest}"

    def _edge_id(self, from_source_id: str, to_source_id: str) -> str:
        raw = "|".join([SourceProject.LOGSEQ.value, EdgeRelation.CONTAINS.value, from_source_id, to_source_id])
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
        return f"logseq-contains-{digest}"

    def _first(self, item: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in item and item[key] not in (None, ""):
                return item[key]
        return None

    def _text(self, value: Any) -> str:
        if value is None or isinstance(value, (dict, list)):
            return ""
        return str(value).strip()

    def _normalize_tag(self, value: str) -> str:
        return re.sub(r"\s+", " ", value.strip().removeprefix("#")).strip().lower()

    def _normalize_ref(self, value: str) -> str:
        return re.sub(r"\s+", " ", value.strip()).strip()

    def _dedupe(self, values: Any) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            key = str(value).casefold()
            if value and key not in seen:
                result.append(value)
                seen.add(key)
        return result

    def _block_title(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        return text if len(text) <= 80 else f"{text[:77]}..."

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


class _EdnParseError(ValueError):
    pass


class _EdnParser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.index = 0

    def parse(self) -> Any:
        value = self._value()
        self._skip_ws()
        if self.index != len(self.text):
            raise _EdnParseError(f"unexpected trailing input at byte {self.index}")
        return value

    def _value(self) -> Any:
        self._skip_ws()
        if self.index >= len(self.text):
            raise _EdnParseError("unexpected end of input")
        char = self.text[self.index]
        if char == "{":
            return self._map()
        if char == "[":
            return self._sequence("]")
        if char == "(":
            return self._sequence(")")
        if char == '"':
            return self._string()
        if char == ":":
            return self._keyword()
        if char == "#" and self._peek(1) == "{":
            self.index += 1
            return self._sequence("}")
        if char == "#":
            return self._tagged()
        return self._atom()

    def _map(self) -> dict[str, Any]:
        self.index += 1
        values: list[Any] = []
        while True:
            self._skip_ws()
            if self._consume("}"):
                break
            values.append(self._value())
        if len(values) % 2:
            raise _EdnParseError("map contains an odd number of forms")
        return {str(values[index]): values[index + 1] for index in range(0, len(values), 2)}

    def _sequence(self, terminator: str) -> list[Any]:
        self.index += 1
        values: list[Any] = []
        while True:
            self._skip_ws()
            if self._consume(terminator):
                return values
            values.append(self._value())

    def _string(self) -> str:
        self.index += 1
        chars: list[str] = []
        while self.index < len(self.text):
            char = self.text[self.index]
            self.index += 1
            if char == '"':
                return "".join(chars)
            if char == "\\":
                if self.index >= len(self.text):
                    raise _EdnParseError("unterminated string escape")
                escaped = self.text[self.index]
                self.index += 1
                chars.append({"n": "\n", "r": "\r", "t": "\t", '"': '"', "\\": "\\"}.get(escaped, escaped))
            else:
                chars.append(char)
        raise _EdnParseError("unterminated string")

    def _keyword(self) -> str:
        self.index += 1
        return self._read_token()

    def _tagged(self) -> Any:
        self.index += 1
        self._read_token()
        return self._value()

    def _atom(self) -> Any:
        token = self._read_token()
        if token == "nil":
            return None
        if token == "true":
            return True
        if token == "false":
            return False
        try:
            return int(token)
        except ValueError:
            pass
        try:
            return float(token)
        except ValueError:
            return token

    def _read_token(self) -> str:
        start = self.index
        while self.index < len(self.text):
            char = self.text[self.index]
            if char.isspace() or char in "{}[](),;":
                break
            self.index += 1
        if self.index == start:
            raise _EdnParseError(f"expected token at byte {self.index}")
        return self.text[start : self.index]

    def _skip_ws(self) -> None:
        while self.index < len(self.text):
            char = self.text[self.index]
            if char.isspace() or char == ",":
                self.index += 1
                continue
            if char == ";":
                while self.index < len(self.text) and self.text[self.index] != "\n":
                    self.index += 1
                continue
            break

    def _consume(self, char: str) -> bool:
        if self.index < len(self.text) and self.text[self.index] == char:
            self.index += 1
            return True
        return False

    def _peek(self, offset: int) -> str:
        index = self.index + offset
        return self.text[index] if index < len(self.text) else ""
