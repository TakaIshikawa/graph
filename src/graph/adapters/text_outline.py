"""Adapter for simple indented plain-text outlines."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


BULLET_RE = re.compile(r"^(?P<bullet>(?:[-*+]|\d+[.)]|[A-Za-z][.)]))\s+(?P<title>.+)$")


@dataclass(frozen=True)
class _OutlineItem:
    source_id: str
    title: str
    raw_line: str
    path: str
    line_number: int
    level: int
    indent: int
    parent_source_id: str | None
    parent_title: str | None
    mtime: float
    ctime: float


class TextOutlineAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "text_outline"

    @property
    def entity_types(self) -> list[str]:
        return ["text_outline_item"]

    def __init__(
        self,
        path: str = "",
        *,
        root_path: str = "",
        source_id_root: str | None = None,
    ) -> None:
        self.path = path or root_path
        self.source_id_root = source_id_root

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "text_outline_item" not in entity_types:
            return result

        root = Path(self.path).expanduser()
        if not root.exists():
            return result

        files = self._text_files(root)
        source_root = Path(self.source_id_root).expanduser() if self.source_id_root else root
        if root.is_file() and not self.source_id_root:
            source_root = root.parent

        sync_at = self._sync_timestamp(since) if since else None
        items: list[_OutlineItem] = []
        for path in files:
            stat = path.stat()
            if sync_at is not None and stat.st_mtime <= sync_at:
                continue

            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            relative_path = self._relative_path(path, source_root)
            items.extend(self._parse_file(text, relative_path, stat.st_mtime, stat.st_ctime))

        for item in items:
            metadata = {
                "path": item.path,
                "line_number": item.line_number,
                "level": item.level,
                "indent": item.indent,
            }
            if item.parent_title:
                metadata["parent_title"] = item.parent_title

            result.units.append(
                KnowledgeUnit(
                    source_project=SourceProject.TEXT_OUTLINE,
                    source_id=item.source_id,
                    source_entity_type="text_outline_item",
                    title=item.title,
                    content=item.raw_line,
                    content_type=ContentType.INSIGHT,
                    metadata=metadata,
                    tags=["outline"],
                    created_at=datetime.fromtimestamp(item.ctime, tz=timezone.utc),
                    updated_at=datetime.fromtimestamp(item.mtime, tz=timezone.utc),
                )
            )

            if item.parent_source_id:
                result.edges.append(
                    KnowledgeEdge(
                        id=self._edge_id(item.parent_source_id, item.source_id),
                        from_unit_id=item.parent_source_id,
                        to_unit_id=item.source_id,
                        relation=EdgeRelation.CONTAINS,
                        source=EdgeSource.SOURCE,
                        metadata={
                            "source_project": SourceProject.TEXT_OUTLINE.value,
                            "from_entity_type": "text_outline_item",
                            "to_entity_type": "text_outline_item",
                            "relation_type": "text_outline_contains",
                            "path": item.path,
                            "parent_title": item.parent_title,
                            "child_title": item.title,
                        },
                    )
                )

        return result

    def _text_files(self, root: Path) -> list[Path]:
        if root.is_file():
            return [root] if root.suffix.lower() == ".txt" else []
        if not root.is_dir():
            return []
        return sorted(path for path in root.rglob("*.txt") if path.is_file())

    def _parse_file(self, text: str, relative_path: str, mtime: float, ctime: float) -> list[_OutlineItem]:
        items: list[_OutlineItem] = []
        stack: list[_OutlineItem] = []
        used_source_ids: set[str] = set()

        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            if self._is_skipped_line(raw_line):
                continue

            indent = self._indent_width(raw_line)
            title = self._title(raw_line)
            if not title:
                continue

            while stack and stack[-1].indent >= indent:
                stack.pop()

            parent = stack[-1] if stack else None
            level = len(stack) + 1
            source_id = self._source_id(relative_path, line_number, raw_line, used_source_ids)
            item = _OutlineItem(
                source_id=source_id,
                title=title,
                raw_line=raw_line,
                path=relative_path,
                line_number=line_number,
                level=level,
                indent=indent,
                parent_source_id=parent.source_id if parent else None,
                parent_title=parent.title if parent else None,
                mtime=mtime,
                ctime=ctime,
            )
            items.append(item)
            stack.append(item)

        return items

    def _is_skipped_line(self, line: str) -> bool:
        stripped = line.strip()
        return not stripped or stripped.startswith("#") or stripped.startswith("//")

    def _indent_width(self, line: str) -> int:
        width = 0
        for char in line:
            if char == " ":
                width += 1
            elif char == "\t":
                width += 4
            else:
                break
        return width

    def _title(self, line: str) -> str:
        stripped = line.strip()
        match = BULLET_RE.match(stripped)
        if match:
            return re.sub(r"\s+", " ", match.group("title")).strip()
        return re.sub(r"\s+", " ", stripped).strip()

    def _source_id(
        self,
        relative_path: str,
        line_number: int,
        raw_line: str,
        used_source_ids: set[str],
    ) -> str:
        salt = 0
        while True:
            raw = f"{relative_path}\0{line_number}\0{raw_line}\0{salt}"
            digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
            source_id = f"text_outline:{digest}"
            if source_id not in used_source_ids:
                used_source_ids.add(source_id)
                return source_id
            salt += 1

    def _edge_id(self, from_source_id: str, to_source_id: str) -> str:
        raw = "|".join(
            [
                SourceProject.TEXT_OUTLINE.value,
                EdgeRelation.CONTAINS.value,
                from_source_id,
                to_source_id,
            ]
        )
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
        return f"text-outline-contains-{digest}"

    def _relative_path(self, path: Path, source_root: Path) -> str:
        try:
            return path.relative_to(source_root).as_posix()
        except ValueError:
            return path.as_posix()

    def _sync_timestamp(self, since: SyncState) -> float:
        if isinstance(since.last_sync_at, datetime):
            return since.last_sync_at.timestamp()
        return datetime.fromisoformat(str(since.last_sync_at)).timestamp()
