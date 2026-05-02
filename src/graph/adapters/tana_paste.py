"""Adapter for Tana Paste plain-text outlines."""

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


BULLET_RE = re.compile(r"^(?P<indent>[ \t]*)(?P<bullet>[-*+])\s+(?P<text>.*)$")
HASHTAG_RE = re.compile(r"(?<![\w/])#([A-Za-z0-9_/-]*[A-Za-z0-9_])")
WIKI_REF_RE = re.compile(r"\[\[([^\]]+)\]\]")


@dataclass
class _TanaBullet:
    source_id: str
    title: str
    first_line_text: str
    content_lines: list[str]
    path: str
    line_number: int
    level: int
    indent: int
    bullet_marker: str
    parent_source_id: str | None
    parent_title: str | None
    tags: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)

    @property
    def content(self) -> str:
        return "\n".join(line for line in self.content_lines if line).strip() or self.title


class TanaPasteAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "tana_paste"

    @property
    def entity_types(self) -> list[str]:
        return ["bullet"]

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
        if entity_types and "bullet" not in entity_types:
            return result

        root = Path(self.path).expanduser()
        if not root.exists():
            return result

        files = self._text_files(root)
        source_root = Path(self.source_id_root).expanduser() if self.source_id_root else root
        if root.is_file() and not self.source_id_root:
            source_root = root.parent

        sync_at = self._sync_timestamp(since) if since else None
        included_source_ids: set[str] = set()
        pending_edges: list[tuple[str, str, str, str, str]] = []

        for path in files:
            try:
                stat = path.stat()
            except OSError:
                continue
            if sync_at is not None and stat.st_mtime <= sync_at:
                continue

            try:
                text = path.read_text(encoding="utf-8-sig")
            except (OSError, UnicodeDecodeError):
                continue

            relative_path = self._relative_path(path, source_root)
            bullets = self._parse_file(text, relative_path)
            created_at = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc)
            updated_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

            for bullet in bullets:
                included_source_ids.add(bullet.source_id)
                result.units.append(self._unit(bullet, created_at, updated_at))
                if bullet.parent_source_id:
                    pending_edges.append(
                        (
                            bullet.parent_source_id,
                            bullet.source_id,
                            bullet.path,
                            bullet.parent_title or "",
                            bullet.title,
                        )
                    )

        emitted_edges: set[tuple[str, str]] = set()
        for parent_id, child_id, path, parent_title, child_title in pending_edges:
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
                        "source_project": SourceProject.TANA_PASTE.value,
                        "from_entity_type": "bullet",
                        "to_entity_type": "bullet",
                        "relation_type": "tana_paste_contains",
                        "path": path,
                        "parent_title": parent_title,
                        "child_title": child_title,
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

    def _parse_file(self, text: str, relative_path: str) -> list[_TanaBullet]:
        bullets: list[_TanaBullet] = []
        stack: list[_TanaBullet] = []
        last_bullet: _TanaBullet | None = None
        used_source_ids: set[str] = set()

        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            if not raw_line.strip():
                continue

            match = BULLET_RE.match(raw_line)
            if match is None:
                if last_bullet is not None:
                    continuation = raw_line.strip()
                    if continuation:
                        last_bullet.content_lines.append(continuation)
                        self._refresh_extracted_metadata(last_bullet)
                continue

            indent = self._indent_width(match.group("indent"))
            bullet_text = re.sub(r"\s+", " ", match.group("text")).strip()
            if not bullet_text:
                continue

            while stack and stack[-1].indent >= indent:
                stack.pop()

            parent = stack[-1] if stack else None
            title = self._clean_title(bullet_text)
            source_id = self._source_id(relative_path, line_number, raw_line, used_source_ids)
            bullet = _TanaBullet(
                source_id=source_id,
                title=title or "Untitled Tana bullet",
                first_line_text=bullet_text,
                content_lines=[bullet_text],
                path=relative_path,
                line_number=line_number,
                level=len(stack) + 1,
                indent=indent,
                bullet_marker=match.group("bullet"),
                parent_source_id=parent.source_id if parent else None,
                parent_title=parent.title if parent else None,
            )
            self._refresh_extracted_metadata(bullet)
            bullets.append(bullet)
            stack.append(bullet)
            last_bullet = bullet

        return bullets

    def _unit(self, bullet: _TanaBullet, created_at: datetime, updated_at: datetime) -> KnowledgeUnit:
        metadata: dict[str, Any] = {
            "path": bullet.path,
            "line_number": bullet.line_number,
            "level": bullet.level,
            "indent": bullet.indent,
            "bullet_marker": bullet.bullet_marker,
            "tags": bullet.tags,
            "references": bullet.references,
            "raw_title": bullet.first_line_text,
        }
        if bullet.parent_source_id:
            metadata["parent_source_id"] = bullet.parent_source_id
        if bullet.parent_title:
            metadata["parent_title"] = bullet.parent_title

        return KnowledgeUnit(
            source_project=SourceProject.TANA_PASTE,
            source_id=bullet.source_id,
            source_entity_type="bullet",
            title=bullet.title,
            content=bullet.content,
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=bullet.tags,
            created_at=created_at,
            updated_at=updated_at,
        )

    def _refresh_extracted_metadata(self, bullet: _TanaBullet) -> None:
        content = bullet.content
        bullet.tags = self._dedupe(self._normalize_tag(match) for match in HASHTAG_RE.findall(content))
        bullet.references = self._dedupe(self._normalize_ref(match) for match in WIKI_REF_RE.findall(content))

    def _clean_title(self, text: str) -> str:
        text = WIKI_REF_RE.sub("", text)
        text = HASHTAG_RE.sub("", text)
        text = re.sub(r"\s+", " ", text).strip(" -\t")
        return text if len(text) <= 80 else f"{text[:77].rstrip()}..."

    def _normalize_tag(self, value: str) -> str:
        return re.sub(r"\s+", " ", value.strip().removeprefix("#")).strip().lower()

    def _normalize_ref(self, value: str) -> str:
        return re.sub(r"\s+", " ", value.strip()).strip()

    def _dedupe(self, values: Any) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            if not value:
                continue
            key = str(value).casefold()
            if key in seen:
                continue
            result.append(value)
            seen.add(key)
        return result

    def _indent_width(self, indent: str) -> int:
        width = 0
        for char in indent:
            if char == " ":
                width += 1
            elif char == "\t":
                width += 4
        return width

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
            source_id = f"tana_paste:{digest}"
            if source_id not in used_source_ids:
                used_source_ids.add(source_id)
                return source_id
            salt += 1

    def _edge_id(self, from_source_id: str, to_source_id: str) -> str:
        raw = "|".join(
            [
                SourceProject.TANA_PASTE.value,
                EdgeRelation.CONTAINS.value,
                from_source_id,
                to_source_id,
            ]
        )
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
        return f"tana-paste-contains-{digest}"

    def _relative_path(self, path: Path, source_root: Path) -> str:
        try:
            return path.relative_to(source_root).as_posix()
        except ValueError:
            return path.as_posix()

    def _sync_timestamp(self, since: SyncState) -> float:
        if isinstance(since.last_sync_at, datetime):
            return since.last_sync_at.timestamp()
        return datetime.fromisoformat(str(since.last_sync_at).replace("Z", "+00:00")).timestamp()
