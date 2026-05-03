"""Adapter for definition-style entries in Markdown files."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


INLINE_DEFINITION_RE = re.compile(
    r"^\s{0,3}(?![-*+]\s)([^:\n][^:\n]{0,200}?)::\s+(.+?)\s*$"
)
DEFINITION_MARKER_RE = re.compile(r"^\s{0,3}:\s+(.+?)\s*$")
HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
FENCE_RE = re.compile(r"^\s{0,3}(```+|~~~+)")
TAG_RE = re.compile(r"(?<![\w/])#([A-Za-z0-9][A-Za-z0-9_/-]*)")
LIST_ITEM_RE = re.compile(r"^\s{0,3}[-*+]\s+")


@dataclass(frozen=True)
class _MarkdownDefinition:
    term: str
    definition: str
    file_path: str
    line_number: int
    heading_path: list[str]


class MarkdownDefinitionsAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "markdown_definitions"

    @property
    def entity_types(self) -> list[str]:
        return ["markdown_definition"]

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
        if entity_types and "markdown_definition" not in entity_types:
            return result

        root = Path(self.path).expanduser()
        if not root.exists():
            return result

        files = self._markdown_files(root)
        source_root = Path(self.source_id_root).expanduser() if self.source_id_root else root
        if root.is_file() and not self.source_id_root:
            source_root = root.parent

        sync_at = self._sync_datetime(since) if since else None
        for file_path in files:
            stat = file_path.stat()
            updated_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            if sync_at and updated_at <= sync_at:
                continue

            relative_path = self._relative_path(file_path, source_root)
            for definition in self._extract_definitions(file_path, relative_path):
                result.units.append(
                    KnowledgeUnit(
                        source_project=SourceProject.MARKDOWN_DEFINITIONS,
                        source_id=self._source_id(definition),
                        source_entity_type="markdown_definition",
                        title=definition.term,
                        content=definition.definition,
                        content_type=ContentType.INSIGHT,
                        metadata={
                            "term": definition.term,
                            "source_file": definition.file_path,
                            "file_path": definition.file_path,
                            "line_number": definition.line_number,
                            "heading_path": definition.heading_path,
                        },
                        tags=self._tags(definition.definition),
                        created_at=updated_at,
                        updated_at=updated_at,
                    )
                )

        return result

    def _markdown_files(self, root: Path) -> list[Path]:
        if root.is_file():
            return [root] if root.suffix.lower() in {".md", ".markdown"} else []
        if not root.is_dir():
            return []
        return sorted(
            path
            for pattern in ("*.md", "*.markdown")
            for path in root.rglob(pattern)
            if path.is_file()
        )

    def _extract_definitions(
        self, path: Path, relative_path: str
    ) -> list[_MarkdownDefinition]:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        definitions: list[_MarkdownDefinition] = []
        headings: list[str] = []
        in_fence = False
        index = 0

        while index < len(lines):
            line = lines[index]
            if FENCE_RE.match(line):
                in_fence = not in_fence
                index += 1
                continue
            if in_fence:
                index += 1
                continue

            heading = HEADING_RE.match(line)
            if heading:
                level = len(heading.group(1))
                term = self._clean_heading(heading.group(2))
                headings = headings[: level - 1]
                headings.append(term)
                paragraph, next_index = self._following_paragraph(lines, index + 1)
                if paragraph:
                    definitions.append(
                        _MarkdownDefinition(
                            term=term,
                            definition=paragraph,
                            file_path=relative_path,
                            line_number=index + 1,
                            heading_path=list(headings),
                        )
                    )
                index = max(index + 1, next_index)
                continue

            inline = INLINE_DEFINITION_RE.match(line)
            if inline:
                term = self._clean_text(inline.group(1))
                definition = self._clean_text(inline.group(2))
                if term and definition:
                    definitions.append(
                        _MarkdownDefinition(
                            term=term,
                            definition=definition,
                            file_path=relative_path,
                            line_number=index + 1,
                            heading_path=list(headings),
                        )
                    )
                index += 1
                continue

            marker = self._definition_marker(lines, index)
            if marker:
                term, definition = marker
                definitions.append(
                    _MarkdownDefinition(
                        term=term,
                        definition=definition,
                        file_path=relative_path,
                        line_number=index + 1,
                        heading_path=list(headings),
                    )
                )
                index += 2
                continue

            index += 1

        return definitions

    def _definition_marker(self, lines: list[str], index: int) -> tuple[str, str] | None:
        if index + 1 >= len(lines):
            return None
        term = self._clean_text(lines[index])
        if not term or self._is_block_boundary(lines[index]) or term.endswith(":"):
            return None
        marker = DEFINITION_MARKER_RE.match(lines[index + 1])
        if not marker:
            return None
        definition = self._clean_text(marker.group(1))
        if not definition:
            return None
        return term, definition

    def _following_paragraph(
        self, lines: list[str], start_index: int
    ) -> tuple[str, int]:
        index = start_index
        while index < len(lines) and not lines[index].strip():
            index += 1
        if index >= len(lines) or self._is_block_boundary(lines[index]):
            return "", index

        paragraph: list[str] = []
        while index < len(lines):
            line = lines[index]
            if not paragraph and index + 1 < len(lines) and DEFINITION_MARKER_RE.match(
                lines[index + 1]
            ):
                return "", index
            if not line.strip() or self._is_block_boundary(line):
                break
            paragraph.append(line.strip())
            index += 1
        return self._clean_text(" ".join(paragraph)), index

    def _is_block_boundary(self, line: str) -> bool:
        return bool(
            HEADING_RE.match(line)
            or FENCE_RE.match(line)
            or DEFINITION_MARKER_RE.match(line)
            or LIST_ITEM_RE.match(line)
            or INLINE_DEFINITION_RE.match(line)
        )

    def _tags(self, definition: str) -> list[str]:
        tags: list[str] = []
        for match in TAG_RE.finditer(definition):
            tag = match.group(1).strip("_-/").lower()
            if tag and tag not in tags:
                tags.append(tag)
        return tags

    def _source_id(self, definition: _MarkdownDefinition) -> str:
        digest = hashlib.sha256(
            f"{definition.file_path}\0{definition.line_number}\0{definition.term}".encode(
                "utf-8"
            )
        ).hexdigest()[:16]
        return f"markdown_definitions:{digest}"

    def _relative_path(self, path: Path, source_root: Path) -> str:
        try:
            return path.relative_to(source_root).as_posix()
        except ValueError:
            return path.as_posix()

    def _clean_heading(self, value: str) -> str:
        return value.strip().strip("#").strip()

    def _clean_text(self, value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()

    def _sync_datetime(self, since: SyncState) -> datetime:
        value = since.last_sync_at
        if isinstance(value, datetime):
            parsed = value
        else:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
