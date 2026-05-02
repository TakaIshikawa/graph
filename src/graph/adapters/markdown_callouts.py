"""Adapter for Obsidian-style Markdown callouts."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


CALLOUT_START_RE = re.compile(
    r"^\s{0,3}>\s*\[!([A-Za-z0-9_-]+)\]([+-]?)(?:\s+(.*))?$"
)
BLOCKQUOTE_RE = re.compile(r"^\s{0,3}>(?: ?(.*))?$")
HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")


@dataclass(frozen=True)
class _Callout:
    callout_type: str
    title: str
    body: str
    file_path: str
    line_number: int
    headings: list[str]
    modifier: str


class MarkdownCalloutsAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "markdown_callouts"

    @property
    def entity_types(self) -> list[str]:
        return ["markdown_callout"]

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
        if entity_types and "markdown_callout" not in entity_types:
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
            for callout in self._extract_callouts(file_path, relative_path):
                result.units.append(
                    KnowledgeUnit(
                        source_project=SourceProject.MARKDOWN_CALLOUTS,
                        source_id=self._source_id(callout),
                        source_entity_type="markdown_callout",
                        title=callout.title or f"{callout.callout_type.title()} callout",
                        content=callout.body,
                        content_type=ContentType.INSIGHT,
                        metadata={
                            "callout_type": callout.callout_type,
                            "title": callout.title,
                            "body": callout.body,
                            "source_path": callout.file_path,
                            "path": callout.file_path,
                            "line_number": callout.line_number,
                            "heading": callout.headings[-1] if callout.headings else "",
                            "headings": callout.headings,
                            "modifier": callout.modifier,
                        },
                        tags=["markdown-callout", f"callout-{callout.callout_type}"],
                        created_at=updated_at,
                        updated_at=updated_at,
                    )
                )

        return result

    def _markdown_files(self, root: Path) -> list[Path]:
        if root.is_file():
            return [root] if root.suffix.lower() == ".md" else []
        if not root.is_dir():
            return []
        return sorted(path for path in root.rglob("*.md") if path.is_file())

    def _extract_callouts(self, path: Path, relative_path: str) -> list[_Callout]:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        callouts: list[_Callout] = []
        headings: list[str] = []
        index = 0

        while index < len(lines):
            line = lines[index]
            heading = HEADING_RE.match(line)
            if heading:
                level = len(heading.group(1))
                title = self._clean_heading(heading.group(2))
                headings = headings[: level - 1]
                headings.append(title)
                index += 1
                continue

            start = CALLOUT_START_RE.match(line)
            if not start:
                index += 1
                continue

            callout_type = start.group(1).strip().lower()
            modifier = start.group(2)
            title = (start.group(3) or "").strip()
            line_number = index + 1
            body_lines: list[str] = []
            index += 1

            while index < len(lines):
                if CALLOUT_START_RE.match(lines[index]):
                    break
                body_match = BLOCKQUOTE_RE.match(lines[index])
                if not body_match:
                    break
                body_lines.append(body_match.group(1) or "")
                index += 1

            callouts.append(
                _Callout(
                    callout_type=callout_type,
                    title=title,
                    body="\n".join(body_lines).strip(),
                    file_path=relative_path,
                    line_number=line_number,
                    headings=list(headings),
                    modifier=modifier,
                )
            )

        return callouts

    def _clean_heading(self, value: str) -> str:
        return value.strip().strip("#").strip()

    def _source_id(self, callout: _Callout) -> str:
        raw = f"{callout.file_path}:{callout.line_number}:{callout.callout_type}:{callout.title}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        return f"markdown_callouts:{digest}"

    def _relative_path(self, path: Path, source_root: Path) -> str:
        try:
            return path.relative_to(source_root).as_posix()
        except ValueError:
            return path.as_posix()

    def _sync_datetime(self, since: SyncState) -> datetime:
        value = since.last_sync_at
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
