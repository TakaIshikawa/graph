"""Adapter for Markdown task list items."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


TASK_RE = re.compile(r"^\s*[-*+]\s+\[([ xX])\]\s+(.+?)\s*$")
HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
FENCE_RE = re.compile(r"^\s{0,3}(```+|~~~+)")
DATE_TOKEN_RE = re.compile(
    r"(?:^|\s)(?:(due|date)\s*[:=]\s*|@(due|date)\()(\d{4}-\d{2}-\d{2})(?:\))?",
    re.IGNORECASE,
)
BRACKET_DATE_TOKEN_RE = re.compile(
    r"(?:^|\s)\[(due|date)::\s*(\d{4}-\d{2}-\d{2})\]",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _MarkdownTask:
    text: str
    completed: bool
    file_path: str
    line_number: int
    heading_path: list[str]
    due: str | None = None
    date: str | None = None


class MarkdownTasksAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "markdown_tasks"

    @property
    def entity_types(self) -> list[str]:
        return ["markdown_task"]

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
        if entity_types and "markdown_task" not in entity_types:
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
            for task in self._extract_tasks(file_path, relative_path):
                metadata = {
                    "source_file": task.file_path,
                    "line_number": task.line_number,
                    "completed": task.completed,
                    "heading_path": task.heading_path,
                }
                if task.due:
                    metadata["due"] = task.due
                if task.date:
                    metadata["date"] = task.date

                result.units.append(
                    KnowledgeUnit(
                        source_project=SourceProject.MARKDOWN_TASKS,
                        source_id=self._source_id(task),
                        source_entity_type="markdown_task",
                        title=task.text,
                        content=task.text,
                        content_type=ContentType.ARTIFACT,
                        metadata=metadata,
                        tags=["markdown-task"],
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

    def _extract_tasks(self, path: Path, relative_path: str) -> list[_MarkdownTask]:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        tasks: list[_MarkdownTask] = []
        headings: list[str] = []
        in_fence = False

        for line_number, line in enumerate(lines, start=1):
            if FENCE_RE.match(line):
                in_fence = not in_fence
                continue
            if in_fence:
                continue

            heading = HEADING_RE.match(line)
            if heading:
                level = len(heading.group(1))
                headings = headings[: level - 1]
                headings.append(self._clean_heading(heading.group(2)))
                continue

            task_match = TASK_RE.match(line)
            if not task_match:
                continue

            text = self._clean_text(task_match.group(2))
            if not text:
                continue
            tokens = self._date_tokens(text)
            tasks.append(
                _MarkdownTask(
                    text=text,
                    completed=task_match.group(1).lower() == "x",
                    file_path=relative_path,
                    line_number=line_number,
                    heading_path=list(headings),
                    due=tokens.get("due"),
                    date=tokens.get("date"),
                )
            )

        return tasks

    def _date_tokens(self, text: str) -> dict[str, str]:
        tokens: dict[str, str] = {}
        for match in DATE_TOKEN_RE.finditer(text):
            name = (match.group(1) or match.group(2) or "").lower()
            if name and name not in tokens:
                tokens[name] = match.group(3)
        for match in BRACKET_DATE_TOKEN_RE.finditer(text):
            name = match.group(1).lower()
            if name not in tokens:
                tokens[name] = match.group(2)
        return tokens

    def _source_id(self, task: _MarkdownTask) -> str:
        digest = hashlib.sha256(
            f"{task.file_path}\0{task.line_number}".encode("utf-8")
        ).hexdigest()[:16]
        return f"markdown_tasks:{digest}"

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
