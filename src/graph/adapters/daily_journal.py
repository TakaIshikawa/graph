"""Adapter for daily journal Markdown files."""

from __future__ import annotations

import hashlib
import re
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any

import yaml

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


FRONTMATTER_DELIMITER = "---"
DATE_RE = re.compile(r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)")
TAG_SPLIT_RE = re.compile(r"[,;|]")


class DailyJournalAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "daily_journal"

    @property
    def entity_types(self) -> list[str]:
        return ["journal_entry"]

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
        if entity_types and "journal_entry" not in entity_types:
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
            file_updated_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            if sync_at and file_updated_at <= sync_at:
                continue

            relative_path = self._relative_path(file_path, source_root)
            text = file_path.read_text(encoding="utf-8", errors="replace")
            frontmatter, body = self._split_frontmatter(text, file_path)
            journal_date = self._journal_date(frontmatter, file_path)
            journal_datetime = datetime.combine(
                journal_date, time.min, tzinfo=timezone.utc
            )
            updated_at = self._updated_at(frontmatter, file_updated_at)

            result.units.append(
                KnowledgeUnit(
                    source_project=SourceProject.DAILY_JOURNAL,
                    source_id=self._source_id(relative_path, journal_date),
                    source_entity_type="journal_entry",
                    title=self._title(frontmatter, journal_date, file_path),
                    content=body,
                    content_type=ContentType.ARTIFACT,
                    metadata={
                        "source_file": relative_path,
                        "journal_date": journal_date.isoformat(),
                        "frontmatter": self._jsonable(frontmatter),
                    },
                    tags=self._tags(frontmatter.get("tags")),
                    created_at=journal_datetime,
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

    def _split_frontmatter(self, text: str, path: Path) -> tuple[dict[str, Any], str]:
        lines = text.splitlines()
        if not lines or lines[0].strip() != FRONTMATTER_DELIMITER:
            return {}, text

        for index, line in enumerate(lines[1:], start=1):
            if line.strip() != FRONTMATTER_DELIMITER:
                continue

            raw_frontmatter = "\n".join(lines[1:index])
            body = "\n".join(lines[index + 1 :])
            if text.endswith("\n"):
                body += "\n"
            try:
                data = yaml.safe_load(raw_frontmatter) or {}
            except yaml.YAMLError as exc:
                raise ValueError(f"Invalid YAML frontmatter in {path}: {exc}") from exc
            if not isinstance(data, dict):
                raise ValueError(f"Invalid YAML frontmatter in {path}: expected mapping")
            return data, body

        return {}, text

    def _journal_date(self, frontmatter: dict[str, Any], path: Path) -> date:
        if "date" in frontmatter:
            return self._date_from_value(frontmatter["date"], path, "frontmatter date")
        if "title" in frontmatter:
            parsed = self._date_from_text(frontmatter["title"])
            if parsed is not None:
                return parsed

        parsed = self._date_from_text(path.name)
        if parsed is not None:
            return parsed

        raise ValueError(
            f"Missing journal date in {path}: expected YYYY-MM-DD in frontmatter "
            "date/title or filename"
        )

    def _date_from_value(self, value: Any, path: Path, field_name: str) -> date:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        parsed = self._date_from_text(value)
        if parsed is not None:
            return parsed
        raise ValueError(f"Invalid {field_name} in {path}: expected YYYY-MM-DD")

    def _date_from_text(self, value: Any) -> date | None:
        match = DATE_RE.search(str(value))
        if not match:
            return None
        try:
            return date.fromisoformat(match.group(1))
        except ValueError as exc:
            raise ValueError(f"Invalid journal date: {match.group(1)}") from exc

    def _updated_at(
        self, frontmatter: dict[str, Any], fallback: datetime
    ) -> datetime:
        for key in ("updated", "updated_at", "modified", "lastmod"):
            parsed = self._datetime_from_value(frontmatter.get(key))
            if parsed is not None:
                return parsed
        return fallback

    def _datetime_from_value(self, value: Any) -> datetime | None:
        if isinstance(value, datetime):
            parsed = value
        elif isinstance(value, date):
            parsed = datetime.combine(value, time.min)
        elif isinstance(value, str):
            raw = value.strip()
            if not raw:
                return None
            try:
                parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                return None
        else:
            return None

        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _title(
        self, frontmatter: dict[str, Any], journal_date: date, file_path: Path
    ) -> str:
        title = frontmatter.get("title")
        if title is not None and str(title).strip():
            return str(title).strip()
        if file_path.stem:
            return file_path.stem
        return journal_date.isoformat()

    def _tags(self, value: Any) -> list[str]:
        tags = ["journal"]
        raw_tags: list[Any]
        if isinstance(value, str):
            raw_tags = TAG_SPLIT_RE.split(value)
        elif isinstance(value, list | tuple | set):
            raw_tags = list(value)
        else:
            raw_tags = []

        for raw_tag in raw_tags:
            if isinstance(raw_tag, str) and TAG_SPLIT_RE.search(raw_tag):
                candidates = TAG_SPLIT_RE.split(raw_tag)
            else:
                candidates = [raw_tag]
            for candidate in candidates:
                tag = str(candidate).strip().removeprefix("#").strip().lower()
                if tag and tag not in tags:
                    tags.append(tag)
        return tags

    def _source_id(self, relative_path: str, journal_date: date) -> str:
        raw = f"{relative_path}|{journal_date.isoformat()}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        return f"daily_journal:{journal_date.isoformat()}:{digest}"

    def _relative_path(self, path: Path, source_root: Path) -> str:
        try:
            return path.relative_to(source_root).as_posix()
        except ValueError:
            return path.as_posix()

    def _sync_datetime(self, since: SyncState) -> datetime:
        value = since.last_sync_at
        if isinstance(value, datetime):
            parsed = value
        else:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _jsonable(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._jsonable(item) for key, item in value.items()}
        if isinstance(value, list | tuple | set):
            return [self._jsonable(item) for item in value]
        if isinstance(value, datetime | date):
            return value.isoformat()
        if isinstance(value, str | int | float | bool) or value is None:
            return value
        return str(value)
