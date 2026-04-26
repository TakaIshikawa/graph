"""Adapter for local Markdown notes."""

from __future__ import annotations

import re
from datetime import date, datetime, time, timezone
from pathlib import Path

import yaml

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


FRONT_MATTER_DELIMITER = "---"
INLINE_TAG_RE = re.compile(r"(?<![\w/])#([A-Za-z0-9_/-]*[A-Za-z0-9_])")
WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
MAPPED_FRONT_MATTER_KEYS = {
    "title",
    "tags",
    "created_at",
    "updated_at",
    "content_type",
    "confidence",
    "utility_score",
}


class MarkdownAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "markdown"

    @property
    def entity_types(self) -> list[str]:
        return ["markdown_note"]

    def __init__(self, root_path: str = "") -> None:
        self.root_path = root_path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "markdown_note" not in entity_types:
            return result

        root = Path(self.root_path).expanduser()
        if not root.exists() or not root.is_dir():
            return result

        files = sorted(path for path in root.rglob("*.md") if path.is_file())
        all_notes = [self._read_note(root, path) for path in files]
        source_ids = {note["source_id"] for note in all_notes}
        title_index = self._build_title_index(all_notes)

        for note in all_notes:
            if since and note["mtime"] <= self._sync_timestamp(since):
                continue

            result.units.append(
                KnowledgeUnit(
                    source_project=SourceProject.ME,
                    source_id=note["source_id"],
                    source_entity_type="markdown_note",
                    title=note["title"],
                    content=note["body"],
                    content_type=note["content_type"],
                    metadata={
                        "path": note["source_id"],
                        "front_matter": self._jsonable(note["front_matter"]),
                    },
                    tags=note["tags"],
                    confidence=note["confidence"],
                    utility_score=note["utility_score"],
                    created_at=note["created_at"],
                    updated_at=note["updated_at"],
                )
            )

            for target in self._resolve_wikilinks(note["body"], title_index, source_ids):
                if target == note["source_id"]:
                    continue
                result.edges.append(
                    KnowledgeEdge(
                        from_unit_id=note["source_id"],
                        to_unit_id=target,
                        relation=EdgeRelation.RELATES_TO,
                        source=EdgeSource.SOURCE,
                        metadata={
                            "source_project": str(SourceProject.ME),
                            "from_entity_type": "markdown_note",
                            "to_entity_type": "markdown_note",
                            "relation_type": "wikilink",
                        },
                    )
                )

        return result

    def _read_note(self, root: Path, path: Path) -> dict:
        text = path.read_text(encoding="utf-8")
        front_matter, body = self._split_front_matter(text)
        source_id = path.relative_to(root).as_posix()
        title = str(front_matter.get("title") or path.stem)
        stat = path.stat()
        created_at = self._parse_front_matter_datetime(
            front_matter.get("created_at"),
            fallback=datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc),
        )
        updated_at = self._parse_front_matter_datetime(
            front_matter.get("updated_at"),
            fallback=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
        )

        return {
            "source_id": source_id,
            "title": title,
            "body": body,
            "front_matter": self._unmapped_front_matter(front_matter),
            "tags": self._collect_tags(front_matter, body),
            "mtime": stat.st_mtime,
            "created_at": created_at,
            "updated_at": updated_at,
            "content_type": self._parse_content_type(front_matter.get("content_type")),
            "confidence": self._parse_float(front_matter.get("confidence")),
            "utility_score": self._parse_float(front_matter.get("utility_score")),
        }

    def _split_front_matter(self, text: str) -> tuple[dict, str]:
        lines = text.splitlines()
        if not lines or lines[0].strip() != FRONT_MATTER_DELIMITER:
            return {}, text

        for index, line in enumerate(lines[1:], start=1):
            if line.strip() == FRONT_MATTER_DELIMITER:
                raw_front_matter = "\n".join(lines[1:index])
                body = "\n".join(lines[index + 1 :])
                if text.endswith("\n"):
                    body += "\n"
                data = yaml.safe_load(raw_front_matter) or {}
                return data if isinstance(data, dict) else {}, body

        return {}, text

    def _collect_tags(self, front_matter: dict, body: str) -> list[str]:
        tags: list[str] = []
        front_matter_tags = front_matter.get("tags", [])
        if isinstance(front_matter_tags, str):
            front_matter_tags = [
                tag.strip() for tag in front_matter_tags.split(",") if tag.strip()
            ]
        elif not isinstance(front_matter_tags, list):
            front_matter_tags = []

        for tag in front_matter_tags:
            self._append_tag(tags, str(tag))
        for match in INLINE_TAG_RE.finditer(body):
            self._append_tag(tags, match.group(1))
        return tags

    def _append_tag(self, tags: list[str], tag: str) -> None:
        normalized = tag.strip().removeprefix("#").strip()
        if normalized and normalized not in tags:
            tags.append(normalized)

    def _unmapped_front_matter(self, front_matter: dict) -> dict:
        return {
            key: value
            for key, value in front_matter.items()
            if str(key) not in MAPPED_FRONT_MATTER_KEYS
        }

    def _parse_front_matter_datetime(self, value, *, fallback: datetime) -> datetime:
        if isinstance(value, datetime):
            parsed = value
        elif isinstance(value, date):
            parsed = datetime.combine(value, time.min)
        elif isinstance(value, str):
            raw = value.strip()
            if not raw:
                return fallback
            try:
                parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                return fallback
        else:
            return fallback

        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _parse_content_type(self, value) -> ContentType:
        if isinstance(value, ContentType):
            return value
        if isinstance(value, str):
            try:
                return ContentType(value.strip().lower())
            except ValueError:
                return ContentType.INSIGHT
        return ContentType.INSIGHT

    def _parse_float(self, value) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _build_title_index(self, notes: list[dict]) -> dict[str, str]:
        index: dict[str, str] = {}
        for note in notes:
            source_id = note["source_id"]
            keys = {
                note["title"],
                Path(source_id).stem,
                source_id,
                source_id.removesuffix(".md"),
            }
            for key in keys:
                index.setdefault(self._normalize_link_target(key), source_id)
        return index

    def _resolve_wikilinks(
        self,
        body: str,
        title_index: dict[str, str],
        source_ids: set[str],
    ) -> list[str]:
        targets: list[str] = []
        for match in WIKILINK_RE.finditer(body):
            target = match.group(1).split("|", 1)[0].split("#", 1)[0].strip()
            if not target:
                continue
            if not target.endswith(".md"):
                direct_path = f"{target}.md"
            else:
                direct_path = target

            source_id = direct_path if direct_path in source_ids else None
            source_id = source_id or title_index.get(self._normalize_link_target(target))
            if source_id and source_id not in targets:
                targets.append(source_id)
        return targets

    def _normalize_link_target(self, target: str) -> str:
        return target.strip().removesuffix(".md").lower()

    def _sync_timestamp(self, since: SyncState) -> float:
        if isinstance(since.last_sync_at, datetime):
            return since.last_sync_at.timestamp()
        return datetime.fromisoformat(str(since.last_sync_at)).timestamp()

    def _jsonable(self, value):
        if isinstance(value, dict):
            return {str(key): self._jsonable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._jsonable(item) for item in value]
        if isinstance(value, str | int | float | bool) or value is None:
            return value
        return str(value)
