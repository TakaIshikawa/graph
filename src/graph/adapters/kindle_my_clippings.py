"""Adapter for Amazon Kindle My Clippings.txt exports."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class KindleMyClippingsAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "kindle_my_clippings"

    @property
    def entity_types(self) -> list[str]:
        return ["clipping"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        allowed = set(entity_types or self.entity_types)
        if not allowed.intersection(self.entity_types):
            return result

        sync_at = self._sync_datetime(since) if since else None
        for path in self._iter_paths():
            try:
                blocks = self._read_blocks(path)
            except (OSError, UnicodeDecodeError):
                continue
            for index, block in enumerate(blocks):
                unit = self._unit_from_block(block, path, index)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)

        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _iter_paths(self) -> list[Path]:
        path = Path(self.path).expanduser() if self.path else None
        if path is None:
            return []
        if path.is_file():
            return [path]
        if path.is_dir():
            return sorted(child for child in path.rglob("*.txt") if child.is_file())
        return []

    def _read_blocks(self, path: Path) -> list[list[str]]:
        text = path.read_text(encoding="utf-8-sig")
        return [
            [line.rstrip() for line in block.strip().splitlines()]
            for block in re.split(r"\n=+\s*(?:\n|$)", text)
            if block.strip()
        ]

    def _unit_from_block(self, lines: list[str], path: Path, block_index: int) -> KnowledgeUnit | None:
        if len(lines) < 2:
            return None

        title, author = self._parse_title(lines[0].strip())
        details = lines[1].strip()
        if not title or not details.startswith("-"):
            return None

        clipping_type = self._parse_type(details)
        page = self._parse_position(details, "page")
        location = self._parse_position(details, "location")
        clipped_at_text = self._parse_clipped_at(details)
        clipped_at = self._parse_datetime(clipped_at_text)
        body = "\n".join(line for line in lines[3:] if line.strip()).strip()
        if not body and clipping_type != "bookmark":
            return None

        content = body or self._bookmark_content(title, author, page, location)
        created_at = clipped_at or datetime.now(timezone.utc)
        metadata = {
            "title": title,
            "author": author,
            "clipping_type": clipping_type,
            "location": location,
            "page": page,
            "clipped_at": clipped_at_text,
            "source_file": str(path),
            "block_index": block_index,
        }
        return KnowledgeUnit(
            source_project=SourceProject.KINDLE,
            source_id=self._source_id(title, author, details, body, block_index),
            source_entity_type="clipping",
            title=self._display_title(title, clipping_type, page, location),
            content=content,
            content_type=ContentType.INSIGHT if clipping_type in {"highlight", "note"} else ContentType.METADATA,
            metadata=metadata,
            tags=[author] if author else [],
            created_at=created_at,
            updated_at=created_at,
        )

    def _parse_title(self, value: str) -> tuple[str, str]:
        match = re.match(r"^(?P<title>.+?)\s+\((?P<author>[^()]*)\)\s*$", value)
        if match:
            return match.group("title").strip(), match.group("author").strip()
        return value.strip(), ""

    def _parse_type(self, details: str) -> str:
        match = re.search(r"\bYour\s+([A-Za-z]+)", details)
        clipping_type = match.group(1).casefold() if match else "clipping"
        return clipping_type if clipping_type in {"highlight", "note", "bookmark"} else clipping_type

    def _parse_position(self, details: str, name: str) -> str:
        match = re.search(rf"\b{name}\s+([^|]+)", details, flags=re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _parse_clipped_at(self, details: str) -> str:
        match = re.search(r"\bAdded on\s+(.+)$", details)
        return match.group(1).strip() if match else ""

    def _parse_datetime(self, value: str) -> datetime | None:
        if not value:
            return None
        for fmt in (
            "%A, %B %d, %Y %I:%M:%S %p",
            "%A, %B %d, %Y %H:%M:%S",
            "%B %d, %Y %I:%M:%S %p",
            "%B %d, %Y %H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ):
            try:
                return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)

    def _display_title(self, title: str, clipping_type: str, page: str, location: str) -> str:
        label = clipping_type.title() if clipping_type else "Clipping"
        position = " - ".join(part for part in (f"Page {page}" if page else "", f"Location {location}" if location else "") if part)
        return f"{title}: {label} ({position})" if position else f"{title}: {label}"

    def _bookmark_content(self, title: str, author: str, page: str, location: str) -> str:
        parts = [f"Bookmark in {title}"]
        if author:
            parts.append(f"Author: {author}")
        if page:
            parts.append(f"Page: {page}")
        if location:
            parts.append(f"Location: {location}")
        return "\n".join(parts)

    def _source_id(self, title: str, author: str, details: str, body: str, block_index: int) -> str:
        digest = hashlib.sha256("|".join((title, author, details, body, str(block_index))).encode("utf-8")).hexdigest()[:24]
        return f"kindle_my_clippings:{digest}"

    def _sync_datetime(self, since: SyncState) -> datetime:
        value = since.last_sync_at
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
