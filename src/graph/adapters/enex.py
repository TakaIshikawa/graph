"""Adapter for local Evernote ENEX exports."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


BLOCK_TAGS = {
    "address",
    "blockquote",
    "br",
    "dd",
    "div",
    "dl",
    "dt",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "hr",
    "li",
    "ol",
    "p",
    "pre",
    "table",
    "td",
    "th",
    "tr",
    "ul",
}
SKIP_TAGS = {"en-media", "script", "style"}
DOCTYPE_RE = re.compile(r"<!DOCTYPE[^>]*>", re.IGNORECASE)


class _EnexContentTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag in BLOCK_TAGS:
            self._separator()

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in SKIP_TAGS:
            return
        if tag in BLOCK_TAGS:
            self._separator()

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
            return
        if tag in BLOCK_TAGS:
            self._separator()

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = unescape(data).strip()
        if text:
            self.parts.append(text)

    def _separator(self) -> None:
        if self.parts and self.parts[-1] != "\n":
            self.parts.append("\n")

    def text(self) -> str:
        lines: list[str] = []
        current: list[str] = []
        for part in self.parts:
            if part == "\n":
                if current:
                    lines.append(" ".join(" ".join(current).split()))
                    current = []
            else:
                current.append(part)
        if current:
            lines.append(" ".join(" ".join(current).split()))
        return "\n".join(line for line in lines if line)


class EnexAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "enex"

    @property
    def entity_types(self) -> list[str]:
        return ["note"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "note" not in entity_types:
            return result

        paths = self._discover_paths()
        if not paths:
            return result

        root = Path(self.path).expanduser()
        source_root = root if root.is_dir() else root.parent
        sync_at = self._sync_datetime(since) if since else None
        for path in paths:
            for unit in self._read_notes(source_root, path):
                changed_at = unit.updated_at or unit.created_at
                if sync_at is not None and changed_at <= sync_at:
                    continue
                result.units.append(unit)

        return result

    def _discover_paths(self) -> list[Path]:
        configured = Path(self.path).expanduser()
        if configured.is_file() and configured.suffix.lower() == ".enex":
            return [configured]
        if configured.is_dir():
            return sorted(
                item
                for item in configured.rglob("*")
                if item.is_file() and item.suffix.lower() == ".enex"
            )
        return []

    def _read_notes(self, root: Path, path: Path) -> list[KnowledgeUnit]:
        try:
            tree = ET.parse(path)
        except (ET.ParseError, OSError):
            return []

        rel_path = path.relative_to(root).as_posix()
        units: list[KnowledgeUnit] = []
        for index, note in enumerate(tree.getroot().findall("note"), start=1):
            unit = self._unit_from_note(note, rel_path, index)
            if unit is not None:
                units.append(unit)
        return units

    def _unit_from_note(
        self, note: ET.Element, source_path: str, index: int
    ) -> KnowledgeUnit | None:
        title = self._child_text(note, "title") or "Untitled"
        guid = self._child_text(note, "guid")
        content = self._content_text(self._child_text(note, "content"))
        created_at = self._parse_enex_datetime(self._child_text(note, "created"))
        updated_at = self._parse_enex_datetime(self._child_text(note, "updated"))
        tags = self._tags(note)

        attributes = note.find("note-attributes")
        metadata = {"source_path": source_path}
        if guid:
            metadata["guid"] = guid
        if attributes is not None:
            for enex_name, metadata_name in (
                ("author", "author"),
                ("source-url", "source_url"),
                ("latitude", "latitude"),
                ("longitude", "longitude"),
                ("altitude", "altitude"),
            ):
                value = self._child_text(attributes, enex_name)
                if value:
                    metadata[metadata_name] = value

        timestamp = updated_at or created_at
        return KnowledgeUnit(
            source_project=SourceProject.ENEX,
            source_id=guid or f"{source_path}:{index}",
            source_entity_type="note",
            title=title,
            content=content,
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=tags,
            created_at=created_at or datetime.now(timezone.utc),
            updated_at=timestamp or datetime.now(timezone.utc),
        )

    def _content_text(self, content: str) -> str:
        if not content:
            return ""
        content = DOCTYPE_RE.sub("", content).strip()
        parser = _EnexContentTextParser()
        parser.feed(content)
        parser.close()
        return parser.text()

    def _tags(self, note: ET.Element) -> list[str]:
        tags: list[str] = []
        seen: set[str] = set()
        for tag in note.findall("tag"):
            value = (tag.text or "").strip()
            if value and value not in seen:
                tags.append(value)
                seen.add(value)
        return tags

    def _child_text(self, parent: ET.Element | None, name: str) -> str:
        if parent is None:
            return ""
        child = parent.find(name)
        if child is None or child.text is None:
            return ""
        return child.text.strip()

    def _parse_enex_datetime(self, value: str) -> datetime | None:
        if not value:
            return None
        normalized = value.strip()
        for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%dT%H%M%S"):
            try:
                return datetime.strptime(normalized, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        try:
            parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _sync_datetime(self, since: SyncState) -> datetime:
        if isinstance(since.last_sync_at, datetime):
            sync_at = since.last_sync_at
        else:
            sync_at = datetime.fromisoformat(str(since.last_sync_at))
        if sync_at.tzinfo is None:
            return sync_at.replace(tzinfo=timezone.utc)
        return sync_at.astimezone(timezone.utc)
