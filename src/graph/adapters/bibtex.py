"""Adapter for local BibTeX reference files."""

from __future__ import annotations

import re
import warnings
from datetime import datetime, timezone
from pathlib import Path

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


SKIPPED_ENTRY_TYPES = {"comment", "preamble", "string"}
VENUE_FIELDS = ("journal", "booktitle", "publisher", "school", "institution", "organization")
CONTENT_FIELDS = ("abstract", "note", "annote", "author", "year")


class BibtexParseError(ValueError):
    """Raised when a BibTeX entry cannot be parsed."""


class BibtexAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "bibtex"

    @property
    def entity_types(self) -> list[str]:
        return ["bibtex_entry"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "bibtex_entry" not in entity_types:
            return result

        paths = self._discover_paths()
        if not paths:
            return result

        sync_at = self._sync_timestamp(since) if since else None
        root = Path(self.path).expanduser()
        root = root if root.is_dir() else root.parent
        malformed_entries = 0

        for path in paths:
            stat = path.stat()
            if sync_at is not None and stat.st_mtime <= sync_at:
                continue

            try:
                text = path.read_text(encoding="utf-8-sig")
            except (OSError, UnicodeDecodeError):
                malformed_entries += 1
                continue

            entries, malformed = self._parse_entries(text)
            malformed_entries += malformed
            for entry in entries:
                result.units.append(
                    self._unit_from_entry(
                        root,
                        path,
                        entry,
                        created_timestamp=stat.st_ctime,
                    )
                )

        if malformed_entries:
            suffix = "y" if malformed_entries == 1 else "ies"
            warnings.warn(
                f"Skipped {malformed_entries} malformed BibTeX entr{suffix}.",
                stacklevel=2,
            )

        return result

    def _discover_paths(self) -> list[Path]:
        configured = Path(self.path).expanduser()
        if configured.is_file() and configured.suffix.lower() == ".bib":
            return [configured]
        if configured.is_dir():
            return sorted(
                item
                for item in configured.rglob("*")
                if item.is_file() and item.suffix.lower() == ".bib"
            )
        return []

    def _parse_entries(self, text: str) -> tuple[list[dict[str, object]], int]:
        entries: list[dict[str, object]] = []
        malformed = 0
        index = 0
        while True:
            at_index = text.find("@", index)
            if at_index == -1:
                break

            try:
                entry, index = self._parse_entry_at(text, at_index)
            except BibtexParseError:
                malformed += 1
                next_entry = text.find("@", at_index + 1)
                index = next_entry if next_entry != -1 else len(text)
                continue

            if entry is not None:
                entries.append(entry)

        return entries, malformed

    def _parse_entry_at(
        self,
        text: str,
        at_index: int,
    ) -> tuple[dict[str, object] | None, int]:
        cursor = at_index + 1
        type_start = cursor
        while cursor < len(text) and (text[cursor].isalpha() or text[cursor] in "_-"):
            cursor += 1
        entry_type = text[type_start:cursor].strip().lower()
        if not entry_type:
            raise BibtexParseError("missing entry type")

        cursor = self._skip_space(text, cursor)
        if cursor >= len(text) or text[cursor] not in "{(":
            raise BibtexParseError("missing entry body")

        opener = text[cursor]
        closer = "}" if opener == "{" else ")"
        body_start = cursor + 1
        body_end = self._find_matching(text, cursor, opener, closer)
        next_index = body_end + 1
        if entry_type in SKIPPED_ENTRY_TYPES:
            return None, next_index

        body = text[body_start:body_end]
        comma = self._find_top_level_comma(body)
        if comma == -1:
            raise BibtexParseError("missing citation key")

        citation_key = body[:comma].strip()
        if not citation_key:
            raise BibtexParseError("empty citation key")

        fields = self._parse_fields(body[comma + 1 :])
        return {
            "entry_type": entry_type,
            "citation_key": citation_key,
            "fields": fields,
        }, next_index

    def _parse_fields(self, body: str) -> dict[str, str]:
        fields: dict[str, str] = {}
        cursor = 0
        while cursor < len(body):
            cursor = self._skip_space_and_commas(body, cursor)
            if cursor >= len(body):
                break

            name_start = cursor
            while cursor < len(body) and (body[cursor].isalnum() or body[cursor] in "_-"):
                cursor += 1
            name = body[name_start:cursor].strip().lower()
            cursor = self._skip_space(body, cursor)
            if not name or cursor >= len(body) or body[cursor] != "=":
                raise BibtexParseError("malformed field")

            cursor = self._skip_space(body, cursor + 1)
            value, cursor = self._parse_value(body, cursor)
            fields[name] = self._clean_value(value)
            cursor = self._skip_space(body, cursor)
            if cursor < len(body) and body[cursor] == ",":
                cursor += 1

        return fields

    def _parse_value(self, text: str, cursor: int) -> tuple[str, int]:
        if cursor >= len(text):
            raise BibtexParseError("missing field value")

        if text[cursor] == "{":
            end = self._find_matching(text, cursor, "{", "}")
            return text[cursor + 1 : end], end + 1

        if text[cursor] == '"':
            cursor += 1
            start = cursor
            escaped = False
            while cursor < len(text):
                char = text[cursor]
                if char == '"' and not escaped:
                    return text[start:cursor], cursor + 1
                escaped = char == "\\" and not escaped
                if char != "\\":
                    escaped = False
                cursor += 1
            raise BibtexParseError("unterminated quoted value")

        start = cursor
        while cursor < len(text) and text[cursor] != ",":
            cursor += 1
        value = text[start:cursor].strip()
        if not value:
            raise BibtexParseError("empty field value")
        return value, cursor

    def _unit_from_entry(
        self,
        root: Path,
        path: Path,
        entry: dict[str, object],
        *,
        created_timestamp: float,
    ) -> KnowledgeUnit:
        fields = entry["fields"]
        assert isinstance(fields, dict)
        citation_key = str(entry["citation_key"])
        source_file = path.relative_to(root).as_posix()
        title = self._field(fields, "title") or self._field(fields, "booktitle") or citation_key
        authors = self._authors(self._field(fields, "author"))

        metadata = {
            "citation_key": citation_key,
            "entry_type": str(entry["entry_type"]),
            "authors": authors,
            "year": self._field(fields, "year"),
            "doi": self._field(fields, "doi"),
            "url": self._field(fields, "url"),
            "journal": self._field(fields, "journal"),
            "booktitle": self._field(fields, "booktitle"),
            "source_file": source_file,
        }

        return KnowledgeUnit(
            source_project=SourceProject.BIBTEX,
            source_id=f"{source_file}:{citation_key}",
            source_entity_type="bibtex_entry",
            title=title,
            content=self._content(fields),
            content_type=ContentType.INSIGHT,
            metadata=metadata,
            tags=self._tags(self._field(fields, "keywords")),
            created_at=datetime.fromtimestamp(created_timestamp, tz=timezone.utc),
        )

    def _content(self, fields: dict[str, str]) -> str:
        parts: list[str] = []
        author = self._field(fields, "author")
        year = self._field(fields, "year")
        venue = self._venue(fields)
        if author:
            parts.append(f"Authors: {author}")
        if year:
            parts.append(f"Year: {year}")
        if venue:
            parts.append(f"Venue: {venue}")
        for key in CONTENT_FIELDS:
            if key in {"author", "year"}:
                continue
            value = self._field(fields, key)
            if value:
                label = "Notes" if key in {"note", "annote"} else key.title()
                parts.append(f"{label}: {value}")
        return "\n\n".join(parts)

    def _venue(self, fields: dict[str, str]) -> str:
        for key in VENUE_FIELDS:
            value = self._field(fields, key)
            if value:
                return value
        return ""

    def _authors(self, value: str) -> list[str]:
        return [author.strip() for author in re.split(r"\s+\band\b\s+", value) if author.strip()]

    def _tags(self, value: str) -> list[str]:
        tags: list[str] = []
        for raw_tag in re.split(r"[,;]", value):
            tag = raw_tag.strip().removeprefix("#").strip()
            if tag and tag not in tags:
                tags.append(tag)
        return tags

    def _field(self, fields: dict[str, str], key: str) -> str:
        return fields.get(key, "").strip()

    def _clean_value(self, value: str) -> str:
        cleaned = value.replace("\n", " ")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = cleaned.replace(r"\&", "&")
        return re.sub(r"[{}]", "", cleaned).strip()

    def _find_matching(self, text: str, cursor: int, opener: str, closer: str) -> int:
        depth = 0
        in_quote = False
        escaped = False
        while cursor < len(text):
            char = text[cursor]
            if char == '"' and not escaped:
                in_quote = not in_quote
            elif not in_quote:
                if char == opener:
                    depth += 1
                elif char == closer:
                    depth -= 1
                    if depth == 0:
                        return cursor
            escaped = char == "\\" and not escaped
            if char != "\\":
                escaped = False
            cursor += 1
        raise BibtexParseError("unterminated balanced value")

    def _find_top_level_comma(self, text: str) -> int:
        depth = 0
        in_quote = False
        escaped = False
        for index, char in enumerate(text):
            if char == '"' and not escaped:
                in_quote = not in_quote
            elif not in_quote:
                if char in "{(":
                    depth += 1
                elif char in "})" and depth:
                    depth -= 1
                elif char == "," and depth == 0:
                    return index
            escaped = char == "\\" and not escaped
            if char != "\\":
                escaped = False
        return -1

    def _skip_space(self, text: str, cursor: int) -> int:
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        return cursor

    def _skip_space_and_commas(self, text: str, cursor: int) -> int:
        while cursor < len(text) and (text[cursor].isspace() or text[cursor] == ","):
            cursor += 1
        return cursor

    def _sync_timestamp(self, since: SyncState) -> float:
        if isinstance(since.last_sync_at, datetime):
            return since.last_sync_at.timestamp()
        return datetime.fromisoformat(str(since.last_sync_at)).timestamp()
