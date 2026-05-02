"""Adapter for outbound links embedded in Markdown files."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


INLINE_LINK_RE = re.compile(r"(?<!!)\[([^\]\n]+)\]\(\s*([^)\s]+)(?:\s+\"[^\"]*\")?\s*\)")
REFERENCE_LINK_RE = re.compile(r"(?<!!)\[([^\]\n]+)\]\[([^\]\n]*)\]")
SHORTCUT_REFERENCE_LINK_RE = re.compile(r"(?<!!)\[([^\]\n]+)\](?![\[(])")
REFERENCE_DEFINITION_RE = re.compile(r"^\s{0,3}\[([^\]]+)\]:\s*(\S+)(?:\s+.*)?$")


@dataclass(frozen=True)
class _MarkdownLink:
    url: str
    file_path: str
    line_number: int
    link_text: str
    line: str


class MarkdownLinksAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "markdown_links"

    @property
    def entity_types(self) -> list[str]:
        return ["markdown_link"]

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
        if entity_types and "markdown_link" not in entity_types:
            return result

        root = Path(self.path).expanduser()
        if not root.exists():
            return result

        files = self._markdown_files(root)
        source_root = Path(self.source_id_root).expanduser() if self.source_id_root else root
        if root.is_file() and not self.source_id_root:
            source_root = root.parent

        sync_at = self._sync_datetime(since) if since else None
        seen: set[str] = set()
        for file_path in files:
            stat = file_path.stat()
            updated_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            if sync_at and updated_at <= sync_at:
                continue

            relative_path = self._relative_path(file_path, source_root)
            for link in self._extract_links(file_path, relative_path):
                source_id = self._source_id(link.file_path, link.url)
                if source_id in seen:
                    continue
                seen.add(source_id)
                result.units.append(
                    KnowledgeUnit(
                        source_project=SourceProject.MARKDOWN_LINKS,
                        source_id=source_id,
                        source_entity_type="markdown_link",
                        title=link.link_text or self._hostname(link.url) or link.url,
                        content=self._content(link),
                        content_type=ContentType.ARTIFACT,
                        metadata={
                            "url": link.url,
                            "file_path": link.file_path,
                            "line_number": link.line_number,
                            "link_text": link.link_text,
                        },
                        tags=["markdown-link"],
                        created_at=updated_at,
                        updated_at=updated_at,
                    )
                )

        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _markdown_files(self, root: Path) -> list[Path]:
        if root.is_file():
            return [root] if root.suffix.lower() == ".md" else []
        if not root.is_dir():
            return []
        return sorted(path for path in root.rglob("*.md") if path.is_file())

    def _extract_links(self, path: Path, relative_path: str) -> list[_MarkdownLink]:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        definitions = self._reference_definitions(lines)
        links: list[_MarkdownLink] = []

        for line_number, line in enumerate(lines, start=1):
            if REFERENCE_DEFINITION_RE.match(line):
                label, url = self._definition_parts(line)
                if label and url:
                    links.append(
                        _MarkdownLink(
                            url=url,
                            file_path=relative_path,
                            line_number=line_number,
                            link_text=label,
                            line=line.strip(),
                        )
                    )
                continue

            for match in INLINE_LINK_RE.finditer(line):
                link_text = self._clean_text(match.group(1))
                url = self._clean_url(match.group(2))
                if url:
                    links.append(
                        _MarkdownLink(
                            url=url,
                            file_path=relative_path,
                            line_number=line_number,
                            link_text=link_text,
                            line=line.strip(),
                        )
                    )

            for match in REFERENCE_LINK_RE.finditer(line):
                link_text = self._clean_text(match.group(1))
                label = self._normalize_label(match.group(2) or link_text)
                url = definitions.get(label, "")
                if url:
                    links.append(
                        _MarkdownLink(
                            url=url,
                            file_path=relative_path,
                            line_number=line_number,
                            link_text=link_text,
                            line=line.strip(),
                        )
                    )

            for match in SHORTCUT_REFERENCE_LINK_RE.finditer(line):
                link_text = self._clean_text(match.group(1))
                url = definitions.get(self._normalize_label(link_text), "")
                if url:
                    links.append(
                        _MarkdownLink(
                            url=url,
                            file_path=relative_path,
                            line_number=line_number,
                            link_text=link_text,
                            line=line.strip(),
                        )
                    )

        return links

    def _reference_definitions(self, lines: list[str]) -> dict[str, str]:
        definitions: dict[str, str] = {}
        for line in lines:
            label, url = self._definition_parts(line)
            if label and url:
                definitions[self._normalize_label(label)] = url
        return definitions

    def _definition_parts(self, line: str) -> tuple[str, str]:
        match = REFERENCE_DEFINITION_RE.match(line)
        if not match:
            return "", ""
        return self._clean_text(match.group(1)), self._clean_url(match.group(2))

    def _source_id(self, file_path: str, url: str) -> str:
        digest = hashlib.sha256(f"{file_path}\0{url}".encode("utf-8")).hexdigest()[:16]
        return f"markdown_links:{digest}"

    def _content(self, link: _MarkdownLink) -> str:
        return "\n".join(
            [
                f"URL: {link.url}",
                f"File: {link.file_path}:{link.line_number}",
                f"Context: {link.line}",
            ]
        )

    def _hostname(self, url: str) -> str:
        parsed = urlparse(url)
        return parsed.netloc or parsed.path.split("/", 1)[0]

    def _relative_path(self, path: Path, source_root: Path) -> str:
        try:
            return path.relative_to(source_root).as_posix()
        except ValueError:
            return path.as_posix()

    def _clean_text(self, value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()

    def _clean_url(self, value: str) -> str:
        return value.strip().strip("<>").rstrip(".,;")

    def _normalize_label(self, value: str) -> str:
        return self._clean_text(value).casefold()

    def _sync_datetime(self, since: SyncState) -> datetime:
        value = since.last_sync_at
        if isinstance(value, datetime):
            parsed = value
        else:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
