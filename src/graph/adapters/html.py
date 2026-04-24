"""Adapter for local HTML documents."""

from __future__ import annotations

from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "br",
    "dd",
    "div",
    "dl",
    "dt",
    "figcaption",
    "figure",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "td",
    "th",
    "tr",
    "ul",
}
SKIP_CONTENT_TAGS = {"script", "style"}


class _ReadableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.h1_parts: list[str] = []
        self.text_parts: list[str] = []
        self.description = ""
        self.canonical_url = ""
        self.keywords: list[str] = []
        self._tag_stack: list[str] = []
        self._skip_depth = 0
        self._in_title = False
        self._in_first_h1 = False
        self._captured_h1 = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "body" and "head" in self._tag_stack:
            self._tag_stack = [item for item in self._tag_stack if item != "head"]
        self._tag_stack.append(tag)
        attrs_dict = {name.lower(): value or "" for name, value in attrs}

        if tag in SKIP_CONTENT_TAGS:
            self._skip_depth += 1
            return

        if tag == "title":
            self._in_title = True
        elif tag == "h1" and not self._captured_h1:
            self._in_first_h1 = True
        elif tag == "meta":
            self._handle_meta(attrs_dict)
        elif tag == "link":
            self._handle_link(attrs_dict)

        if tag in BLOCK_TAGS:
            self._append_separator()

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if self._tag_stack:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in SKIP_CONTENT_TAGS and self._skip_depth:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False
        elif tag == "h1" and self._in_first_h1:
            self._in_first_h1 = False
            self._captured_h1 = True

        if tag in BLOCK_TAGS:
            self._append_separator()

        for index in range(len(self._tag_stack) - 1, -1, -1):
            if self._tag_stack[index] == tag:
                del self._tag_stack[index:]
                break

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = unescape(data).strip()
        if not text:
            return

        if self._in_title:
            self.title_parts.append(text)
            return

        if self._in_first_h1:
            self.h1_parts.append(text)

        if "head" not in self._tag_stack:
            self.text_parts.append(text)

    def _handle_meta(self, attrs: dict[str, str]) -> None:
        name = attrs.get("name", "").strip().lower()
        property_name = attrs.get("property", "").strip().lower()
        content = attrs.get("content", "").strip()
        if not content:
            return
        if name == "description" or property_name == "og:description":
            if not self.description:
                self.description = unescape(content)
        elif name == "keywords":
            for keyword in content.split(","):
                tag = keyword.strip()
                if tag and tag not in self.keywords:
                    self.keywords.append(tag)

    def _handle_link(self, attrs: dict[str, str]) -> None:
        rels = {rel.strip().lower() for rel in attrs.get("rel", "").split()}
        href = attrs.get("href", "").strip()
        if "canonical" in rels and href and not self.canonical_url:
            self.canonical_url = unescape(href)

    def _append_separator(self) -> None:
        if self.text_parts and self.text_parts[-1] != "\n":
            self.text_parts.append("\n")

    def title(self) -> str:
        return self._normalize_text(" ".join(self.title_parts))

    def first_h1(self) -> str:
        return self._normalize_text(" ".join(self.h1_parts))

    def readable_text(self) -> str:
        lines: list[str] = []
        current: list[str] = []
        for part in self.text_parts:
            if part == "\n":
                if current:
                    lines.append(self._normalize_text(" ".join(current)))
                    current = []
            else:
                current.append(part)
        if current:
            lines.append(self._normalize_text(" ".join(current)))
        return "\n".join(line for line in lines if line)

    def _normalize_text(self, text: str) -> str:
        return " ".join(text.split())


class HtmlAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "html"

    @property
    def entity_types(self) -> list[str]:
        return ["html_document"]

    def __init__(self, root_path: str = "") -> None:
        self.root_path = root_path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "html_document" not in entity_types:
            return result

        root = Path(self.root_path).expanduser()
        if not root.exists() or not root.is_dir():
            return result

        sync_at = self._sync_timestamp(since) if since else None
        for path in sorted(
            item for item in root.rglob("*") if item.is_file() and item.suffix.lower() in {".html", ".htm"}
        ):
            stat = path.stat()
            if sync_at is not None and stat.st_mtime <= sync_at:
                continue

            unit = self._read_html(root, path, stat.st_size, stat.st_ctime)
            if unit is not None:
                result.units.append(unit)

        return result

    def _read_html(
        self,
        root: Path,
        path: Path,
        file_size: int,
        created_timestamp: float,
    ) -> KnowledgeUnit | None:
        try:
            html = path.read_text(encoding="utf-8")
            parser = _ReadableHTMLParser()
            parser.feed(html)
            parser.close()
        except (OSError, UnicodeDecodeError, Exception):
            return None

        source_id = path.relative_to(root).as_posix()
        title = parser.title() or parser.first_h1() or path.stem
        metadata = {
            "path": source_id,
            "file_size": file_size,
        }
        if parser.description:
            metadata["description"] = parser.description
        if parser.canonical_url:
            metadata["canonical_url"] = parser.canonical_url

        return KnowledgeUnit(
            source_project=SourceProject.ME,
            source_id=source_id,
            source_entity_type="html_document",
            title=title,
            content=parser.readable_text(),
            content_type=ContentType.INSIGHT,
            metadata=metadata,
            tags=parser.keywords,
            created_at=datetime.fromtimestamp(created_timestamp, tz=timezone.utc),
        )

    def _sync_timestamp(self, since: SyncState) -> float:
        if isinstance(since.last_sync_at, datetime):
            return since.last_sync_at.timestamp()
        return datetime.fromisoformat(str(since.last_sync_at)).timestamp()
