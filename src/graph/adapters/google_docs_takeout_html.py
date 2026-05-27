"""Adapter for Google Docs Takeout HTML exports."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, iter_paths
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class _Extractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.skip = 0
        self.title = ""
        self.current_heading = ""
        self.headings: list[str] = []
        self.links: list[dict[str, str]] = []
        self.parts: list[str] = []
        self.href = ""
        self.in_title = False
        self.heading_tag = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {key: value or "" for key, value in attrs}
        if tag in {"script", "style"}:
            self.skip += 1
        if tag == "title":
            self.in_title = True
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self.heading_tag = tag
            self.current_heading = ""
        if tag == "a":
            self.href = attrs_dict.get("href", "")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self.skip:
            self.skip -= 1
        if tag == "title":
            self.in_title = False
        if tag == self.heading_tag:
            heading = self.current_heading.strip()
            if heading:
                self.headings.append(heading)
            self.heading_tag = ""
        if tag == "a":
            self.href = ""
        if tag in {"p", "div", "li", "br", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.skip:
            return
        text = data.strip()
        if not text:
            return
        if self.in_title:
            self.title += text
        if self.heading_tag:
            self.current_heading += f" {text}"
        if self.href:
            self.links.append({"text": text, "url": self.href})
        self.parts.append(text)


class GoogleDocsTakeoutHtmlAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "google_docs_takeout_html"

    @property
    def entity_types(self) -> list[str]:
        return ["document"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if "document" not in set(entity_types or self.entity_types):
            return result
        for path in iter_paths(self.path, {".html", ".htm"}):
            unit = self._unit(path)
            if unit:
                result.units.append(unit)
        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _unit(self, path: Path) -> KnowledgeUnit | None:
        try:
            html = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError):
            return None
        parser = _Extractor()
        parser.feed(html)
        text = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", "\n".join(parser.parts))).strip()
        title = parser.title.strip() or (parser.headings[0] if parser.headings else path.stem)
        metadata = clean_metadata({"source_path": str(path), "links": parser.links, "headings": parser.headings, "file_name": path.name, "file_size": path.stat().st_size})
        return KnowledgeUnit(source_project=self.name, source_id=digest_source_id(self.name, path), source_entity_type="document", title=title, content=text, content_type=ContentType.ARTIFACT, metadata=metadata, tags=["google_docs", "document"])
