"""Adapter for Obsidian note outgoing backlinks."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

from graph.adapters._personal_exports import clean_metadata, ensure_utc
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


WIKI_RE = re.compile(r"!?\[\[([^\]\n]+)\]\]")
MD_LINK_RE = re.compile(r"(?<!!)\[([^\]\n]+)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


class ObsidianBacklinksMarkdownAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "obsidian_backlinks_markdown"

    @property
    def entity_types(self) -> list[str]:
        return ["backlink_index"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "backlink_index" not in entity_types:
            return result
        root = Path(self.path).expanduser()
        if not root.exists():
            return result
        files = [root] if root.is_file() and root.suffix.lower() == ".md" else sorted(path for path in root.rglob("*.md") if path.is_file())
        base = root.parent if root.is_file() else root
        sync_at = ensure_utc(since.last_sync_at) if since else None
        for path in files:
            updated = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
            if sync_at and updated <= sync_at:
                continue
            unit = self._unit(path, base, updated)
            if unit:
                result.units.append(unit)
        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _unit(self, path: Path, base: Path, updated: datetime) -> KnowledgeUnit | None:
        text = path.read_text(encoding="utf-8", errors="replace")
        outgoing, unresolved = _links(text)
        if not outgoing:
            return None
        try:
            relative = path.relative_to(base).as_posix()
        except ValueError:
            relative = path.as_posix()
        title = _title(text) or path.stem
        metadata = clean_metadata({"title": title, "path": relative, "outgoing_links": outgoing, "unresolved_link_texts": unresolved, "link_count": len(outgoing)})
        digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:16]
        return KnowledgeUnit(source_project=self.name, source_id=f"{self.name}:{digest}", source_entity_type="backlink_index", title=title, content="\n".join([title, *[f"- {link}" for link in outgoing]]), content_type=ContentType.ARTIFACT, metadata=metadata, tags=["obsidian", "backlinks"], created_at=updated, updated_at=updated)


def _links(text: str) -> tuple[list[str], list[str]]:
    links: list[str] = []
    unresolved: list[str] = []
    for match in WIKI_RE.finditer(text):
        raw = match.group(1).strip()
        target = raw.split("|", 1)[0].split("#", 1)[0].strip()
        if target and target not in links:
            links.append(target)
        if raw and raw not in unresolved:
            unresolved.append(raw)
    for match in MD_LINK_RE.finditer(text):
        href = match.group(2).strip().strip("<>").rstrip(".,;")
        if href and href not in links:
            links.append(href)
    return links, unresolved


def _title(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return ""
