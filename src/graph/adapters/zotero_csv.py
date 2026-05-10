"""Adapter for Zotero CSV exports with bibliographic items."""

from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState

# Map Zotero item types to normalized entity types
_ITEM_TYPE_MAP = {
    "journalarticle": "article",
    "journal article": "article",
    "article": "article",
    "book": "book",
    "booksection": "book",
    "book section": "book",
    "conferencepaper": "conference_paper",
    "conference paper": "conference_paper",
    "thesis": "thesis",
    "report": "report",
    "webpage": "article",
    "document": "article",
    "preprint": "article",
    "manuscript": "article",
    "presentation": "report",
}


class ZoteroCsvAdapter(SourceAdapter):
    """Import Zotero CSV exports preserving bibliographic metadata."""

    @property
    def name(self) -> str:
        return "zotero_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["article", "book", "conference_paper", "thesis", "report"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if not self.path:
            return result

        csv_path = Path(self.path).expanduser()
        if not csv_path.exists():
            return result

        allowed_types = set(entity_types) if entity_types else None

        try:
            text = csv_path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError):
            return result

        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            title = (row.get("Title") or "").strip()
            if not title:
                continue

            # Map item type
            raw_type = (row.get("Item Type") or "article").strip().lower()
            entity_type = _ITEM_TYPE_MAP.get(raw_type, "article")

            if allowed_types and entity_type not in allowed_types:
                continue

            # Parse tags from Manual Tags and Automatic Tags (semicolon-separated)
            tags: list[str] = []
            for col in ("Manual Tags", "Automatic Tags"):
                raw_tags = (row.get(col) or "").strip()
                if raw_tags:
                    for t in raw_tags.split(";"):
                        t = t.strip().lower()
                        if t and t not in tags:
                            tags.append(t)

            # Build metadata
            key = (row.get("Key") or "").strip()
            author = (row.get("Author") or "").strip() or None
            pub_title = (row.get("Publication Title") or "").strip() or None
            pub_year = (row.get("Publication Year") or "").strip() or None
            doi = (row.get("DOI") or "").strip() or None
            isbn = (row.get("ISBN") or "").strip() or None
            url = (row.get("Url") or row.get("URL") or "").strip() or None
            date_val = (row.get("Date") or "").strip() or None
            date_added = (row.get("Date Added") or "").strip() or None
            date_modified = (row.get("Date Modified") or "").strip() or None
            abstract = (row.get("Abstract Note") or "").strip() or None

            # Deterministic source ID
            id_input = key or f"{title}|{author or ''}"
            digest = hashlib.sha1(id_input.encode("utf-8")).hexdigest()[:16]
            source_id = f"zotero_csv:{entity_type}:{digest}"

            metadata: dict = {}
            if key:
                metadata["key"] = key
            if author:
                metadata["author"] = author
            if pub_title:
                metadata["publication_title"] = pub_title
            if pub_year:
                metadata["publication_year"] = pub_year
            if doi:
                metadata["doi"] = doi
            if isbn:
                metadata["isbn"] = isbn
            if url:
                metadata["url"] = url
            if date_val:
                metadata["date"] = date_val
            if date_added:
                metadata["date_added"] = date_added
            if date_modified:
                metadata["date_modified"] = date_modified
            metadata["item_type"] = raw_type

            unit = KnowledgeUnit(
                source_project=SourceProject.ZOTERO_CSV,
                source_id=source_id,
                source_entity_type=entity_type,
                title=title,
                content=abstract or title,
                content_type=ContentType.ARTIFACT,
                metadata=metadata,
                tags=sorted(tags),
            )
            result.units.append(unit)

        result.units.sort(key=lambda u: (u.source_entity_type, u.source_id))
        return result
