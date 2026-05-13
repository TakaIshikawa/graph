"""Adapter for Zotero CSV exports with bibliographic items."""

from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState

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
        return ["article", "book", "conference_paper", "thesis", "report", "author"]

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

        allowed_types = set(entity_types or self.entity_types)
        include_authors = "author" in allowed_types
        author_items: dict[str, list[KnowledgeUnit]] = {}
        author_names: dict[str, str] = {}

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
            authors = self._parse_authors(author or "")
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
            for author_name in authors:
                author_key = self._author_key(author_name)
                if not author_key:
                    continue
                author_items.setdefault(author_key, []).append(unit)
                author_names.setdefault(author_key, author_name)
            if entity_type in allowed_types:
                result.units.append(unit)

        author_units = [
            self._author_unit(author_key, author_names[author_key], author_items[author_key])
            for author_key in sorted(author_items)
        ]
        if include_authors:
            result.units.extend(author_units)
        if include_authors:
            item_ids = {unit.source_id for unit in result.units if unit.source_entity_type != "author"}
            author_by_key = {unit.metadata["normalized_name"]: unit for unit in author_units}
            for author_key, items in author_items.items():
                author_unit = author_by_key[author_key]
                for item in items:
                    if item.source_id in item_ids:
                        result.edges.append(self._item_author_edge(item, author_unit))

        result.units.sort(key=lambda u: (u.source_entity_type, u.source_id))
        result.edges.sort(key=lambda edge: edge.id)
        return result

    def _parse_authors(self, value: str) -> list[str]:
        authors: list[str] = []
        for raw in value.replace("\n", ";").split(";"):
            author = " ".join(raw.strip().split())
            if author and author.casefold() not in {item.casefold() for item in authors}:
                authors.append(author)
        return authors

    def _author_key(self, name: str) -> str:
        normalized = " ".join(name.casefold().split())
        return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16] if normalized else ""

    def _author_unit(self, author_key: str, name: str, items: list[KnowledgeUnit]) -> KnowledgeUnit:
        item_ids = sorted({item.source_id for item in items})
        item_titles = sorted({item.title for item in items})
        item_types = sorted({item.source_entity_type for item in items})
        return KnowledgeUnit(
            source_project=SourceProject.ZOTERO_CSV,
            source_id=f"zotero_csv:author:{author_key}",
            source_entity_type="author",
            title=name,
            content=f"Zotero author: {name}\nItems: {len(item_ids)}",
            content_type=ContentType.METADATA,
            metadata={
                "name": name,
                "normalized_name": author_key,
                "item_count": len(item_ids),
                "item_source_ids": item_ids,
                "item_titles": item_titles,
                "item_types": item_types,
            },
            tags=["author"],
            created_at=min(item.created_at for item in items),
            updated_at=max(item.updated_at for item in items),
        )

    def _item_author_edge(self, item: KnowledgeUnit, author: KnowledgeUnit) -> KnowledgeEdge:
        digest = hashlib.sha1(f"{item.source_id}|{author.source_id}|relates_to".encode("utf-8")).hexdigest()[:16]
        return KnowledgeEdge(
            id=f"zotero-csv-item-author-{digest}",
            from_unit_id=item.source_id,
            to_unit_id=author.source_id,
            relation=EdgeRelation.RELATES_TO,
            source=EdgeSource.SOURCE,
            metadata={
                "source_project": SourceProject.ZOTERO_CSV.value,
                "from_entity_type": item.source_entity_type,
                "to_entity_type": "author",
                "author": author.title,
            },
            created_at=item.created_at,
        )
