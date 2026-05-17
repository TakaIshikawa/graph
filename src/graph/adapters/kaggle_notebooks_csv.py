"""Adapter for Kaggle notebooks CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import (
    clean_metadata,
    digest_source_id,
    ensure_utc,
    first,
    iter_paths,
    parse_datetime,
    parse_int,
    read_csv_rows,
    split_values,
)
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class KaggleNotebooksCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "kaggle_notebooks_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["notebook", "dataset", "author", "competition"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        allowed_types = set(entity_types) if entity_types is not None else set(self.entity_types)
        if not allowed_types.intersection(self.entity_types):
            return result

        sync_at = ensure_utc(since.last_sync_at) if since else None
        notebooks: list[KnowledgeUnit] = []
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._notebook_unit(row, path.name, index)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                notebooks.append(unit)

        datasets = self._aggregate_units(notebooks, "dataset")
        authors = self._aggregate_units(notebooks, "author")
        competitions = self._aggregate_units(notebooks, "competition")
        if "notebook" in allowed_types:
            result.units.extend(notebooks)
        if "dataset" in allowed_types:
            result.units.extend(datasets)
        if "author" in allowed_types:
            result.units.extend(authors)
        if "competition" in allowed_types:
            result.units.extend(competitions)
        if {"dataset", "notebook"}.issubset(allowed_types):
            result.edges.extend(self._aggregate_edges(datasets, notebooks, "dataset", EdgeRelation.RELATES_TO))
        if {"author", "notebook"}.issubset(allowed_types):
            result.edges.extend(self._aggregate_edges(authors, notebooks, "author", EdgeRelation.CONTAINS))
        if {"competition", "notebook"}.issubset(allowed_types):
            result.edges.extend(self._aggregate_edges(competitions, notebooks, "competition", EdgeRelation.RELATES_TO))

        result.units.sort(key=lambda unit: (unit.source_entity_type, unit.source_id))
        result.edges.sort(key=lambda edge: edge.id)
        return result

    def _notebook_unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        title = first(row, "Title", "Notebook Title", "Name")
        url = first(row, "Notebook URL", "URL", "Url", "Link")
        author = first(row, "Author", "Author Name", "Owner", "User")
        dataset = first(row, "Dataset", "Datasets", "Data")
        competition = first(row, "Competition", "Challenge")
        last_run_at = parse_datetime(first(row, "Last Run Time", "Last Run", "Updated At"))
        created_at = parse_datetime(first(row, "Created Date", "Created At", "Date"))
        tags = split_values(first(row, "Tags", "Tag"))
        metadata = clean_metadata(
            {
                "notebook_id": first(row, "Notebook Id", "Notebook ID", "Kernel Id", "ID", "Id"),
                "title": title,
                "notebook_url": url,
                "author": author,
                "dataset": dataset,
                "competition": competition,
                "language": first(row, "Language", "Script Language"),
                "votes": parse_int(first(row, "Votes", "Vote Count")),
                "views": parse_int(first(row, "Views", "View Count")),
                "comments": parse_int(first(row, "Comments", "Comment Count")),
                "last_run_at": last_run_at.isoformat() if last_run_at else "",
                "created_at": created_at.isoformat() if created_at else "",
                "tags": tags,
                "description": first(row, "Description", "Summary"),
                "slug": first(row, "Slug", "Notebook Slug"),
                "source_file": source_file,
            }
        )
        if not any([title, url, author, dataset, competition, metadata.get("notebook_id"), metadata.get("slug")]):
            return None

        now = datetime.now(timezone.utc)
        unit_created_at = created_at or last_run_at or now
        updated_at = last_run_at or unit_created_at
        display_title = title or metadata.get("slug") or url or "Kaggle notebook"
        return KnowledgeUnit(
            source_project="kaggle_notebooks_csv",
            source_id=self._notebook_source_id(metadata, url, title, author, source_file, index),
            source_entity_type="notebook",
            title=str(display_title),
            content=self._notebook_content(str(display_title), metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["kaggle", "notebook", author, dataset, competition, *tags] if tag)),
            created_at=unit_created_at,
            updated_at=updated_at,
        )

    def _notebook_source_id(
        self,
        metadata: dict[str, Any],
        url: str,
        title: str,
        author: str,
        source_file: str,
        index: int,
    ) -> str:
        if metadata.get("notebook_id"):
            return digest_source_id("kaggle_notebooks_csv", metadata["notebook_id"])
        if url:
            return digest_source_id("kaggle_notebooks_csv", url)
        if metadata.get("slug") or title or author:
            return digest_source_id("kaggle_notebooks_csv", metadata.get("slug", ""), title, author)
        return digest_source_id("kaggle_notebooks_csv", source_file, index)

    def _aggregate_units(self, notebooks: list[KnowledgeUnit], entity_type: str) -> list[KnowledgeUnit]:
        grouped: dict[str, list[KnowledgeUnit]] = {}
        names: dict[str, str] = {}
        for notebook in notebooks:
            value = str(notebook.metadata.get(entity_type) or "").strip()
            if not value:
                continue
            key = value.casefold()
            names.setdefault(key, value)
            grouped.setdefault(key, []).append(notebook)

        units: list[KnowledgeUnit] = []
        for key, group in sorted(grouped.items()):
            name = names[key]
            metadata = self._aggregate_metadata(group, {entity_type: name})
            units.append(
                KnowledgeUnit(
                    source_project="kaggle_notebooks_csv",
                    source_id=digest_source_id(f"kaggle_notebooks_csv_{entity_type}", key),
                    source_entity_type=entity_type,
                    title=name,
                    content=f"Kaggle {entity_type}: {name}\nNotebooks: {len(group)}",
                    content_type=ContentType.METADATA,
                    metadata=metadata,
                    tags=["kaggle", entity_type, name],
                    created_at=min(notebook.created_at for notebook in group),
                    updated_at=max(notebook.updated_at for notebook in group),
                )
            )
        return units

    def _aggregate_metadata(self, notebooks: list[KnowledgeUnit], base: dict[str, Any]) -> dict[str, Any]:
        votes = [value for notebook in notebooks if (value := notebook.metadata.get("votes")) is not None]
        views = [value for notebook in notebooks if (value := notebook.metadata.get("views")) is not None]
        comments = [value for notebook in notebooks if (value := notebook.metadata.get("comments")) is not None]
        metadata = {
            **base,
            "notebook_count": len(notebooks),
            "total_votes": sum(votes) if votes else None,
            "total_views": sum(views) if views else None,
            "total_comments": sum(comments) if comments else None,
            "authors": sorted({str(notebook.metadata.get("author")) for notebook in notebooks if notebook.metadata.get("author")}),
            "datasets": sorted({str(notebook.metadata.get("dataset")) for notebook in notebooks if notebook.metadata.get("dataset")}),
            "competitions": sorted({str(notebook.metadata.get("competition")) for notebook in notebooks if notebook.metadata.get("competition")}),
            "languages": sorted({str(notebook.metadata.get("language")) for notebook in notebooks if notebook.metadata.get("language")}),
            "tags": sorted({tag for notebook in notebooks for tag in notebook.metadata.get("tags", [])}),
            "notebook_source_ids": sorted(notebook.source_id for notebook in notebooks),
            "first_created_at": min(notebook.created_at for notebook in notebooks).isoformat(),
            "last_run_at": max(notebook.updated_at for notebook in notebooks).isoformat(),
        }
        return clean_metadata(metadata)

    def _aggregate_edges(
        self,
        aggregate_units: list[KnowledgeUnit],
        notebooks: list[KnowledgeUnit],
        entity_type: str,
        relation: EdgeRelation,
    ) -> list[KnowledgeEdge]:
        aggregate_ids = {str(unit.metadata.get(entity_type) or "").casefold(): unit.source_id for unit in aggregate_units}
        edges: list[KnowledgeEdge] = []
        for notebook in notebooks:
            aggregate_id = aggregate_ids.get(str(notebook.metadata.get(entity_type) or "").casefold())
            if not aggregate_id:
                continue
            edges.append(
                KnowledgeEdge(
                    id=digest_source_id("kaggle_notebooks_csv_edge", entity_type, aggregate_id, notebook.source_id),
                    from_unit_id=aggregate_id,
                    to_unit_id=notebook.source_id,
                    relation=relation,
                    source=EdgeSource.SOURCE,
                    metadata={"relation_type": f"{entity_type}_notebook", entity_type: notebook.metadata.get(entity_type)},
                )
            )
        return edges

    def _notebook_content(self, title: str, metadata: dict[str, Any]) -> str:
        parts = [title]
        for key, label in (
            ("notebook_url", "URL"),
            ("author", "Author"),
            ("dataset", "Dataset"),
            ("competition", "Competition"),
            ("language", "Language"),
            ("votes", "Votes"),
            ("views", "Views"),
            ("comments", "Comments"),
            ("last_run_at", "Last run"),
            ("created_at", "Created"),
            ("tags", "Tags"),
            ("description", "Description"),
        ):
            if metadata.get(key) not in ("", None, []):
                parts.append(f"{label}: {metadata[key]}")
        return "\n".join(parts)
