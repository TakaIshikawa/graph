from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO

from graph.export import export_unit_citation_inventory_csv
from graph.types.enums import SourceProject
from graph.types.models import KnowledgeUnit

UNIT_TIME = datetime(2026, 5, 1, tzinfo=timezone.utc)


def unit(unit_id: str, *, metadata: dict | None = None) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="item",
        title=f"Title {unit_id}",
        content="Content",
        metadata=metadata or {},
        tags=[],
        created_at=UNIT_TIME,
        ingested_at=UNIT_TIME,
        updated_at=UNIT_TIME,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_unit_citation_inventory_csv_empty_input_returns_header():
    assert export_unit_citation_inventory_csv([]) == (
        "unit_id,title,citation_field_count,has_doi,has_isbn,has_pmid,has_arxiv_id,"
        "has_url,has_author,has_publisher,has_published_date\n"
    )


def test_export_unit_citation_inventory_csv_detects_common_unit_fields():
    text = export_unit_citation_inventory_csv(
        [
            unit(
                "a",
                metadata={
                    "doi": "10.1000/example",
                    "isbn": "123",
                    "pmid": "42",
                    "arxiv_id": "2401.00001",
                    "url": "https://example.test",
                    "author": "Ada",
                    "publisher": "Press",
                    "published": "2026-01-01",
                },
            )
        ]
    )

    row = rows(text)[0]
    assert row["citation_field_count"] == "8"
    assert row["has_doi"] == "true"
    assert row["has_isbn"] == "true"
    assert row["has_pmid"] == "true"
    assert row["has_arxiv_id"] == "true"
    assert row["has_url"] == "true"
    assert row["has_author"] == "true"
    assert row["has_publisher"] == "true"
    assert row["has_published_date"] == "true"


def test_export_unit_citation_inventory_csv_considers_source_metadata_without_double_counting():
    text = export_unit_citation_inventory_csv(
        [
            unit(
                "a",
                metadata={
                    "doi": "10.1000/unit",
                    "source": {"doi": "10.1000/source", "authors": ["Ada"], "date": "2026"},
                },
            )
        ]
    )

    row = rows(text)[0]
    assert row["citation_field_count"] == "3"
    assert row["has_doi"] == "true"
    assert row["has_author"] == "true"
    assert row["has_published_date"] == "true"


def test_export_unit_citation_inventory_csv_includes_units_without_citation_metadata():
    row = rows(export_unit_citation_inventory_csv([unit("a", metadata={"doi": ""})]))[0]

    assert row["citation_field_count"] == "0"
    assert row["has_doi"] == "false"


def test_export_unit_citation_inventory_csv_path_mode(tmp_path):
    path = tmp_path / "citations.csv"
    stats = export_unit_citation_inventory_csv([unit("a", metadata={"doi": "10.1000/example"})], path)

    assert rows(path.read_text(encoding="utf-8"))[0]["has_doi"] == "true"
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
