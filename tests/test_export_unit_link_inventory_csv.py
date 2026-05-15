from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO

from graph.export import export_unit_link_inventory_csv
from graph.types.enums import SourceProject
from graph.types.models import KnowledgeUnit

UNIT_TIME = datetime(2026, 5, 1, tzinfo=timezone.utc)


def unit(unit_id: str, *, metadata: dict | None = None, title: str | None = None) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="item",
        title=title or f"Title {unit_id}",
        content="Content",
        metadata=metadata or {},
        tags=[],
        created_at=UNIT_TIME,
        ingested_at=UNIT_TIME,
        updated_at=UNIT_TIME,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_unit_link_inventory_csv_empty_input_returns_header():
    assert export_unit_link_inventory_csv([]) == "unit_id,title,link_count,unique_domain_count,domains,urls\n"


def test_export_unit_link_inventory_csv_counts_duplicates_and_sorts_domains():
    text = export_unit_link_inventory_csv(
        [
            unit(
                "a",
                metadata={
                    "links": ["https://b.test/2", "https://a.test/1", "https://a.test/1"],
                    "url": "https://B.test/3",
                },
            )
        ]
    )

    row = rows(text)[0]
    assert row["link_count"] == "4"
    assert row["unique_domain_count"] == "2"
    assert row["domains"] == "a.test; b.test"
    assert row["urls"] == "https://a.test/1; https://b.test/2; https://B.test/3"


def test_export_unit_link_inventory_csv_extracts_source_urls():
    text = export_unit_link_inventory_csv(
        [
            unit(
                "a",
                metadata={
                    "source": {"url": "https://example.test/a"},
                    "sources": [{"source_url": "https://other.test/b"}],
                },
            )
        ]
    )

    row = rows(text)[0]
    assert row["link_count"] == "2"
    assert row["domains"] == "example.test; other.test"


def test_export_unit_link_inventory_csv_includes_units_without_links():
    text = export_unit_link_inventory_csv([unit("a", metadata={"url": "not a url"})])

    assert rows(text)[0] == {
        "unit_id": "a",
        "title": "Title a",
        "link_count": "0",
        "unique_domain_count": "0",
        "domains": "",
        "urls": "",
    }


def test_export_unit_link_inventory_csv_path_mode(tmp_path):
    path = tmp_path / "links.csv"
    stats = export_unit_link_inventory_csv([unit("a", metadata={"url": "https://example.test"})], path)

    assert rows(path.read_text(encoding="utf-8"))[0]["domains"] == "example.test"
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
