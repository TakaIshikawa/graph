from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_source_url_duplicates_csv
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    *,
    title: str | None = None,
    source_project: str = "Project A",
    source_entity_type: str = "note",
    metadata: dict | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=unit_id,
        source_entity_type=source_entity_type,
        title=title or f"Title {unit_id}",
        content="content",
        metadata=metadata or {},
        tags=[],
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_source_url_duplicates_csv_empty_input_has_header_only():
    assert export_source_url_duplicates_csv([]) == (
        "normalized_url,unit_count,source_project_count,source_projects,source_entity_types,"
        "unit_ids,titles,raw_urls\n"
    )


def test_source_url_duplicates_csv_groups_equivalent_urls():
    text = export_source_url_duplicates_csv(
        [
            unit("b", title="Beta", metadata={"url": "HTTPS://Example.com/Path?a=1#section"}),
            unit("a", title="Alpha", source_project="Project B", metadata={"source_url": "https://example.com/Path/?a=1"}),
            unit("c", title="Gamma", metadata={"url": "https://example.com/other"}),
        ]
    )

    assert rows(text) == [
        {
            "normalized_url": "https://example.com/Path?a=1",
            "unit_count": "2",
            "source_project_count": "2",
            "source_projects": "Project A; Project B",
            "source_entity_types": "note",
            "unit_ids": "a; b",
            "titles": "Alpha; Beta",
            "raw_urls": "https://example.com/Path/?a=1; HTTPS://Example.com/Path?a=1#section",
        }
    ]


def test_source_url_duplicates_csv_requires_multiple_distinct_units():
    text = export_source_url_duplicates_csv(
        [
            unit("a", metadata={"url": "https://example.com/page", "canonical_url": "https://example.com/page#top"}),
            unit("b", metadata={"url": "https://example.com/other"}),
        ]
    )

    assert rows(text) == []


def test_source_url_duplicates_csv_handles_direct_fields_and_non_string_metadata_values():
    text = export_source_url_duplicates_csv(
        [
            {
                "id": "a",
                "title": "Alpha",
                "source_project": "Project A",
                "source_entity_type": "bookmark",
                "url": "https://example.com/item/",
                "metadata": {"href": 123, "external_url": ["https://example.com/item#fragment"]},
            },
            unit("b", title="Beta", source_entity_type="note", metadata={"link": "HTTPS://EXAMPLE.COM/item"}),
        ]
    )

    assert rows(text)[0] == {
        "normalized_url": "https://example.com/item",
        "unit_count": "2",
        "source_project_count": "1",
        "source_projects": "Project A",
        "source_entity_types": "bookmark; note",
        "unit_ids": "a; b",
        "titles": "Alpha; Beta",
        "raw_urls": "HTTPS://EXAMPLE.COM/item; https://example.com/item#fragment; https://example.com/item/",
    }


def test_source_url_duplicates_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "source-url-duplicates.csv"
    units = [
        unit("a", metadata={"url": "https://example.com/page"}),
        unit("b", metadata={"url": "https://example.com/page/"}),
    ]

    expected = export_source_url_duplicates_csv(units)
    stats = export_source_url_duplicates_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 2,
        "duplicate_url_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }
