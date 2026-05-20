from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_url_scheme_inventory_csv
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, *, metadata: dict | None = None, content: str = "", source_project: str = "Project") -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=f"Title {unit_id}",
        content=content,
        metadata=metadata or {},
        tags=[],
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_url_scheme_inventory_csv_empty_input_has_header_only():
    assert export_unit_url_scheme_inventory_csv([]) == (
        "scheme,source_project,source_entity_type,unit_count,url_count,sample_units\n"
    )


def test_unit_url_scheme_inventory_csv_extracts_metadata_nested_values_and_content():
    text = export_unit_url_scheme_inventory_csv(
        [
            unit(
                "a",
                metadata={
                    "url": "https://example.test/a",
                    "nested": {"links": ["mailto:a@example.test", "obsidian://open?vault=notes"]},
                },
                content="See http://example.test/b and things:///show?id=1",
            ),
            unit("b", metadata={"source_url": "https://example.test/b"}),
        ]
    )

    assert rows(text) == [
        {"scheme": "http", "source_project": "Project", "source_entity_type": "note", "unit_count": "1", "url_count": "1", "sample_units": "a"},
        {"scheme": "https", "source_project": "Project", "source_entity_type": "note", "unit_count": "2", "url_count": "2", "sample_units": "a; b"},
        {"scheme": "mailto", "source_project": "Project", "source_entity_type": "note", "unit_count": "1", "url_count": "1", "sample_units": "a"},
        {"scheme": "obsidian", "source_project": "Project", "source_entity_type": "note", "unit_count": "1", "url_count": "1", "sample_units": "a"},
        {"scheme": "things", "source_project": "Project", "source_entity_type": "note", "unit_count": "1", "url_count": "1", "sample_units": "a"},
    ]


def test_unit_url_scheme_inventory_csv_ignores_invalid_and_schemeless_links():
    text = export_unit_url_scheme_inventory_csv(
        [unit("a", metadata={"url": "example.test/path", "note": "Reminder: call back"}, content="bad://")]
    )

    assert rows(text) == []


def test_unit_url_scheme_inventory_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "schemes.csv"
    units = [unit("a", metadata={"url": "https://example.test"})]

    expected = export_unit_url_scheme_inventory_csv(units)
    stats = export_unit_url_scheme_inventory_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "scheme_group_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }
