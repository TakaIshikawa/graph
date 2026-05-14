from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO

from graph.export import export_unit_attachment_inventory_csv
from graph.types.enums import SourceProject
from graph.types.models import KnowledgeUnit

UNIT_TIME = datetime(2026, 5, 1, tzinfo=timezone.utc)


def unit(unit_id: str, *, metadata: dict | None = None, source_project: object = SourceProject.MAX) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=f"Title {unit_id}",
        content="Content",
        metadata=metadata or {},
        created_at=UNIT_TIME,
        ingested_at=UNIT_TIME,
        updated_at=UNIT_TIME,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_unit_attachment_inventory_csv_empty_input_returns_header():
    assert export_unit_attachment_inventory_csv([]) == (
        "unit_id,source_project,source_entity_type,metadata_key,value_type,value\n"
    )


def test_export_unit_attachment_inventory_csv_flattens_scalar_and_list_values():
    text = export_unit_attachment_inventory_csv(
        [
            unit(
                "a",
                metadata={
                    "url": "https://example.test/doc",
                    "files": ["/tmp/a.pdf", "notes.md"],
                    "ignored": "https://example.test/skip",
                },
            )
        ]
    )

    assert rows(text) == [
        {
            "unit_id": "a",
            "source_project": "max",
            "source_entity_type": "note",
            "metadata_key": "files",
            "value_type": "path",
            "value": "/tmp/a.pdf",
        },
        {
            "unit_id": "a",
            "source_project": "max",
            "source_entity_type": "note",
            "metadata_key": "files",
            "value_type": "text",
            "value": "notes.md",
        },
        {
            "unit_id": "a",
            "source_project": "max",
            "source_entity_type": "note",
            "metadata_key": "url",
            "value_type": "url",
            "value": "https://example.test/doc",
        },
    ]


def test_export_unit_attachment_inventory_csv_classifies_relative_paths_and_text():
    text = export_unit_attachment_inventory_csv(
        [unit("a", metadata={"links": ["../doc.md", "plain reference"]})]
    )

    assert [row["value_type"] for row in rows(text)] == ["path", "text"]


def test_export_unit_attachment_inventory_csv_uses_unknown_fallbacks():
    text = export_unit_attachment_inventory_csv(
        [unit("a", metadata={"attachment": "x"}, source_project="")]
    )

    assert rows(text)[0]["source_project"] == "Unknown"


def test_export_unit_attachment_inventory_csv_path_mode(tmp_path):
    units = [unit("a", metadata={"url": "https://example.test"})]
    expected = export_unit_attachment_inventory_csv(units)
    path = tmp_path / "attachments.csv"

    stats = export_unit_attachment_inventory_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
