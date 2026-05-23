from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO

from graph.export import export_unit_mime_type_inventory_csv
from graph.types.enums import SourceProject
from graph.types.models import KnowledgeUnit

UNIT_TIME = datetime(2026, 5, 1, tzinfo=timezone.utc)


def unit(unit_id: str, *, metadata: dict | None = None) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
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


def test_export_unit_mime_type_inventory_csv_empty_input_returns_header():
    assert export_unit_mime_type_inventory_csv([]) == "mime_type,count,unit_ids,source_keys\n"


def test_export_unit_mime_type_inventory_csv_counts_metadata_mime_values():
    text = export_unit_mime_type_inventory_csv(
        [
            unit("a", metadata={"mime_type": "Text/HTML; charset=utf-8"}),
            unit("b", metadata={"content_type": ["text/html", "application/json"]}),
        ]
    )

    assert rows(text) == [
        {
            "mime_type": "application/json",
            "count": "1",
            "unit_ids": "b",
            "source_keys": "content_type",
        },
        {
            "mime_type": "text/html",
            "count": "2",
            "unit_ids": "a; b",
            "source_keys": "content_type; mime_type",
        },
    ]


def test_export_unit_mime_type_inventory_csv_uses_extension_fallback():
    text = export_unit_mime_type_inventory_csv(
        [
            unit("a", metadata={"attachment": "report.pdf"}),
            unit("b", metadata={"content_path": "https://example.test/readme.md?download=1"}),
        ]
    )

    assert rows(text) == [
        {
            "mime_type": "application/pdf",
            "count": "1",
            "unit_ids": "a",
            "source_keys": "attachment:extension",
        },
        {
            "mime_type": "text/markdown",
            "count": "1",
            "unit_ids": "b",
            "source_keys": "content_path:extension",
        },
    ]


def test_export_unit_mime_type_inventory_csv_uses_unknown_for_blank_and_missing_values():
    text = export_unit_mime_type_inventory_csv(
        [
            unit("a", metadata={"media_type": " "}),
            unit("b", metadata={"title": "No MIME hints"}),
        ]
    )

    assert rows(text) == [
        {
            "mime_type": "unknown",
            "count": "2",
            "unit_ids": "a; b",
            "source_keys": "media_type; missing",
        }
    ]


def test_export_unit_mime_type_inventory_csv_path_mode(tmp_path):
    units = [unit("a", metadata={"attachment_mime_type": "image/png"})]
    expected = export_unit_mime_type_inventory_csv(units)
    path = tmp_path / "mime-types.csv"

    stats = export_unit_mime_type_inventory_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
