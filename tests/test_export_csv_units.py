from __future__ import annotations

import csv
from datetime import datetime, timezone

from graph.export.csv_units import export_units_to_csv
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit

UNIT_TIME = datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)


def unit(
    unit_id: str,
    *,
    content: str = "Alpha content",
    metadata: dict | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=f"Title {unit_id}",
        content=content,
        content_type=ContentType.FINDING,
        tags=["storage", "solar"],
        confidence=0.8,
        utility_score=0.6,
        created_at=UNIT_TIME,
        ingested_at=UNIT_TIME,
        updated_at=UNIT_TIME,
        metadata=metadata or {},
    )


def rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_export_units_to_csv_basic_export(tmp_path):
    path = tmp_path / "units.csv"

    export_units_to_csv([unit("unit-a")], path)

    row = rows(path)[0]
    assert row["id"] == "unit-a"
    assert row["title"] == "Title unit-a"
    assert row["content"] == "Alpha content"


def test_export_units_to_csv_column_selection(tmp_path):
    path = tmp_path / "units.csv"

    stats = export_units_to_csv([unit("unit-a")], path, columns=["id", "title"])

    assert rows(path) == [{"id": "unit-a", "title": "Title unit-a"}]
    assert stats["columns"] == ["id", "title"]


def test_export_units_to_csv_flattens_metadata_fields(tmp_path):
    path = tmp_path / "units.csv"

    export_units_to_csv(
        [unit("unit-a", metadata={"source": {"url": "https://example.test"}, "rank": 3})],
        path,
        columns=["id", "metadata.source.url", "metadata.rank"],
        metadata_fields=["source.url", "rank"],
    )

    assert rows(path) == [
        {
            "id": "unit-a",
            "metadata.source.url": "https://example.test",
            "metadata.rank": "3",
        }
    ]


def test_export_units_to_csv_escapes_multiline_content(tmp_path):
    path = tmp_path / "units.csv"

    export_units_to_csv([unit("unit-a", content='Line 1\n"Line, 2"')], path)

    assert rows(path)[0]["content"] == 'Line 1\n"Line, 2"'
