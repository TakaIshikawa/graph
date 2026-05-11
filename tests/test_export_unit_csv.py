from __future__ import annotations

import csv
from datetime import datetime, timezone

from graph.export import export_units_to_csv
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


UNIT_TIME = datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)


def unit(
    unit_id: str,
    *,
    source_project: SourceProject | str = SourceProject.MAX,
    content_type: ContentType = ContentType.FINDING,
    tags: list[str] | None = None,
    content: str = "Alpha content",
    confidence: float | None = 0.8,
    utility_score: float | None = 0.6,
    metadata: dict | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=f"Title {unit_id}",
        content=content,
        content_type=content_type,
        tags=tags or ["storage", "solar"],
        confidence=confidence,
        utility_score=utility_score,
        created_at=UNIT_TIME,
        ingested_at=UNIT_TIME,
        updated_at=UNIT_TIME,
        metadata=metadata or {},
    )


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_export_units_csv_writes_stable_headers_and_stats(tmp_path):
    path = tmp_path / "nested" / "units.csv"

    stats = export_units_to_csv([unit("unit-a")], path)

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == [
        "id",
        "source_project",
        "source_id",
        "source_entity_type",
        "title",
        "content_type",
        "tags",
        "confidence",
        "utility_score",
        "created_at",
        "updated_at",
        "content",
    ]
    assert stats == {
        "path": str(path),
        "units_scanned": 1,
        "units_exported": 1,
        "content_included": True,
        "bytes_written": path.stat().st_size,
    }
    assert rows[0]["id"] == "unit-a"


def test_export_units_csv_sorts_rows_by_unit_id(tmp_path):
    path = tmp_path / "units.csv"

    export_units_to_csv([unit("unit-c"), unit("unit-a"), unit("unit-b")], path)

    assert [row["id"] for row in read_rows(path)] == ["unit-a", "unit-b", "unit-c"]


def test_export_units_csv_serializes_enums_datetimes_and_numeric_fields(tmp_path):
    path = tmp_path / "units.csv"

    export_units_to_csv(
        [
            unit(
                "unit-a",
                source_project=SourceProject.PINBOARD,
                content_type=ContentType.IDEA,
                confidence=None,
                utility_score=0.0,
            )
        ],
        path,
    )

    row = read_rows(path)[0]
    assert row["source_project"] == "pinboard"
    assert row["content_type"] == "idea"
    assert row["confidence"] == ""
    assert row["utility_score"] == "0.0"
    assert row["created_at"] == "2026-05-01T10:15:00+00:00"
    assert row["updated_at"] == "2026-05-01T10:15:00+00:00"


def test_export_units_csv_serializes_tags_as_semicolon_separated_stable_list(tmp_path):
    path = tmp_path / "units.csv"

    export_units_to_csv([unit("unit-a", tags=["zeta", "alpha", "alpha"])], path)

    assert read_rows(path)[0]["tags"] == "alpha;alpha;zeta"


def test_export_units_csv_can_omit_content(tmp_path):
    path = tmp_path / "units.csv"

    stats = export_units_to_csv([unit("unit-a", content="Hidden")], path, include_content=False)

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == [
        "id",
        "source_project",
        "source_id",
        "source_entity_type",
        "title",
        "content_type",
        "tags",
        "confidence",
        "utility_score",
        "created_at",
        "updated_at",
    ]
    assert "content" not in rows[0]
    assert stats["content_included"] is False


def test_export_units_csv_empty_export_writes_header_only(tmp_path):
    path = tmp_path / "empty.csv"

    stats = export_units_to_csv([], path, include_content=False)
    text = path.read_text(encoding="utf-8")

    assert (
        text
        == "id,source_project,source_id,source_entity_type,title,content_type,tags,confidence,utility_score,created_at,updated_at\n"
    )
    assert stats == {
        "path": str(path),
        "units_scanned": 0,
        "units_exported": 0,
        "content_included": False,
        "bytes_written": len(text.encode("utf-8")),
    }


def test_export_units_csv_supports_configurable_columns_and_metadata_fields(tmp_path):
    path = tmp_path / "units.csv"

    stats = export_units_to_csv(
        [unit("unit-a", metadata={"source": {"url": "https://example.test"}, "rank": 3})],
        path,
        columns=["id", "title", "metadata.source.url", "metadata.rank"],
        metadata_fields=["source.url", "rank"],
    )

    rows = read_rows(path)
    assert rows == [
        {
            "id": "unit-a",
            "title": "Title unit-a",
            "metadata.source.url": "https://example.test",
            "metadata.rank": "3",
        }
    ]
    assert stats["columns"] == ["id", "title", "metadata.source.url", "metadata.rank"]
