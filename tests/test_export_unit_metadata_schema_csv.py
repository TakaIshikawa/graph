from __future__ import annotations

import csv
from io import StringIO

import pytest

from graph.export import export_unit_metadata_schema_csv
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    *,
    source_project: SourceProject | str | None = SourceProject.MAX,
    source_entity_type: str | None = "note",
    metadata: object | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type=source_entity_type,
        title=f"Title {unit_id}",
        content="content",
        content_type=ContentType.INSIGHT,
        metadata={} if metadata is None else metadata,
        tags=[],
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_metadata_schema_csv_reports_key_usage_by_source_and_type():
    text = export_unit_metadata_schema_csv(
        [
            unit("a", metadata={"Author": "Ada", "rating": 5}),
            unit("b", metadata={"Author": "Bob", "rating": "5"}),
            unit("c", source_project=SourceProject.PINBOARD, source_entity_type="bookmark", metadata={"url": "https://example.com"}),
        ]
    )

    assert text.splitlines()[0] == (
        "source_project,source_entity_type,metadata_key,unit_count,present_count,"
        "coverage_percent,value_types,sample_values"
    )
    assert rows(text) == [
        {
            "source_project": "max",
            "source_entity_type": "note",
            "metadata_key": "Author",
            "unit_count": "2",
            "present_count": "2",
            "coverage_percent": "100.00",
            "value_types": "str",
            "sample_values": "Ada; Bob",
        },
        {
            "source_project": "max",
            "source_entity_type": "note",
            "metadata_key": "rating",
            "unit_count": "2",
            "present_count": "2",
            "coverage_percent": "100.00",
            "value_types": "int; str",
            "sample_values": "5",
        },
        {
            "source_project": "pinboard",
            "source_entity_type": "bookmark",
            "metadata_key": "url",
            "unit_count": "1",
            "present_count": "1",
            "coverage_percent": "100.00",
            "value_types": "str",
            "sample_values": "https://example.com",
        },
    ]


def test_unit_metadata_schema_csv_handles_non_dict_metadata_as_empty():
    text = export_unit_metadata_schema_csv(
        [
            unit("a", metadata=["not", "metadata"]),
            unit("b", metadata={"source": "manual"}),
        ]
    )

    assert rows(text) == [
        {
            "source_project": "max",
            "source_entity_type": "note",
            "metadata_key": "source",
            "unit_count": "2",
            "present_count": "1",
            "coverage_percent": "50.00",
            "value_types": "str",
            "sample_values": "manual",
        }
    ]


def test_unit_metadata_schema_csv_filters_rows_below_min_count():
    text = export_unit_metadata_schema_csv(
        [
            unit("a", metadata={"common": "one", "rare": True}),
            unit("b", metadata={"common": "two"}),
        ],
        min_count=2,
    )

    assert [row["metadata_key"] for row in rows(text)] == ["common"]


def test_unit_metadata_schema_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "metadata-schema.csv"
    units = [unit("a", metadata={"author": "Ada"})]

    expected = export_unit_metadata_schema_csv(units)
    stats = export_unit_metadata_schema_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "rows_exported": 1,
        "min_count": 1,
        "bytes_written": path.stat().st_size,
    }


@pytest.mark.parametrize("min_count", [0, -1, 1.5, True, "2"])
def test_unit_metadata_schema_csv_validates_min_count(min_count):
    with pytest.raises(ValueError, match="min_count must be a positive integer"):
        export_unit_metadata_schema_csv([], min_count=min_count)
