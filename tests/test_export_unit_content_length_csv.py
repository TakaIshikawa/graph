from __future__ import annotations

import csv
from io import StringIO

import pytest

from graph.export import export_unit_content_length_csv
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    *,
    source_project: SourceProject | str | None = SourceProject.MAX,
    source_entity_type: str | None = "note",
    title: str | None = "",
    content: str | None = "",
) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type=source_entity_type,
        title=title,
        content=content,
        content_type=ContentType.INSIGHT,
        metadata={},
        tags=[],
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_content_length_csv_groups_by_source_project_and_entity_type():
    text = export_unit_content_length_csv(
        [
            unit("a", source_project=SourceProject.MAX, source_entity_type="note", title="Alpha", content="one two"),
            unit("b", source_project=SourceProject.MAX, source_entity_type="note", title="Beta", content="  spaced\ntext "),
            unit("c", source_project=SourceProject.PINBOARD, source_entity_type="bookmark", title="Link"),
        ],
        bucket_size=10,
    )

    assert text.splitlines()[0] == (
        "source_project,source_entity_type,unit_count,min_chars,max_chars,"
        "average_chars,empty_content_count,length_buckets"
    )
    assert rows(text) == [
        {
            "source_project": "max",
            "source_entity_type": "note",
            "unit_count": "2",
            "min_chars": "13",
            "max_chars": "16",
            "average_chars": "14.50",
            "empty_content_count": "0",
            "length_buckets": "10-19:2",
        },
        {
            "source_project": "pinboard",
            "source_entity_type": "bookmark",
            "unit_count": "1",
            "min_chars": "4",
            "max_chars": "4",
            "average_chars": "4.00",
            "empty_content_count": "0",
            "length_buckets": "0-9:1",
        },
    ]


def test_unit_content_length_csv_groups_missing_source_fields_as_unknown():
    text = export_unit_content_length_csv(
        [
            unit("a", source_project=None, source_entity_type=None, title=None, content=""),
            unit("b", source_project="", source_entity_type="", title="  ", content="\n"),
        ],
        bucket_size=5,
    )

    assert rows(text) == [
        {
            "source_project": "Unknown",
            "source_entity_type": "Unknown",
            "unit_count": "2",
            "min_chars": "0",
            "max_chars": "0",
            "average_chars": "0.00",
            "empty_content_count": "2",
            "length_buckets": "0-4:2",
        }
    ]


def test_unit_content_length_csv_is_deterministic_for_reversed_input():
    units = [
        unit("c", source_project="Source B", source_entity_type="zeta", title="ccc"),
        unit("a", source_project="Source A", source_entity_type="zeta", title="a"),
        unit("b", source_project="Source A", source_entity_type="alpha", title="bb"),
    ]

    assert export_unit_content_length_csv(units, bucket_size=2) == export_unit_content_length_csv(
        reversed(units), bucket_size=2
    )


def test_unit_content_length_csv_empty_input_returns_header_only_csv():
    assert export_unit_content_length_csv([]) == (
        "source_project,source_entity_type,unit_count,min_chars,max_chars,"
        "average_chars,empty_content_count,length_buckets\n"
    )


def test_unit_content_length_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "reports" / "unit-length.csv"
    units = [unit("a", title="Title", content="content")]

    expected = export_unit_content_length_csv(units, bucket_size=25)
    stats = export_unit_content_length_csv(units, path, bucket_size=25)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "rows_exported": 1,
        "bucket_size": 25,
        "bytes_written": path.stat().st_size,
    }


@pytest.mark.parametrize("bucket_size", [0, -1, 1.2, True, "100"])
def test_unit_content_length_csv_validates_bucket_size(bucket_size):
    with pytest.raises(ValueError, match="bucket_size must be a positive integer"):
        export_unit_content_length_csv([], bucket_size=bucket_size)
