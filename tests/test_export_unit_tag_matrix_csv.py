from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_tag_matrix_csv
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    *,
    source_project: SourceProject | str = SourceProject.CSV,
    source_id: str | None = None,
    content_type: ContentType | str = ContentType.INSIGHT,
    metadata: dict | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=source_id or f"source-{unit_id}",
        source_entity_type="item",
        title=f"Unit {unit_id}",
        content="",
        content_type=content_type,
        metadata=metadata or {},
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_tag_matrix_csv_empty_input_returns_header():
    assert export_unit_tag_matrix_csv([]) == (
        "source_project,tag,unit_count,content_type_count,source_id_count,example_unit_ids\n"
    )


def test_unit_tag_matrix_csv_normalizes_string_and_list_tags():
    text = export_unit_tag_matrix_csv(
        [
            unit("b", metadata={"tags": " Beta ; alpha, beta ", "keywords": ["Alpha", "delta"]}),
            unit("a", metadata={"labels": ["alpha", "gamma", " "], "tag": "gamma"}),
        ]
    )

    assert rows(text) == [
        {
            "source_project": "csv",
            "tag": "alpha",
            "unit_count": "2",
            "content_type_count": "1",
            "source_id_count": "2",
            "example_unit_ids": "a; b",
        },
        {
            "source_project": "csv",
            "tag": "Beta",
            "unit_count": "1",
            "content_type_count": "1",
            "source_id_count": "1",
            "example_unit_ids": "b",
        },
        {
            "source_project": "csv",
            "tag": "delta",
            "unit_count": "1",
            "content_type_count": "1",
            "source_id_count": "1",
            "example_unit_ids": "b",
        },
        {
            "source_project": "csv",
            "tag": "gamma",
            "unit_count": "1",
            "content_type_count": "1",
            "source_id_count": "1",
            "example_unit_ids": "a",
        },
    ]


def test_unit_tag_matrix_csv_counts_distinct_content_types_and_source_ids():
    text = export_unit_tag_matrix_csv(
        [
            unit("a", source_id="same", content_type="note", metadata={"tags": ["shared"]}),
            unit("b", source_id="same", content_type=ContentType.INSIGHT, metadata={"tags": ["shared"]}),
            unit("c", source_id="other", content_type=ContentType.INSIGHT, metadata={"tags": ["shared"]}),
        ]
    )

    assert rows(text)[0] == {
        "source_project": "csv",
        "tag": "shared",
        "unit_count": "3",
        "content_type_count": "2",
        "source_id_count": "2",
        "example_unit_ids": "a; b; c",
    }


def test_unit_tag_matrix_csv_excludes_untagged_rows_but_counts_path_stats(tmp_path):
    path = tmp_path / "tags.csv"
    units = [unit("a", metadata={"tags": ["alpha"]}), unit("b", metadata={})]

    expected = export_unit_tag_matrix_csv(units)
    stats = export_unit_tag_matrix_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 2,
        "untagged_unit_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }
