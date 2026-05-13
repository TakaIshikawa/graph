from __future__ import annotations

import csv
from io import StringIO

import pytest

from graph.export import export_source_entity_type_matrix_csv
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    *,
    source_project: SourceProject | str | None,
    source_entity_type: str | None,
) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type=source_entity_type,
        title=f"Title {unit_id}",
        content="content",
        content_type=ContentType.INSIGHT,
        metadata={},
        tags=[],
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_source_entity_type_matrix_csv_counts_types_across_source_projects():
    text = export_source_entity_type_matrix_csv(
        [
            unit("a", source_project=SourceProject.MAX, source_entity_type="note"),
            unit("b", source_project=SourceProject.MAX, source_entity_type="bookmark"),
            unit("c", source_project=SourceProject.PINBOARD, source_entity_type="bookmark"),
        ]
    )

    assert text.splitlines()[0] == "source_entity_type,max,pinboard,total"
    assert rows(text) == [
        {"source_entity_type": "bookmark", "max": "1", "pinboard": "1", "total": "2"},
        {"source_entity_type": "note", "max": "1", "pinboard": "0", "total": "1"},
    ]


def test_source_entity_type_matrix_csv_deduplicates_by_unit_id():
    text = export_source_entity_type_matrix_csv(
        [
            unit("same", source_project="A", source_entity_type="note"),
            unit("same", source_project="A", source_entity_type="note"),
            unit("other", source_project="A", source_entity_type="note"),
        ]
    )

    assert rows(text) == [{"source_entity_type": "note", "A": "2", "total": "2"}]


def test_source_entity_type_matrix_csv_normalizes_blank_values_to_unknown():
    text = export_source_entity_type_matrix_csv(
        [
            unit("a", source_project=None, source_entity_type=None),
            unit("b", source_project="", source_entity_type=" "),
        ]
    )

    assert text.splitlines()[0] == "source_entity_type,Unknown,total"
    assert rows(text) == [{"source_entity_type": "Unknown", "Unknown": "2", "total": "2"}]


def test_source_entity_type_matrix_csv_min_count_filters_low_frequency_types():
    text = export_source_entity_type_matrix_csv(
        [
            unit("a", source_project="A", source_entity_type="common"),
            unit("b", source_project="B", source_entity_type="common"),
            unit("c", source_project="A", source_entity_type="rare"),
        ],
        min_count=2,
    )

    assert rows(text) == [{"source_entity_type": "common", "A": "1", "B": "1", "total": "2"}]


def test_source_entity_type_matrix_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "source-entity-type-matrix.csv"
    units = [unit("a", source_project="A", source_entity_type="note")]

    expected = export_source_entity_type_matrix_csv(units)
    stats = export_source_entity_type_matrix_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "source_project_count": 1,
        "source_entity_type_count": 1,
        "rows_exported": 1,
        "min_count": 1,
        "bytes_written": path.stat().st_size,
    }


@pytest.mark.parametrize("min_count", [0, -1, 1.2, True, "2"])
def test_source_entity_type_matrix_csv_validates_min_count(min_count):
    with pytest.raises(ValueError, match="min_count must be a positive integer"):
        export_source_entity_type_matrix_csv([], min_count=min_count)
