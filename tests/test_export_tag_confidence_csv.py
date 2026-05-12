from __future__ import annotations

import csv
from io import StringIO

import pytest

from graph.export import export_tag_confidence_csv
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    *,
    source_project: SourceProject | str | None = SourceProject.MAX,
    tags: list[str] | None = None,
    confidence: object = None,
) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=f"Title {unit_id}",
        content=f"Content {unit_id}",
        content_type=ContentType.INSIGHT,
        metadata={},
        tags=tags or [],
        confidence=confidence,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_tag_confidence_csv_summarizes_confidence_by_tag():
    text = export_tag_confidence_csv(
        [
            unit("a", source_project=SourceProject.MAX, tags=["AI", "storage", "AI"], confidence=0.9),
            unit("b", source_project=SourceProject.PINBOARD, tags=["AI", " storage\nsystems "], confidence=0.3),
            unit("c", source_project=SourceProject.MAX, tags=["AI"], confidence=None),
        ]
    )

    assert text.splitlines()[0] == (
        "tag,unit_count,source_project_count,average_confidence,"
        "low_confidence_unit_count,missing_confidence_unit_count"
    )
    assert rows(text) == [
        {
            "tag": "AI",
            "unit_count": "3",
            "source_project_count": "2",
            "average_confidence": "0.60",
            "low_confidence_unit_count": "1",
            "missing_confidence_unit_count": "1",
        },
        {
            "tag": "storage",
            "unit_count": "1",
            "source_project_count": "1",
            "average_confidence": "0.90",
            "low_confidence_unit_count": "0",
            "missing_confidence_unit_count": "0",
        },
        {
            "tag": "storage systems",
            "unit_count": "1",
            "source_project_count": "1",
            "average_confidence": "0.30",
            "low_confidence_unit_count": "1",
            "missing_confidence_unit_count": "0",
        },
    ]


def test_tag_confidence_csv_treats_boolean_and_non_numeric_confidence_as_missing():
    text = export_tag_confidence_csv(
        [
            unit("a", tags=["tag"], confidence=True),
            unit("b", tags=["tag"], confidence="0.7"),
            unit("c", tags=["tag"], confidence=0.7),
        ]
    )

    assert rows(text) == [
        {
            "tag": "tag",
            "unit_count": "3",
            "source_project_count": "1",
            "average_confidence": "0.70",
            "low_confidence_unit_count": "0",
            "missing_confidence_unit_count": "2",
        }
    ]


def test_tag_confidence_csv_filters_by_min_units():
    text = export_tag_confidence_csv(
        [
            unit("a", tags=["keep", "drop"], confidence=0.2),
            unit("b", tags=["keep"], confidence=0.8),
        ],
        min_units=2,
    )

    assert rows(text) == [
        {
            "tag": "keep",
            "unit_count": "2",
            "source_project_count": "1",
            "average_confidence": "0.50",
            "low_confidence_unit_count": "1",
            "missing_confidence_unit_count": "0",
        }
    ]


def test_tag_confidence_csv_is_deterministic_for_reversed_input():
    units = [
        unit("b", source_project="Source B", tags=["beta"], confidence=0.7),
        unit("a", source_project="Source A", tags=["alpha"], confidence=0.4),
        unit("c", source_project="Source A", tags=["beta"], confidence=None),
    ]

    assert export_tag_confidence_csv(units) == export_tag_confidence_csv(reversed(units))


def test_tag_confidence_csv_empty_input_returns_header_only_csv():
    assert export_tag_confidence_csv([]) == (
        "tag,unit_count,source_project_count,average_confidence,"
        "low_confidence_unit_count,missing_confidence_unit_count\n"
    )


def test_tag_confidence_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "reports" / "tag-confidence.csv"
    units = [unit("a", tags=["ai"], confidence=0.6)]

    expected = export_tag_confidence_csv(units)
    stats = export_tag_confidence_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "rows_exported": 1,
        "min_units": 1,
        "bytes_written": path.stat().st_size,
    }


@pytest.mark.parametrize("min_units", [0, -1, 1.2, True, "2"])
def test_tag_confidence_csv_validates_min_units(min_units):
    with pytest.raises(ValueError, match="min_units must be a positive integer"):
        export_tag_confidence_csv([], min_units=min_units)
