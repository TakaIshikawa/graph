from __future__ import annotations

import csv
from io import StringIO

import pytest

from graph.export.source_content_richness_csv import export_source_content_richness_csv
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, source_project: SourceProject | str | None, content: object) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=f"Title {unit_id}",
        content=content,
        content_type=ContentType.INSIGHT,
        metadata={},
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_source_content_richness_aggregates_word_counts_by_source_project():
    text = export_source_content_richness_csv(
        [
            unit("a", SourceProject.MAX, "one two three"),
            unit("b", SourceProject.MAX, "one two three four five six"),
            unit("c", SourceProject.MAX, "  \n\t "),
            unit("d", None, "solo"),
        ],
        short_word_threshold=4,
    )

    assert text.splitlines()[0] == (
        "source_project,unit_count,empty_content_count,short_content_count,"
        "min_words,max_words,average_words"
    )
    assert rows(text) == [
        {
            "source_project": "max",
            "unit_count": "3",
            "empty_content_count": "1",
            "short_content_count": "1",
            "min_words": "0",
            "max_words": "6",
            "average_words": "3.00",
        },
        {
            "source_project": "Unknown",
            "unit_count": "1",
            "empty_content_count": "0",
            "short_content_count": "1",
            "min_words": "1",
            "max_words": "1",
            "average_words": "1.00",
        },
    ]


def test_source_content_richness_normalizes_whitespace_before_counting_words():
    text = export_source_content_richness_csv(
        [
            unit("a", "Source A", " alpha\nbeta\tgamma  "),
            unit("b", "Source A", None),
        ],
        short_word_threshold=3,
    )

    assert rows(text)[0] == {
        "source_project": "Source A",
        "unit_count": "2",
        "empty_content_count": "1",
        "short_content_count": "0",
        "min_words": "0",
        "max_words": "3",
        "average_words": "1.50",
    }


def test_source_content_richness_validates_short_word_threshold():
    for value in (-1, 1.5, "3", True):
        with pytest.raises(ValueError, match="short_word_threshold"):
            export_source_content_richness_csv([], short_word_threshold=value)

    assert rows(export_source_content_richness_csv([unit("a", "Source A", "one")], short_word_threshold=0))[
        0
    ]["short_content_count"] == "0"


def test_source_content_richness_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "reports" / "content-richness.csv"
    units = [unit("a", "Source A", "one two"), unit("b", "Source A", "")]

    expected = export_source_content_richness_csv(units)
    stats = export_source_content_richness_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 2,
        "source_project_count": 1,
        "rows_exported": 1,
        "short_word_threshold": 25,
        "bytes_written": path.stat().st_size,
    }


def test_source_content_richness_is_deterministic_for_reversed_input():
    units = [
        unit("a", "Source B", "one"),
        unit("b", "Source A", "one two"),
        unit("c", "Source A", "one two three"),
    ]

    assert export_source_content_richness_csv(units) == export_source_content_richness_csv(reversed(units))
