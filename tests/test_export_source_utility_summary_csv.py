from __future__ import annotations

import csv
from io import StringIO

import pytest

from graph.export.source_utility_summary_csv import export_source_utility_summary_csv
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, source_project: SourceProject | str | None, utility_score: object) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=f"Title {unit_id}",
        content="content",
        content_type=ContentType.INSIGHT,
        metadata={},
        utility_score=utility_score,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_source_utility_summary_aggregates_by_source_project():
    text = export_source_utility_summary_csv(
        [
            unit("a", SourceProject.MAX, 0.2),
            unit("b", SourceProject.MAX, 0.7),
            unit("c", SourceProject.MAX, None),
            unit("d", None, 0.95),
        ]
    )

    assert text.splitlines()[0] == (
        "source_project,unit_count,utility_count,missing_utility_count,min_utility,"
        "max_utility,average_utility,low_utility_count,high_utility_count"
    )
    assert rows(text) == [
        {
            "source_project": "max",
            "unit_count": "3",
            "utility_count": "2",
            "missing_utility_count": "1",
            "min_utility": "0.20",
            "max_utility": "0.70",
            "average_utility": "0.45",
            "low_utility_count": "1",
            "high_utility_count": "1",
        },
        {
            "source_project": "Unknown",
            "unit_count": "1",
            "utility_count": "1",
            "missing_utility_count": "0",
            "min_utility": "0.95",
            "max_utility": "0.95",
            "average_utility": "0.95",
            "low_utility_count": "0",
            "high_utility_count": "1",
        },
    ]


def test_source_utility_summary_thresholds_affect_counts_and_validate():
    text = export_source_utility_summary_csv(
        [unit("a", "Source A", 0.4), unit("b", "Source A", 0.7), unit("c", "Source A", 0.9)],
        low_threshold=0.75,
        high_threshold=0.9,
    )

    assert rows(text)[0]["low_utility_count"] == "2"
    assert rows(text)[0]["high_utility_count"] == "1"

    invalid_calls = [
        {"low_threshold": -0.1},
        {"high_threshold": 1.1},
        {"low_threshold": 0.8, "high_threshold": 0.8},
        {"low_threshold": True},
        {"high_threshold": "0.7"},
    ]
    for kwargs in invalid_calls:
        with pytest.raises(ValueError):
            export_source_utility_summary_csv([], **kwargs)


def test_source_utility_summary_handles_all_missing_utility():
    text = export_source_utility_summary_csv(
        [unit("a", "Source A", None), unit("b", "Source A", "not numeric"), unit("c", "Source A", True)]
    )

    assert rows(text)[0] == {
        "source_project": "Source A",
        "unit_count": "3",
        "utility_count": "0",
        "missing_utility_count": "3",
        "min_utility": "",
        "max_utility": "",
        "average_utility": "",
        "low_utility_count": "0",
        "high_utility_count": "0",
    }


def test_source_utility_summary_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "reports" / "utility.csv"
    units = [unit("a", "Source A", 0.6)]

    expected = export_source_utility_summary_csv(units)
    stats = export_source_utility_summary_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "source_project_count": 1,
        "rows_exported": 1,
        "low_threshold": 0.3,
        "high_threshold": 0.7,
        "bytes_written": path.stat().st_size,
    }


def test_source_utility_summary_is_deterministic_for_reversed_input():
    units = [
        unit("a", "Source B", 0.7),
        unit("b", "Source A", 0.4),
        unit("c", "Source A", 0.9),
    ]

    assert export_source_utility_summary_csv(units) == export_source_utility_summary_csv(reversed(units))
