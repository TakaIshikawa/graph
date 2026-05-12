from __future__ import annotations

import csv
from io import StringIO

import pytest

from graph.export import export_source_confidence_summary_csv
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, source_project: SourceProject | str | None, confidence: object) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=f"Title {unit_id}",
        content="content",
        content_type=ContentType.INSIGHT,
        metadata={},
        confidence=confidence,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_source_confidence_summary_aggregates_by_source_project():
    text = export_source_confidence_summary_csv(
        [
            unit("a", SourceProject.MAX, 0.25),
            unit("b", SourceProject.MAX, 0.8),
            unit("c", SourceProject.MAX, None),
            unit("d", None, 0.95),
        ]
    )

    assert rows(text) == [
        {
            "source_project": "max",
            "unit_count": "3",
            "confidence_count": "2",
            "missing_confidence_count": "1",
            "min_confidence": "0.25",
            "max_confidence": "0.80",
            "average_confidence": "0.53",
            "low_confidence_count": "1",
            "high_confidence_count": "1",
        },
        {
            "source_project": "Unknown",
            "unit_count": "1",
            "confidence_count": "1",
            "missing_confidence_count": "0",
            "min_confidence": "0.95",
            "max_confidence": "0.95",
            "average_confidence": "0.95",
            "low_confidence_count": "0",
            "high_confidence_count": "1",
        },
    ]


def test_source_confidence_summary_thresholds_affect_counts_and_validate():
    text = export_source_confidence_summary_csv(
        [unit("a", "Source A", 0.4), unit("b", "Source A", 0.7), unit("c", "Source A", 0.9)],
        low_threshold=0.75,
        high_threshold=0.9,
    )

    assert rows(text)[0]["low_confidence_count"] == "2"
    assert rows(text)[0]["high_confidence_count"] == "1"

    with pytest.raises(ValueError, match="low_threshold"):
        export_source_confidence_summary_csv([], low_threshold=-0.1)
    with pytest.raises(ValueError, match="high_threshold"):
        export_source_confidence_summary_csv([], high_threshold=1.1)
    with pytest.raises(ValueError, match="less than"):
        export_source_confidence_summary_csv([], low_threshold=0.8, high_threshold=0.8)
    with pytest.raises(ValueError, match="low_threshold"):
        export_source_confidence_summary_csv([], low_threshold=True)


def test_source_confidence_summary_handles_all_missing_confidence():
    text = export_source_confidence_summary_csv(
        [unit("a", "Source A", None), unit("b", "Source A", "not numeric")]
    )

    assert rows(text)[0] == {
        "source_project": "Source A",
        "unit_count": "2",
        "confidence_count": "0",
        "missing_confidence_count": "2",
        "min_confidence": "",
        "max_confidence": "",
        "average_confidence": "",
        "low_confidence_count": "0",
        "high_confidence_count": "0",
    }


def test_source_confidence_summary_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "confidence.csv"
    units = [unit("a", "Source A", 0.6)]

    expected = export_source_confidence_summary_csv(units)
    stats = export_source_confidence_summary_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "source_project_count": 1,
        "rows_exported": 1,
        "low_threshold": 0.5,
        "high_threshold": 0.8,
        "bytes_written": path.stat().st_size,
    }


def test_source_confidence_summary_is_deterministic_for_reversed_input():
    units = [
        unit("a", "Source B", 0.7),
        unit("b", "Source A", 0.4),
        unit("c", "Source A", 0.9),
    ]

    assert export_source_confidence_summary_csv(units) == export_source_confidence_summary_csv(
        reversed(units)
    )
