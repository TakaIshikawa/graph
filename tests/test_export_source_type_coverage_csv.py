from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_source_type_coverage_csv
from graph.types.enums import ContentType, EdgeRelation, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


def unit(
    unit_id: str,
    source_project: SourceProject | str | None,
    source_entity_type: str | None,
    *,
    confidence: object = None,
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
        confidence=confidence,
    )


def edge(edge_id: str, from_unit_id: str, to_unit_id: str) -> KnowledgeEdge:
    return KnowledgeEdge.model_construct(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=EdgeRelation.RELATES_TO,
        metadata={},
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_source_type_coverage_csv_empty_input_has_stable_header():
    assert export_source_type_coverage_csv([]) == (
        "source_type,source_count,unit_count,edge_count,average_confidence,missing_metadata_count\n"
    )


def test_source_type_coverage_csv_aggregates_multiple_source_types():
    units = [
        unit("a", SourceProject.MAX, "note", confidence=0.5, metadata={"title": "A"}),
        unit("b", "pinboard", "bookmark", confidence=0.8, metadata={"url": "https://example.com"}),
        unit("c", "readwise", "note", confidence=0.7, metadata={}),
        unit("d", SourceProject.MAX, "note", confidence="high", metadata={"x": 1}),
    ]
    edges = [edge("e1", "a", "b"), edge("e2", "source-c", "a"), edge("e3", "c", "d")]

    assert rows(export_source_type_coverage_csv(units, edges)) == [
        {
            "source_type": "bookmark",
            "source_count": "1",
            "unit_count": "1",
            "edge_count": "1",
            "average_confidence": "0.80",
            "missing_metadata_count": "0",
        },
        {
            "source_type": "note",
            "source_count": "2",
            "unit_count": "3",
            "edge_count": "3",
            "average_confidence": "0.60",
            "missing_metadata_count": "1",
        },
    ]


def test_source_type_coverage_csv_groups_missing_type_as_unknown_and_ignores_bad_confidence():
    units = [
        unit("a", None, "", confidence=True, metadata=None),
        unit("b", "Source B", None, confidence="unknown", metadata={"key": "value"}),
        unit("c", "Source B", " ", confidence=None, metadata="not metadata"),
    ]
    edges = [edge("e1", "a", "b"), edge("e2", "b", "c")]

    assert rows(export_source_type_coverage_csv(units, edges)) == [
        {
            "source_type": "Unknown",
            "source_count": "2",
            "unit_count": "3",
            "edge_count": "2",
            "average_confidence": "",
            "missing_metadata_count": "2",
        }
    ]


def test_source_type_coverage_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "coverage.csv"
    units = [unit("a", "Source A", "note", confidence=0.9, metadata={"a": 1})]

    expected = export_source_type_coverage_csv(units)
    stats = export_source_type_coverage_csv(units, None, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "edge_count": 0,
        "source_type_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }


def test_source_type_coverage_csv_is_deterministic_for_reversed_input():
    units = [
        unit("a", "Source B", "zeta", confidence=0.2, metadata={}),
        unit("b", "Source A", "alpha", confidence=0.4, metadata={}),
        unit("c", "Source A", "zeta", confidence=0.6, metadata={}),
    ]
    edges = [edge("e2", "a", "b"), edge("e1", "c", "a")]

    assert export_source_type_coverage_csv(units, edges) == export_source_type_coverage_csv(
        reversed(units), reversed(edges)
    )
