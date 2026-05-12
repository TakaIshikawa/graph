from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_source_diversity_csv
from graph.types.enums import ContentType, EdgeRelation, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


def unit(
    unit_id: str,
    source_project: SourceProject | str | None,
    *,
    source_entity_type: str | None = "note",
    title: str | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type=source_entity_type,
        title=title or f"Title {unit_id}",
        content="content",
        content_type=ContentType.INSIGHT,
        metadata={},
    )


def edge(edge_id: str, from_unit_id: str, to_unit_id: str) -> KnowledgeEdge:
    return KnowledgeEdge.model_construct(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=EdgeRelation.REFERENCES,
        metadata={},
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_source_diversity_csv_empty_input_has_stable_header():
    assert export_unit_source_diversity_csv([]) == (
        "unit_id,unit_name,contributing_source_count,distinct_source_type_count,source_types,"
        "evidence_edge_count,top_source,top_source_evidence_count\n"
    )


def test_unit_source_diversity_csv_counts_single_and_multiple_sources():
    units = [
        unit("target", "writing", title="Target"),
        unit("a", SourceProject.MAX, source_entity_type="note"),
        unit("b", "pinboard", source_entity_type="bookmark"),
        unit("c", SourceProject.MAX, source_entity_type="note"),
        unit("solo", "personal", title="Solo"),
    ]
    edges = [
        edge("e1", "a", "target"),
        edge("e2", "target", "b"),
        edge("e3", "source-c", "target"),
        edge("e4", "solo", "a"),
    ]

    parsed = rows(export_unit_source_diversity_csv(units, edges))

    assert parsed[4] == {
        "unit_id": "target",
        "unit_name": "Target",
        "contributing_source_count": "2",
        "distinct_source_type_count": "2",
        "source_types": "bookmark; note",
        "evidence_edge_count": "3",
        "top_source": "max",
        "top_source_evidence_count": "2",
    }
    assert parsed[3]["unit_id"] == "solo"
    assert parsed[3]["contributing_source_count"] == "1"
    assert parsed[3]["top_source"] == "max"


def test_unit_source_diversity_csv_handles_missing_source_type_and_unknown_endpoint():
    units = [
        unit("a", "Source A", source_entity_type=""),
        unit("b", None, source_entity_type=None),
    ]
    edges = [edge("e1", "a", "b"), edge("e2", "a", "missing")]

    parsed = rows(export_unit_source_diversity_csv(units, edges))

    assert parsed == [
        {
            "unit_id": "a",
            "unit_name": "Title a",
            "contributing_source_count": "1",
            "distinct_source_type_count": "0",
            "source_types": "",
            "evidence_edge_count": "2",
            "top_source": "Unknown",
            "top_source_evidence_count": "2",
        },
        {
            "unit_id": "b",
            "unit_name": "Title b",
            "contributing_source_count": "1",
            "distinct_source_type_count": "0",
            "source_types": "",
            "evidence_edge_count": "1",
            "top_source": "Source A",
            "top_source_evidence_count": "1",
        },
    ]


def test_unit_source_diversity_csv_ties_sort_top_source_deterministically():
    units = [
        unit("target", "target"),
        unit("z", "Source Z", source_entity_type="article"),
        unit("a", "Source A", source_entity_type="note"),
    ]
    edges = [edge("e2", "target", "z"), edge("e1", "a", "target")]

    parsed = rows(export_unit_source_diversity_csv(reversed(units), reversed(edges)))

    assert [row["unit_id"] for row in parsed] == ["a", "target", "z"]
    assert parsed[1]["top_source"] == "Source A"
    assert parsed[1]["source_types"] == "article; note"


def test_unit_source_diversity_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "reports" / "unit-source-diversity.csv"
    units = [unit("a", "Source A"), unit("b", "Source B")]
    edges = [edge("e1", "a", "b")]

    expected = export_unit_source_diversity_csv(units, edges)
    stats = export_unit_source_diversity_csv(units, edges, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 2,
        "edge_count": 1,
        "rows_exported": 2,
        "bytes_written": path.stat().st_size,
    }
