from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_backlink_density_csv
from graph.types.models import KnowledgeEdge, KnowledgeUnit


def unit(unit_id: str, title: str | None = None) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project="Project",
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=title or f"Title {unit_id}",
        content="content",
        metadata={},
        tags=[],
    )


def edge(source: str, target: str, relation: str = "related") -> KnowledgeEdge:
    return KnowledgeEdge.model_construct(from_unit_id=source, to_unit_id=target, relation=relation)


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_backlink_density_csv_counts_inbound_outbound_and_relation_types():
    text = export_unit_backlink_density_csv(
        [unit("b", "Beta"), unit("a", "Alpha"), unit("c", "Gamma")],
        [
            edge("a", "b", "supports"),
            edge("b", "a", "related"),
            edge("a", "b", "related"),
            {"source_unit_id": "x", "target_unit_id": "a", "relation_type": "mentions"},
        ],
    )

    assert rows(text) == [
        {
            "unit_id": "a",
            "title": "Alpha",
            "source": "Project",
            "inbound_count": "2",
            "outbound_count": "2",
            "total_degree": "4",
            "relation_types": "mentions; related; supports",
        },
        {
            "unit_id": "b",
            "title": "Beta",
            "source": "Project",
            "inbound_count": "2",
            "outbound_count": "1",
            "total_degree": "3",
            "relation_types": "related; supports",
        },
        {
            "unit_id": "c",
            "title": "Gamma",
            "source": "Project",
            "inbound_count": "0",
            "outbound_count": "0",
            "total_degree": "0",
            "relation_types": "",
        },
    ]


def test_unit_backlink_density_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "density.csv"
    units = [unit("a"), unit("b")]
    relations = [edge("a", "b")]

    expected = export_unit_backlink_density_csv(units, relations)
    stats = export_unit_backlink_density_csv(units, relations, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["rows_exported"] == 2
    assert stats["relation_count"] == 1
