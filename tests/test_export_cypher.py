from __future__ import annotations

from datetime import datetime, timezone

from graph.export import export_graph_cypher
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


UNIT_TIME = datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)
EDGE_TIME = datetime(2026, 5, 1, 12, 30, tzinfo=timezone.utc)


def unit(
    unit_id: str,
    title: str,
    content: str,
    tags: list[str] | None = None,
    *,
    metadata: dict | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title,
        content=content,
        content_type=ContentType.INSIGHT,
        metadata=metadata or {},
        tags=tags or [],
        confidence=0.8,
        utility_score=0.6,
        created_at=UNIT_TIME,
        ingested_at=UNIT_TIME,
        updated_at=UNIT_TIME,
    )


def edge(
    edge_id: str,
    from_unit_id: str,
    to_unit_id: str,
    relation: EdgeRelation = EdgeRelation.RELATES_TO,
    *,
    source: EdgeSource = EdgeSource.INFERRED,
    weight: float = 1.0,
    metadata: dict | None = None,
) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=relation,
        source=source,
        weight=weight,
        metadata=metadata or {},
        created_at=EDGE_TIME,
    )


def test_export_graph_cypher_writes_units_edges_and_stats(tmp_path):
    path = tmp_path / "nested" / "graph.cypher"

    stats = export_graph_cypher(
        [
            unit(
                "unit-a",
                "Alpha",
                "Alpha content",
                ["solar", "storage"],
                metadata={"reviewed_at": UNIT_TIME, "project": SourceProject.MAX},
            ),
            unit("unit-b", "Beta", "Beta content"),
        ],
        [
            edge(
                "edge-a",
                "unit-a",
                "unit-b",
                EdgeRelation.BUILDS_ON,
                source=EdgeSource.MANUAL,
                weight=0.75,
                metadata={"observed_at": EDGE_TIME, "notes": ["curated"]},
            )
        ],
        path,
    )

    text = path.read_text(encoding="utf-8")

    assert stats == {
        "path": str(path),
        "units_exported": 2,
        "edges_exported": 1,
        "bytes_written": len(text.encode("utf-8")),
        "label": "KnowledgeUnit",
    }
    assert text.startswith(
        "CREATE CONSTRAINT KnowledgeUnit_id_unique IF NOT EXISTS\n"
        "FOR (unit:KnowledgeUnit) REQUIRE unit.id IS UNIQUE;\n\n"
    )
    assert "MERGE (unit:KnowledgeUnit {id: 'unit-a'})\n" in text
    assert (
        "tags: ['solar', 'storage'], title: 'Alpha', updated_at: "
        "'2026-05-01T10:15:00+00:00', utility_score: 0.6"
    ) in text
    assert (
        "metadata: {project: 'max', reviewed_at: '2026-05-01T10:15:00+00:00'}"
    ) in text
    assert "MERGE (from)-[rel:BUILDS_ON {id: 'edge-a'}]->(to)\n" in text
    assert "metadata: {notes: ['curated'], observed_at: '2026-05-01T12:30:00+00:00'}" in text
    assert text.endswith("\n")


def test_export_graph_cypher_escapes_strings_and_nested_metadata(tmp_path):
    path = tmp_path / "graph.cypher"

    export_graph_cypher(
        [
            unit(
                "unit-'a",
                "Title with 'quote'\nnew line",
                "Content with \\ slash\r\nand\ttab",
                ["alpha\nbeta", "quote's"],
                metadata={
                    "complex key": {"nested\nkey": "value's", "slash": "a\\b"},
                    "tick`key": "tick",
                    "enabled": True,
                    "missing": None,
                },
            )
        ],
        [
            edge(
                "edge-'a",
                "unit-'a",
                "unit-'a",
                metadata={"reason": "line\nbreak", "ok?": False},
            )
        ],
        path,
    )

    text = path.read_text(encoding="utf-8")

    assert "MERGE (unit:KnowledgeUnit {id: 'unit-\\'a'})" in text
    assert "title: 'Title with \\'quote\\'\\nnew line'" in text
    assert "content: 'Content with \\\\ slash\\r\\nand\\ttab'" in text
    assert "tags: ['alpha\\nbeta', 'quote\\'s']" in text
    assert "`complex key`: {`nested\\nkey`: 'value\\'s', slash: 'a\\\\b'}" in text
    assert "`tick``key`: 'tick'" in text
    assert "enabled: true" in text
    assert "missing: null" in text
    assert "metadata: {`ok?`: false, reason: 'line\\nbreak'}" in text


def test_export_graph_cypher_sorts_units_and_edges_deterministically(tmp_path):
    first_path = tmp_path / "first.cypher"
    second_path = tmp_path / "second.cypher"
    units = [
        unit("unit-b", "Beta", "Beta content"),
        unit("unit-a", "Alpha", "Alpha content"),
    ]
    edges = [
        edge("edge-c", "unit-b", "unit-a", EdgeRelation.RELATES_TO),
        edge("edge-b", "unit-a", "unit-b", EdgeRelation.REFERENCES),
        edge("edge-a", "unit-a", "unit-b", EdgeRelation.BUILDS_ON),
    ]

    export_graph_cypher(units, edges, first_path)
    export_graph_cypher(reversed(units), reversed(edges), second_path)
    text = first_path.read_text(encoding="utf-8")

    assert first_path.read_text(encoding="utf-8") == second_path.read_text(encoding="utf-8")
    assert text.index("id: 'unit-a'") < text.index("id: 'unit-b'")
    assert text.index("[rel:BUILDS_ON") < text.index("[rel:REFERENCES")
    assert text.index("[rel:REFERENCES") < text.index("[rel:RELATES_TO")


def test_export_graph_cypher_normalizes_custom_label_and_empty_export(tmp_path):
    path = tmp_path / "empty" / "graph.cypher"

    stats = export_graph_cypher([], [], path, batch_label="42 Notes!")

    assert stats == {
        "path": str(path),
        "units_exported": 0,
        "edges_exported": 0,
        "bytes_written": path.stat().st_size,
        "label": "_42_Notes",
    }
    assert path.read_text(encoding="utf-8") == (
        "CREATE CONSTRAINT _42_Notes_id_unique IF NOT EXISTS\n"
        "FOR (unit:_42_Notes) REQUIRE unit.id IS UNIQUE;\n"
    )
