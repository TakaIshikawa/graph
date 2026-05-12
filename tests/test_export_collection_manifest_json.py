from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.export import export_collection_manifest_json
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit

TIME = datetime(2026, 5, 1, 10, 0, tzinfo=timezone.utc)


def unit(unit_id: str, title: str, *, source=SourceProject.MAX, tags=None):
    return KnowledgeUnit(
        id=unit_id,
        source_project=source,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=title,
        content="content",
        content_type=ContentType.INSIGHT,
        tags=tags or [],
        metadata={"rank": unit_id},
        created_at=TIME,
        ingested_at=TIME,
        updated_at=TIME,
    )


def edge(edge_id: str, source: str, target: str):
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=source,
        to_unit_id=target,
        relation=EdgeRelation.RELATES_TO,
        source=EdgeSource.MANUAL,
        created_at=TIME,
    )


def test_collection_manifest_includes_summary_units_and_internal_edges():
    data = json.loads(
        export_collection_manifest_json(
            [unit("b", "Beta", tags=["AI"]), unit("a", "Alpha", tags=["AI", "Search"])],
            [edge("ab", "a", "b"), edge("external", "a", "missing")],
            title="Reading Set",
            generated_at="2026-05-01T00:00:00+00:00",
        )
    )

    assert data["schema_version"] == "1.0"
    assert data["title"] == "Reading Set"
    assert data["unit_count"] == 2
    assert data["edge_count"] == 1
    assert data["sources"] == {"max": 2}
    assert data["tags"] == {"AI": 2, "Search": 1}
    assert [unit["id"] for unit in data["units"]] == ["a", "b"]
    assert data["edges"][0]["id"] == "ab"


def test_collection_manifest_is_deterministic_with_timestamp_override():
    units = [unit("b", "Beta"), unit("a", "Alpha")]
    edges = [edge("ab", "a", "b")]

    first = export_collection_manifest_json(units, edges, generated_at="fixed")
    second = export_collection_manifest_json(reversed(units), reversed(edges), generated_at="fixed")

    assert first == second


def test_collection_manifest_can_omit_edges_and_write_stats(tmp_path):
    path = tmp_path / "manifest.json"
    stats = export_collection_manifest_json(
        [unit("a", "Alpha"), unit("b", "Beta")],
        [edge("ab", "a", "b")],
        path,
        include_edges=False,
        generated_at="fixed",
    )
    data = json.loads(path.read_text(encoding="utf-8"))

    assert stats["unit_count"] == 2
    assert stats["edge_count"] == 0
    assert data["edges"] == []
