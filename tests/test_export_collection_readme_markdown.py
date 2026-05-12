from __future__ import annotations

from datetime import datetime, timezone

from graph.export import export_collection_readme_markdown
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


def edge(edge_id: str, source: str, target: str, relation=EdgeRelation.RELATES_TO):
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=source,
        to_unit_id=target,
        relation=relation,
        source=EdgeSource.MANUAL,
        created_at=TIME,
    )


def test_collection_readme_includes_summary_breakdowns_units_and_internal_edges():
    text = export_collection_readme_markdown(
        [unit("b", "Beta", tags=["AI"]), unit("a", "Alpha", tags=["AI", "Search"])],
        [edge("ab", "a", "b"), edge("external", "a", "missing")],
        title="Reading Set",
        generated_at="2026-05-01T00:00:00+00:00",
    )

    assert text.startswith("# Reading Set\n\nGenerated at: 2026-05-01T00:00:00+00:00\n")
    assert "| Units | 2 |" in text
    assert "| Edges | 1 |" in text
    assert "| max | 2 |" in text
    assert "| AI | 2 |" in text
    assert "| Search | 1 |" in text
    assert text.index("| a | Alpha | max | note | AI, Search |") < text.index("| b | Beta | max | note | AI |")
    assert "| a | relates_to | b | manual | 1 |" in text
    assert "missing" not in text


def test_collection_readme_is_deterministic_with_timestamp_override():
    units = [unit("b", "Beta"), unit("a", "Alpha")]
    edges = [edge("ab", "a", "b")]

    first = export_collection_readme_markdown(units, edges, generated_at="fixed")
    second = export_collection_readme_markdown(reversed(units), reversed(edges), generated_at="fixed")

    assert first == second


def test_collection_readme_can_omit_edges_and_write_stats(tmp_path):
    path = tmp_path / "README.md"
    stats = export_collection_readme_markdown(
        [unit("a", "Alpha"), unit("b", "Beta")],
        [edge("ab", "a", "b")],
        path,
        include_edges=False,
        generated_at="fixed",
    )

    text = path.read_text(encoding="utf-8")
    assert stats["unit_count"] == 2
    assert stats["edge_count"] == 0
    assert stats["bytes_written"] == path.stat().st_size
    assert "## Edges" not in text
