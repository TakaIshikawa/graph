from __future__ import annotations

from pathlib import Path

from graph.adapters.opml import OpmlAdapter
from graph.types.enums import EdgeRelation, EdgeSource


def write_opml(path: Path, body: str) -> Path:
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
        <opml version="2.0">
          <body>
            {body}
          </body>
        </opml>
        """,
        encoding="utf-8",
    )
    return path


def title_by_source_id(result) -> dict[str, str]:
    return {unit.source_id: unit.title for unit in result.units}


def edge_title_pairs(result) -> list[tuple[str, str]]:
    titles = title_by_source_id(result)
    return [(titles[edge.from_unit_id], titles[edge.to_unit_id]) for edge in result.edges]


def test_nested_opml_outlines_emit_contains_edges_for_each_level_and_sibling_feed(tmp_path):
    opml_path = write_opml(
        tmp_path / "feeds.opml",
        """
        <outline text="Research">
          <outline text="AI">
            <outline text="AI Feed" type="rss" xmlUrl="https://example.com/ai.xml" />
            <outline text="ML Feed" type="rss" xmlUrl="https://example.com/ml.xml" />
          </outline>
          <outline text="Systems">
            <outline text="Distributed Systems" htmlUrl="https://example.com/systems" />
          </outline>
        </outline>
        """,
    )

    result = OpmlAdapter(path=str(opml_path)).ingest()

    assert [unit.title for unit in result.units] == [
        "Research",
        "AI",
        "AI Feed",
        "ML Feed",
        "Systems",
        "Distributed Systems",
    ]
    assert edge_title_pairs(result) == [
        ("Research", "AI"),
        ("AI", "AI Feed"),
        ("AI", "ML Feed"),
        ("Research", "Systems"),
        ("Systems", "Distributed Systems"),
    ]
    assert all(edge.relation == EdgeRelation.CONTAINS for edge in result.edges)
    assert all(edge.source == EdgeSource.SOURCE for edge in result.edges)
    assert all(edge.metadata["source_project"] == "opml" for edge in result.edges)
    assert all(edge.metadata["from_entity_type"] == "outline" for edge in result.edges)
    assert all(edge.metadata["to_entity_type"] == "outline" for edge in result.edges)
    edge_keys = {(edge.from_unit_id, edge.to_unit_id, edge.relation) for edge in result.edges}
    assert len(edge_keys) == len(result.edges)


def test_opml_hierarchy_edge_ids_are_deterministic(tmp_path):
    opml_path = write_opml(
        tmp_path / "feeds.opml",
        """
        <outline text="Root">
          <outline text="Child">
            <outline text="Feed" xmlUrl="https://example.com/feed.xml" />
          </outline>
        </outline>
        """,
    )

    first = OpmlAdapter(path=str(opml_path)).ingest()
    second = OpmlAdapter(path=str(opml_path)).ingest()

    assert [edge.id for edge in first.edges] == [edge.id for edge in second.edges]
    assert all(edge.id.startswith("opml-contains-") for edge in first.edges)


def test_flat_opml_ingestion_keeps_units_without_edges(tmp_path):
    opml_path = write_opml(
        tmp_path / "flat.opml",
        """
        <outline text="AI Feed" type="rss" xmlUrl="https://example.com/ai.xml" />
        <outline text="Systems Feed" type="rss" xmlUrl="https://example.com/systems.xml" />
        """,
    )

    result = OpmlAdapter(path=str(opml_path)).ingest()

    assert [unit.title for unit in result.units] == ["AI Feed", "Systems Feed"]
    assert result.edges == []


def test_opml_entity_type_filter_excludes_units_and_edges(tmp_path):
    opml_path = write_opml(
        tmp_path / "feeds.opml",
        """
        <outline text="Research">
          <outline text="AI Feed" xmlUrl="https://example.com/ai.xml" />
        </outline>
        """,
    )

    result = OpmlAdapter(path=str(opml_path)).ingest(entity_types=["jsonl_record"])

    assert result.units == []
    assert result.edges == []
