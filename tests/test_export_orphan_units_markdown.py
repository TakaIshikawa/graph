from __future__ import annotations

from datetime import datetime, timezone

import pytest

from graph.export import export_orphan_units_markdown
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit

TIME = datetime(2026, 5, 1, 10, 0, tzinfo=timezone.utc)


def unit(
    unit_id: str,
    title: str,
    *,
    source=SourceProject.MAX,
    kind="note",
    tags=None,
    content=None,
):
    return KnowledgeUnit(
        id=unit_id,
        source_project=source,
        source_id=f"source-{unit_id}",
        source_entity_type=kind,
        title=title,
        content=content or f"{title} content",
        content_type=ContentType.INSIGHT,
        metadata={},
        tags=tags or [],
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
        source=EdgeSource.INFERRED,
        created_at=TIME,
    )


def test_orphan_units_markdown_lists_only_units_without_any_edge_endpoint():
    text = export_orphan_units_markdown(
        [unit("a", "Alpha"), unit("b", "Beta"), unit("c", "Gamma")],
        [edge("ab", "a", "b")],
    )

    assert "Gamma" in text
    assert "`c`" in text
    assert "Alpha" not in text
    assert "Beta" not in text


def test_orphan_units_markdown_is_deterministic_for_shuffled_inputs():
    units = [
        unit("c", "Gamma", source=SourceProject.KINDLE, kind="quote"),
        unit("a", "Alpha", source=SourceProject.MAX, kind="note"),
        unit("b", "Beta", source=SourceProject.MAX, kind="note"),
    ]
    edges = [edge("za", "z", "a"), edge("xy", "x", "y")]

    first = export_orphan_units_markdown(units, edges)
    second = export_orphan_units_markdown(reversed(units), reversed(edges))

    assert first == second
    assert first.index("## kindle") < first.index("## max")
    assert first.index("Gamma") < first.index("Beta")


def test_orphan_units_markdown_supports_grouping_by_type_and_none():
    units = [
        unit("a", "Alpha", kind="note"),
        unit("b", "Beta", kind="quote"),
    ]

    by_type = export_orphan_units_markdown(units, [], group_by="type")
    no_groups = export_orphan_units_markdown(units, [], group_by="none")

    assert "## note" in by_type
    assert "## quote" in by_type
    assert "## " not in no_groups


def test_orphan_units_markdown_writes_path_and_returns_stats(tmp_path):
    path = tmp_path / "orphans.md"
    stats = export_orphan_units_markdown(
        [unit("a", "Alpha"), unit("b", "Beta", tags=["Review"])],
        [edge("external", "external", "a")],
        path,
    )

    output = path.read_text(encoding="utf-8")
    assert stats["units_scanned"] == 2
    assert stats["orphans_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
    assert "Beta" in output
    assert "Alpha" not in output


def test_orphan_units_markdown_rejects_unknown_grouping():
    with pytest.raises(ValueError, match="group_by"):
        export_orphan_units_markdown([], [], group_by="source_project")
