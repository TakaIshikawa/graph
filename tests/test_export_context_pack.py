from __future__ import annotations

from graph.export import export_context_pack
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


def unit(unit_id: str, title: str, content: str, tags: list[str] | None = None) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title,
        content=content,
        content_type=ContentType.INSIGHT,
        tags=tags or [],
    )


def test_export_context_pack_writes_summary_index_units_and_included_edges(tmp_path):
    units = [
        unit("unit-a", "Alpha", "Alpha content", ["solar", "storage"]),
        unit("unit-b", "Beta", "Beta content"),
        unit("unit-c", "Gamma", "Gamma content"),
    ]
    edges = [
        KnowledgeEdge(
            id="edge-b",
            from_unit_id="unit-b",
            to_unit_id="unit-c",
            relation=EdgeRelation.BUILDS_ON,
            source=EdgeSource.MANUAL,
            weight=0.75,
        ),
        KnowledgeEdge(
            id="edge-a",
            from_unit_id="unit-a",
            to_unit_id="unit-b",
            relation=EdgeRelation.RELATES_TO,
            source=EdgeSource.INFERRED,
        ),
        KnowledgeEdge(
            id="edge-out",
            from_unit_id="unit-a",
            to_unit_id="missing",
            relation=EdgeRelation.REFERENCES,
        ),
    ]

    path = tmp_path / "context.md"
    stats = export_context_pack(units, edges, path, title="Research Pack")
    text = path.read_text()

    assert stats["path"] == str(path)
    assert stats["units_included"] == 3
    assert stats["units_skipped"] == 0
    assert stats["edges_included"] == 2
    assert stats["edges_skipped"] == 1
    assert text.startswith("---\ntitle: \"Research Pack\"\n")
    assert "## Source Index" in text
    assert "1. [Alpha](#unit-unit-a) - `unit-a` - max/insight - `source-unit-a`" in text
    assert "### Alpha" in text
    assert "- Tags: `solar`, `storage`" in text
    assert "Alpha content" in text
    assert "`unit-b` --`builds_on`--> `unit-c`" in text
    assert "`unit-a` --`relates_to`--> `unit-b`" in text
    assert "edge-out" not in text


def test_export_context_pack_max_chars_keeps_complete_unit_sections(tmp_path):
    units = [
        unit("unit-a", "Alpha", "short content"),
        unit("unit-b", "Beta", "second content"),
        unit("unit-c", "Gamma", "third content"),
    ]
    path = tmp_path / "limited.md"
    stats = export_context_pack(units, [], path, max_chars=850)
    text = path.read_text()

    assert len(text) <= 850
    assert stats["units_included"] == 2
    assert stats["units_skipped"] == 1
    assert "### Alpha" in text
    assert "### Beta" in text
    assert "### Gamma" not in text
    assert "third content" not in text


def test_export_context_pack_exports_only_edges_between_included_units_with_limit(tmp_path):
    units = [
        unit("unit-a", "Alpha", "short content"),
        unit("unit-b", "Beta", "second content"),
        unit("unit-c", "Gamma", "third content"),
    ]
    edges = [
        KnowledgeEdge(
            id="edge-in",
            from_unit_id="unit-a",
            to_unit_id="unit-b",
            relation=EdgeRelation.RELATES_TO,
        ),
        KnowledgeEdge(
            id="edge-skipped",
            from_unit_id="unit-b",
            to_unit_id="unit-c",
            relation=EdgeRelation.RELATES_TO,
        ),
    ]
    path = tmp_path / "limited.md"
    stats = export_context_pack(units, edges, path, max_chars=850)
    text = path.read_text()

    assert stats["units_included"] == 2
    assert stats["edges_included"] == 1
    assert stats["edges_skipped"] == 1
    assert "edge-in" in text
    assert "edge-skipped" not in text


def test_export_context_pack_escapes_markdown_title_characters(tmp_path):
    units = [
        unit(
            "unit-a/b",
            "# API [draft](v2) \\ notes",
            "Content with ``` fenced text",
        )
    ]

    path = tmp_path / "context.md"
    export_context_pack(units, [], path)
    text = path.read_text()

    assert r"[# API \[draft\]\(v2\) \\ notes](#unit-unit-a-b)" in text
    assert r"### \# API [draft](v2) \\ notes" in text
    assert "````\nContent with ``` fenced text\n````" in text
