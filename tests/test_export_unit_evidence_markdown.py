from __future__ import annotations

from graph.export import export_unit_evidence_markdown
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


def unit(
    unit_id: str,
    title: str | None,
    *,
    source_project: SourceProject | str | None = SourceProject.MAX,
    source_id: str | None = None,
    metadata: dict | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=source_id,
        source_entity_type="note",
        title=title,
        content=f"{title or unit_id} content",
        content_type=ContentType.INSIGHT,
        metadata=metadata or {},
    )


def edge(
    edge_id: str,
    from_unit_id: str,
    to_unit_id: str,
    *,
    relation: EdgeRelation | str | None = EdgeRelation.REFERENCES,
    metadata: dict | None = None,
) -> KnowledgeEdge:
    return KnowledgeEdge.model_construct(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=relation,
        source=EdgeSource.SOURCE,
        weight=1.0,
        metadata=metadata or {},
    )


def test_unit_evidence_markdown_lists_multiple_sources_and_evidence_edges():
    markdown = export_unit_evidence_markdown(
        [
            unit("unit-a", "Alpha", source_id="alpha-source"),
            unit(
                "unit-b",
                "Beta",
                source_project="docs",
                source_id="beta-source",
                metadata={"source_name": "Docs Team"},
            ),
            unit("unit-c", "Gamma", source_project="web", source_id="gamma-source"),
        ],
        [
            edge(
                "edge-a",
                "unit-a",
                "unit-b",
                metadata={
                    "confidence": 0.86,
                    "label": "Audit note",
                    "url": "https://example.com/audit",
                },
            ),
            edge(
                "edge-b",
                "unit-c",
                "unit-a",
                relation=EdgeRelation.BUILDS_ON,
                metadata={"confidence": 0.6, "evidence_url": "https://example.com/source"},
            ),
        ],
    )

    assert markdown.startswith("# Unit Evidence\n")
    assert "## Alpha (`unit-a`)" in markdown
    assert "- Unit id: `unit-a`" in markdown
    assert "- Unit name: Alpha" in markdown
    assert "- Source: max (`alpha-source`)" in markdown
    assert (
        "- `builds_on`; incoming; source: web (`gamma-source`); "
        "confidence: 0.60; evidence: https://example.com/source; edge: `edge-b`"
    ) in markdown
    assert (
        "- `references`; outgoing; source: Docs Team (`beta-source`); "
        "confidence: 0.86; evidence: Audit note (https://example.com/audit); edge: `edge-a`"
    ) in markdown


def test_unit_evidence_markdown_represents_units_with_no_evidence():
    markdown = export_unit_evidence_markdown(
        [unit("unit-a", "Alpha", source_id="source-a")],
        [],
    )

    assert markdown == (
        "# Unit Evidence\n"
        "\n"
        "## Alpha (`unit-a`)\n"
        "\n"
        "- Unit id: `unit-a`\n"
        "- Unit name: Alpha\n"
        "- Source: max (`source-a`)\n"
        "\n"
        "### Evidence\n"
        "\n"
        "_No evidence._\n"
    )


def test_unit_evidence_markdown_handles_missing_optional_metadata():
    markdown = export_unit_evidence_markdown(
        [
            unit("unit-a", None, source_project=None, source_id=None, metadata=None),
            unit("unit-b", "", source_project="", source_id="", metadata={}),
        ],
        [edge("", "unit-a", "unit-b", relation=None, metadata=None)],
    )

    assert "## unit-a (`unit-a`)" in markdown
    assert "- Source: Unknown (`Unknown`)" in markdown
    assert "- `Unknown`; outgoing; source: Unknown (`Unknown`)" in markdown
    assert "confidence:" not in markdown
    assert "evidence:" not in markdown


def test_unit_evidence_markdown_is_deterministic_for_reversed_input():
    units = [
        unit("unit-b", "Beta", source_project="docs", source_id="source-b"),
        unit("unit-a", "Alpha", source_project="notes", source_id="source-a"),
        unit("unit-c", "Alpha", source_project="web", source_id="source-c"),
    ]
    edges = [
        edge("edge-b", "unit-b", "unit-a", relation=EdgeRelation.REFERENCES),
        edge("edge-a", "unit-a", "unit-c", relation=EdgeRelation.BUILDS_ON),
    ]

    first = export_unit_evidence_markdown(units, edges)
    second = export_unit_evidence_markdown(reversed(units), reversed(edges))

    assert first == second
    assert first.index("## Alpha (`unit-a`)") < first.index("## Alpha (`unit-c`)")
    assert first.index("## Alpha (`unit-c`)") < first.index("## Beta (`unit-b`)")


def test_unit_evidence_markdown_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "unit-evidence.md"
    units = [unit("unit-a", "Alpha", source_id="source-a")]
    edges = [edge("edge-a", "unit-a", "missing")]

    expected = export_unit_evidence_markdown(units, edges)
    stats = export_unit_evidence_markdown(units, edges, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "edge_count": 1,
        "evidence_edge_count": 1,
        "bytes_written": path.stat().st_size,
    }
