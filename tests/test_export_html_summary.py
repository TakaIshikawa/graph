from __future__ import annotations

from datetime import datetime, timezone

from graph.export import export_graph_html_summary
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


def unit(
    unit_id: str,
    title: str,
    *,
    source_project: SourceProject = SourceProject.MAX,
    content_type: ContentType = ContentType.INSIGHT,
    tags: list[str] | None = None,
    created_at: datetime | None = None,
) -> KnowledgeUnit:
    timestamp = created_at or datetime(2026, 5, 1, 12, tzinfo=timezone.utc)
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title,
        content=f"{title} content",
        content_type=content_type,
        tags=tags or [],
        created_at=timestamp,
        ingested_at=timestamp,
        updated_at=timestamp,
    )


def edge(edge_id: str, source: str, target: str) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=source,
        to_unit_id=target,
        relation=EdgeRelation.RELATES_TO,
        source=EdgeSource.INFERRED,
    )


def test_export_graph_html_summary_writes_self_contained_report(tmp_path):
    output_path = tmp_path / "reports" / "summary.html"
    units = [
        unit("unit-a", "Alpha", tags=["solar", "storage"]),
        unit(
            "unit-b",
            "Beta",
            source_project=SourceProject.PRESENCE,
            content_type=ContentType.FINDING,
            tags=["solar"],
            created_at=datetime(2026, 5, 2, 9, tzinfo=timezone.utc),
        ),
    ]
    edges = [edge("edge-a-b", "unit-a", "unit-b")]

    stats = export_graph_html_summary(units, edges, output_path)
    html = output_path.read_text(encoding="utf-8")

    assert stats == {
        "path": str(output_path),
        "units_exported": 2,
        "edges_exported": 1,
        "bytes_written": output_path.stat().st_size,
    }
    assert html.startswith("<!doctype html>")
    assert "<style>" in html
    assert "<strong>2</strong>" in html
    assert "<strong>1</strong>" in html
    assert "Source Breakdown" in html
    assert "Content Types" in html
    assert "Top Tags" in html
    assert "Recent Units" in html
    assert "<td>presence</td>" in html
    assert "<td>finding</td>" in html
    assert "<td>solar</td><td>2</td>" in html
    assert "Beta" in html


def test_export_graph_html_summary_returns_html_without_path():
    html = export_graph_html_summary([unit("unit-a", "Alpha")], [])

    assert isinstance(html, str)
    assert "Graph Summary" in html
    assert "<strong>1</strong>" in html


def test_export_graph_html_summary_escapes_user_text():
    html = export_graph_html_summary(
        [
            unit(
                "unit-a",
                '<script>alert("x")</script>',
                source_project=SourceProject.MAX,
                tags=["<tag>"],
            )
        ],
        [],
    )

    assert '<script>alert("x")</script>' not in html
    assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;" in html
    assert "&lt;tag&gt;" in html


def test_export_graph_html_summary_validates_limits():
    for value in (-1, True, "bad"):
        try:
            export_graph_html_summary([], [], recent_limit=value)  # type: ignore[arg-type]
        except ValueError as exc:
            assert str(exc) == "recent_limit must be a non-negative integer"
        else:
            raise AssertionError("expected ValueError")
