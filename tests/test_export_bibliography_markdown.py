from __future__ import annotations

from datetime import datetime, timezone

from graph.export import export_units_to_bibliography_markdown
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit

UNIT_TIME = datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)


def unit(
    unit_id: str,
    *,
    title: str | None = None,
    content: str = "",
    metadata: dict | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.CSL_JSON,
        source_id=f"source-{unit_id}",
        source_entity_type="publication",
        title=title or f"Title {unit_id}",
        content=content,
        content_type=ContentType.FINDING,
        metadata=metadata or {},
        created_at=UNIT_TIME,
        ingested_at=UNIT_TIME,
        updated_at=UNIT_TIME,
    )


def test_export_units_to_bibliography_markdown_renders_citation_fields():
    text = export_units_to_bibliography_markdown(
        [
            unit(
                "unit-a",
                title="Fallback title",
                metadata={
                    "authors": ["Smith, Ada", {"given": "Grace", "family": "Doe"}],
                    "issued": "2025-04-24",
                    "title": "Graph Interchange",
                    "container_title": "Journal of Graphs",
                    "publisher": "Example Press",
                    "doi": "10.1234/example",
                    "ISBN": "978-1-234",
                    "source_url": "https://example.test/paper",
                    "abstract": "A study of graph export formats.",
                },
            )
        ]
    )

    assert text == (
        "# Bibliography\n"
        "\n"
        "- Smith, Ada and Doe, Grace (2025). Graph Interchange. *Journal of Graphs*. "
        "Example Press. DOI: 10.1234/example. ISBN: 978-1-234. https://example.test/paper\n"
        "  - A study of graph export formats.\n"
    )


def test_export_units_to_bibliography_markdown_degrades_to_title_and_url():
    text = export_units_to_bibliography_markdown(
        [unit("unit-a", title="Plain Note", metadata={"external_url": "https://example.test/note"})]
    )

    assert text == "# Bibliography\n\n- Plain Note. https://example.test/note\n"


def test_export_units_to_bibliography_markdown_sorts_units_deterministically():
    text = export_units_to_bibliography_markdown(
        [
            unit("unit-b", title="Beta"),
            unit("unit-a", title="Alpha"),
        ]
    )

    assert text.index("Alpha") < text.index("Beta")


def test_export_units_to_bibliography_markdown_writes_file(tmp_path):
    path = tmp_path / "nested" / "bibliography.md"

    stats = export_units_to_bibliography_markdown([unit("unit-a", title="Alpha")], path)

    assert path.read_text(encoding="utf-8") == "# Bibliography\n\n- Alpha.\n"
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "bytes_written": path.stat().st_size,
    }
