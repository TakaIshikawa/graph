from __future__ import annotations

from datetime import datetime, timezone

from graph.export import export_units_to_ris
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


UNIT_TIME = datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)


def unit(
    unit_id: str,
    *,
    source_project: SourceProject | str = SourceProject.BIBTEX,
    source_id: str | None = None,
    title: str | None = None,
    content: str = "Content body.",
    tags: list[str] | None = None,
    metadata: dict | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=source_id or f"source-{unit_id}",
        source_entity_type="publication",
        title=title or f"Title {unit_id}",
        content=content,
        content_type=ContentType.FINDING,
        tags=tags or [],
        created_at=UNIT_TIME,
        ingested_at=UNIT_TIME,
        updated_at=UNIT_TIME,
        metadata=metadata or {},
    )


def test_export_units_to_ris_maps_common_bibliography_fields():
    text = export_units_to_ris(
        [
            unit(
                "unit-a",
                title="Graph Interchange",
                content="Fallback content.",
                tags=["graphs", "reference managers"],
                metadata={
                    "authors": ["Smith, Ada", "Doe, Grace"],
                    "source_url": "https://example.test/paper",
                    "published_at": "2025-04-24T12:00:00Z",
                    "abstract": "A study of graph export formats.",
                },
            )
        ]
    )

    assert text == (
        "TY  - ELEC\n"
        "TI  - Graph Interchange\n"
        "AU  - Smith, Ada\n"
        "AU  - Doe, Grace\n"
        "PY  - 2025\n"
        "UR  - https://example.test/paper\n"
        "KW  - graphs\n"
        "KW  - reference managers\n"
        "AB  - A study of graph export formats.\n"
        "ER  - \n"
    )


def test_export_units_to_ris_includes_complete_record_with_missing_metadata():
    text = export_units_to_ris([unit("unit-a", title="Untyped note", content="Body only.")])

    assert text == (
        "TY  - ELEC\n"
        "TI  - Untyped note\n"
        "AB  - Body only.\n"
        "ER  - \n"
    )


def test_export_units_to_ris_sorts_non_sequence_units_deterministically():
    units = (
        item
        for item in [
            unit("unit-c", source_project=SourceProject.RIS, source_id="2", title="Beta"),
            unit("unit-a", source_project=SourceProject.BIBTEX, source_id="2", title="Beta"),
            unit("unit-b", source_project=SourceProject.BIBTEX, source_id="1", title="Zeta"),
            unit("unit-d", source_project=SourceProject.BIBTEX, source_id="1", title="Alpha"),
        ]
    )

    text = export_units_to_ris(units)

    assert text.index("TI  - Alpha") < text.index("TI  - Zeta")
    assert text.index("TI  - Zeta") < text.index("TI  - Beta")
    assert text.rindex("TI  - Beta") > text.index("TI  - Beta")


def test_export_units_to_ris_normalizes_multiple_authors_tags_and_multiline_values():
    text = export_units_to_ris(
        [
            unit(
                "unit-a",
                title="Title\nwith line break",
                content="Content\nwith\r\nline\tbreak",
                tags=["zeta", "alpha", "alpha"],
                metadata={
                    "creator": [{"name": "Chen, Rae"}, {"literal": "Olsen, Kim"}],
                    "url": "https://example.test/\nrecord",
                    "year": 2024,
                },
            )
        ]
    )

    assert text == (
        "TY  - ELEC\n"
        "TI  - Title with line break\n"
        "AU  - Chen, Rae\n"
        "AU  - Olsen, Kim\n"
        "PY  - 2024\n"
        "UR  - https://example.test/ record\n"
        "KW  - alpha\n"
        "KW  - zeta\n"
        "AB  - Content with line break\n"
        "ER  - \n"
    )


def test_export_units_to_ris_emits_normalized_scalar_doi():
    text = export_units_to_ris(
        [
            unit(
                "unit-a",
                metadata={
                    "doi": " https://doi.org/10.1234/example ",
                },
            )
        ]
    )

    assert "DO  - 10.1234/example\n" in text


def test_export_units_to_ris_emits_nested_identifier_doi():
    text = export_units_to_ris(
        [
            unit(
                "unit-a",
                metadata={
                    "identifier": {"doi": "doi:10.5678/nested"},
                },
            )
        ]
    )

    assert "DO  - 10.5678/nested\n" in text
