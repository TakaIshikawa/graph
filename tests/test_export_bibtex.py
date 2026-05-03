from __future__ import annotations

from datetime import datetime, timezone

from graph.export import export_units_to_bibtex
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


def test_export_units_to_bibtex_creates_article_entry():
    """BibTeX export creates @article entries for journal papers."""
    text = export_units_to_bibtex(
        [
            unit(
                "unit-a",
                title="Graph Export Formats",
                content="Study content.",
                metadata={
                    "authors": ["Smith, Ada", "Doe, Grace"],
                    "year": 2025,
                    "journal": "Knowledge Engineering Review",
                    "volume": "12",
                    "number": "3",
                    "pages": "45--67",
                    "url": "https://example.test/paper",
                    "abstract": "A study of graph export formats.",
                },
            )
        ]
    )

    assert text == (
        "@article{Smith2025Graph,\n"
        "  title = {Graph Export Formats},\n"
        "  author = {Smith, Ada and Doe, Grace},\n"
        "  year = {2025},\n"
        "  journal = {Knowledge Engineering Review},\n"
        "  volume = {12},\n"
        "  number = {3},\n"
        "  pages = {45--67},\n"
        "  url = {https://example.test/paper},\n"
        "  abstract = {A study of graph export formats.}\n"
        "}\n"
    )


def test_export_units_to_bibtex_creates_book_entry():
    """BibTeX export creates @book entries."""
    text = export_units_to_bibtex(
        [
            unit(
                "unit-b",
                title="Knowledge Graphs: Theory and Practice",
                metadata={
                    "authors": ["Johnson, Bob"],
                    "year": 2024,
                    "publisher": "Academic Press",
                    "edition": "2nd",
                    "isbn": "978-0-123456-78-9",
                },
            )
        ]
    )

    assert text == (
        "@book{Johnson2024Knowledge,\n"
        "  title = {Knowledge Graphs: Theory and Practice},\n"
        "  author = {Johnson, Bob},\n"
        "  year = {2024},\n"
        "  publisher = {Academic Press},\n"
        "  edition = {2nd},\n"
        "  isbn = {978-0-123456-78-9}\n"
        "}\n"
    )


def test_export_units_to_bibtex_creates_online_entry():
    """BibTeX export creates @online entries for web resources."""
    text = export_units_to_bibtex(
        [
            unit(
                "unit-c",
                title="GraphML Documentation",
                metadata={
                    "authors": ["GraphML Team"],
                    "year": 2023,
                    "url": "https://graphml.graphdrawing.org/",
                },
            )
        ]
    )

    assert text == (
        "@online{GraphML2023GraphML,\n"
        "  title = {GraphML Documentation},\n"
        "  author = {GraphML Team},\n"
        "  year = {2023},\n"
        "  url = {https://graphml.graphdrawing.org/}\n"
        "}\n"
    )


def test_export_units_to_bibtex_creates_misc_entry_with_minimal_metadata():
    """BibTeX export creates @misc entries when type cannot be determined."""
    text = export_units_to_bibtex(
        [
            unit(
                "unit-d",
                title="Random Note",
                metadata={
                    "year": 2025,
                },
            )
        ]
    )

    assert text == (
        "@misc{2025Random,\n" "  title = {Random Note},\n" "  year = {2025}\n" "}\n"
    )


def test_export_units_to_bibtex_handles_doi():
    """BibTeX export recognizes DOI and uses doi field instead of url."""
    text = export_units_to_bibtex(
        [
            unit(
                "unit-e",
                title="Paper with DOI",
                metadata={
                    "authors": ["Author, Test"],
                    "year": 2025,
                    "journal": "Test Journal",
                    "url": "10.1234/test.doi",
                },
            )
        ]
    )

    assert "doi = {10.1234/test.doi}" in text
    assert "url =" not in text


def test_export_units_to_bibtex_escapes_special_characters():
    """BibTeX export escapes special LaTeX characters."""
    text = export_units_to_bibtex(
        [
            unit(
                "unit-f",
                title="Research on $ & % # _ ~ ^ symbols",
                metadata={
                    "authors": ["Test & Author"],
                    "year": 2025,
                },
            )
        ]
    )

    assert r"\$" in text
    assert r"\&" in text
    assert r"\%" in text
    assert r"\#" in text
    assert r"\_" in text
    assert r"\textasciitilde{}" in text
    assert r"\textasciicircum{}" in text


def test_export_units_to_bibtex_generates_unique_cite_keys():
    """BibTeX export generates unique citation keys."""
    text = export_units_to_bibtex(
        [
            unit(
                "unit-1",
                title="First Paper",
                metadata={
                    "authors": ["Smith, John"],
                    "year": 2025,
                },
            ),
            unit(
                "unit-2",
                title="Second Paper",
                metadata={
                    "authors": ["Smith, John"],
                    "year": 2025,
                },
            ),
        ]
    )

    # Both should have different cite keys (different title words)
    assert "Smith2025First" in text
    assert "Smith2025Second" in text


def test_export_units_to_bibtex_handles_various_author_formats():
    """BibTeX export handles different author metadata formats."""
    text = export_units_to_bibtex(
        [
            unit(
                "unit-author-string",
                title="Single Author String",
                metadata={
                    "author": "Smith, Ada",
                    "year": 2025,
                },
            ),
            unit(
                "unit-author-list",
                title="Author List",
                metadata={
                    "authors": ["Doe, Bob", "Lee, Carol"],
                    "year": 2025,
                },
            ),
            unit(
                "unit-author-dict",
                title="Author Dict",
                metadata={
                    "authors": [{"name": "Johnson, Dave"}, {"name": "Kim, Eve"}],
                    "year": 2025,
                },
            ),
        ]
    )

    assert "author = {Smith, Ada}" in text
    assert "author = {Doe, Bob and Lee, Carol}" in text
    assert "author = {Johnson, Dave and Kim, Eve}" in text


def test_export_units_to_bibtex_handles_various_date_formats():
    """BibTeX export extracts year from various date formats."""
    text = export_units_to_bibtex(
        [
            unit(
                "unit-year-int",
                title="Year as Int",
                metadata={"year": 2025},
            ),
            unit(
                "unit-year-str",
                title="Year as String",
                metadata={"publication_year": "2024"},
            ),
            unit(
                "unit-date-iso",
                title="Full Date",
                metadata={"published_at": "2023-04-15T10:00:00Z"},
            ),
            unit(
                "unit-date-obj",
                title="Date Object",
                metadata={"date": datetime(2022, 6, 1, tzinfo=timezone.utc)},
            ),
        ]
    )

    assert "@misc{2025Year," in text
    assert "year = {2025}" in text

    assert "@misc{2024Year," in text
    assert "year = {2024}" in text

    assert "@misc{2023Full," in text
    assert "year = {2023}" in text

    assert "@misc{2022Date," in text
    assert "year = {2022}" in text


def test_export_units_to_bibtex_returns_empty_for_units_without_title():
    """BibTeX export skips units without a title."""
    # The unit() helper always provides a default title, so we need to override it
    empty_unit = KnowledgeUnit(
        id="unit-empty",
        source_project=SourceProject.BIBTEX,
        source_id="source-empty",
        source_entity_type="test",
        title="",  # Empty title
        content="Some content",
        content_type=ContentType.FINDING,
        tags=[],
        created_at=UNIT_TIME,
        ingested_at=UNIT_TIME,
        updated_at=UNIT_TIME,
        metadata={},
    )
    text = export_units_to_bibtex([empty_unit])

    assert text == ""


def test_export_units_to_bibtex_handles_multiple_entries():
    """BibTeX export handles multiple units with proper separation."""
    text = export_units_to_bibtex(
        [
            unit(
                "unit-1",
                title="First Paper",
                metadata={
                    "authors": ["Author One"],
                    "year": 2025,
                },
            ),
            unit(
                "unit-2",
                title="Second Paper",
                metadata={
                    "authors": ["Author Two"],
                    "year": 2024,
                },
            ),
        ]
    )

    # Should have two entries separated by blank line
    entries = text.strip().split("\n\n")
    assert len(entries) == 2
    assert entries[0].startswith("@misc{")
    assert entries[1].startswith("@misc{")


def test_export_units_to_bibtex_uses_title_from_metadata_over_unit_title():
    """BibTeX export prefers title from metadata over unit.title."""
    text = export_units_to_bibtex(
        [
            unit(
                "unit-title",
                title="Unit Title",
                metadata={
                    "title": "Metadata Title",
                    "year": 2025,
                },
            )
        ]
    )

    assert "title = {Metadata Title}" in text
    assert "Unit Title" not in text


def test_export_units_to_bibtex_includes_note_field():
    """BibTeX export includes note field from metadata."""
    text = export_units_to_bibtex(
        [
            unit(
                "unit-note",
                title="Paper with Note",
                metadata={
                    "year": 2025,
                    "note": "Preprint version",
                },
            )
        ]
    )

    assert "note = {Preprint version}" in text


def test_export_units_to_bibtex_is_deterministic():
    """BibTeX export produces consistent output for same input."""
    units = [
        unit(
            "unit-a",
            title="Alpha",
            metadata={"year": 2025},
        ),
        unit(
            "unit-b",
            title="Beta",
            metadata={"year": 2024},
        ),
    ]

    text1 = export_units_to_bibtex(units)
    text2 = export_units_to_bibtex(units)

    assert text1 == text2
