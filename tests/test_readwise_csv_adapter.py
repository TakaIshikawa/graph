from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.readwise_csv import ReadwiseCsvAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import ContentType, EdgeRelation, SourceProject


def test_readwise_csv_ingests_quoted_multiline_highlights_and_notes(tmp_path):
    export = tmp_path / "readwise.csv"
    export.write_text(
        (
            "Highlight,Book Title,Book Author,URL,Category,Tags,Note,Location,Highlighted at\n"
            '"First line\nsecond line",Deep Work,Cal Newport,https://example.com/deep,'
            'books,"focus, #productivity","Remember this\nfor planning",42,'
            "2025-01-02T10:30:00Z\n"
        ),
        encoding="utf-8",
    )

    result = ReadwiseCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.READWISE_CSV
    assert unit.source_entity_type == "highlight"
    assert unit.content_type == ContentType.INSIGHT
    assert unit.title == "Deep Work"
    assert "First line\nsecond line" in unit.content
    assert "Note: Remember this\nfor planning" in unit.content
    assert "Author: Cal Newport" in unit.content
    assert "URL: https://example.com/deep" in unit.content
    assert "Location: 42" in unit.content
    assert "Category: books" in unit.content
    assert unit.tags == ["focus", "productivity"]
    assert unit.metadata["tags"] == ["focus", "productivity"]
    assert unit.metadata["title"] == "Deep Work"
    assert unit.metadata["author"] == "Cal Newport"
    assert unit.metadata["url"] == "https://example.com/deep"
    assert unit.metadata["category"] == "books"
    assert unit.metadata["note"] == "Remember this\nfor planning"
    assert unit.metadata["highlighted_at"] == "2025-01-02T10:30:00Z"
    assert unit.created_at == datetime(2025, 1, 2, 10, 30, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2025, 1, 2, 10, 30, tzinfo=timezone.utc)


def test_readwise_csv_handles_missing_optional_columns(tmp_path):
    export = tmp_path / "minimal.csv"
    export.write_text("Highlight\nA standalone highlight\n", encoding="utf-8")

    result = ReadwiseCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "Readwise highlight"
    assert unit.content == "A standalone highlight"
    assert unit.metadata["source_file"] == "minimal.csv"
    assert unit.metadata["row_number"] == 2
    assert unit.metadata["title"] == ""
    assert unit.metadata["author"] == ""
    assert unit.metadata["tags"] == []


def test_readwise_csv_parses_tags_from_multiple_delimiters(tmp_path):
    export = tmp_path / "tags.csv"
    export.write_text(
        "Highlight,Tags\nTagged highlight,\"#alpha; beta|gamma, alpha\"\n",
        encoding="utf-8",
    )

    unit = ReadwiseCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.tags == ["alpha", "beta", "gamma"]
    assert unit.metadata["tags"] == ["alpha", "beta", "gamma"]
    assert "Tags: alpha, beta, gamma" in unit.content


def test_readwise_csv_uses_author_title_metadata(tmp_path):
    export = tmp_path / "book.csv"
    export.write_text(
        "Highlight,Book Title,Book Author\nA useful passage,The Book,The Author\n",
        encoding="utf-8",
    )

    unit = ReadwiseCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.title == "The Book"
    assert unit.metadata["title"] == "The Book"
    assert unit.metadata["author"] == "The Author"
    assert "Title: The Book" in unit.content
    assert "Author: The Author" in unit.content


def test_readwise_csv_source_ids_are_deterministic(tmp_path):
    export = tmp_path / "stable.csv"
    export.write_text(
        "Highlight,Book Title,Book Author,Location,Highlighted at\n"
        "Stable passage,Stable Book,Stable Author,7,2025-01-01T00:00:00Z\n",
        encoding="utf-8",
    )

    first = ReadwiseCsvAdapter(path=str(export)).ingest().units[0]
    second = ReadwiseCsvAdapter(path=str(export)).ingest().units[0]

    assert first.source_id == second.source_id
    assert first.source_id.startswith("readwise_csv:")


def test_readwise_csv_uses_exported_highlight_id_when_present(tmp_path):
    export = tmp_path / "id.csv"
    export.write_text(
        "Highlight ID,Highlight,Book Title\nrw-1,Highlight with id,Book\n",
        encoding="utf-8",
    )

    unit = ReadwiseCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.source_id == "readwise_csv:rw-1"


def test_readwise_csv_adapter_is_registered():
    assert "readwise_csv" in list_adapters()
    adapter = get_adapter("readwise_csv", path="/tmp/readwise.csv")
    assert isinstance(adapter, ReadwiseCsvAdapter)
    assert adapter.name == "readwise_csv"


def test_readwise_csv_groups_multiple_highlights_under_document(tmp_path):
    export = tmp_path / "readwise.csv"
    export.write_text(
        "Highlight,Book Title,Book Author,URL,Category,Highlighted at\n"
        "One,Deep Work,Cal Newport,https://example.com/deep,books,2025-01-01T00:00:00Z\n"
        "Two,Deep Work,Cal Newport,https://example.com/deep,books,2025-01-02T00:00:00Z\n",
        encoding="utf-8",
    )

    result = ReadwiseCsvAdapter(path=str(export)).ingest(entity_types=["document", "highlight"])

    documents = [unit for unit in result.units if unit.source_entity_type == "document"]
    highlights = [unit for unit in result.units if unit.source_entity_type == "highlight"]
    assert len(documents) == 1
    assert len(highlights) == 2
    document = documents[0]
    assert document.title == "Deep Work"
    assert document.metadata["author"] == "Cal Newport"
    assert document.metadata["url"] == "https://example.com/deep"
    assert document.metadata["category"] == "books"
    assert document.metadata["source_files"] == ["readwise.csv"]
    assert document.metadata["highlight_count"] == 2
    assert {(edge.from_unit_id, edge.to_unit_id) for edge in result.edges} == {
        (document.source_id, highlight.source_id) for highlight in highlights
    }
    assert {edge.relation for edge in result.edges} == {EdgeRelation.CONTAINS}


def test_readwise_csv_document_missing_title_falls_back_to_url(tmp_path):
    export = tmp_path / "readwise.csv"
    export.write_text(
        "Highlight,URL,Category\nA note,https://example.com/article,articles\n",
        encoding="utf-8",
    )

    document = ReadwiseCsvAdapter(path=str(export)).ingest(entity_types=["document"]).units[0]

    assert document.title == "https://example.com/article"
    assert document.metadata["title"] == ""
    assert document.metadata["highlight_count"] == 1


def test_readwise_csv_document_identity_prefers_url(tmp_path):
    export = tmp_path / "readwise.csv"
    export.write_text(
        "Highlight,Book Title,Book Author,URL,Category\n"
        "One,First Title,Ada,https://example.com/same,article\n"
        "Two,Second Title,Grace,https://example.com/same,article\n",
        encoding="utf-8",
    )

    documents = ReadwiseCsvAdapter(path=str(export)).ingest(entity_types=["document"]).units

    assert len(documents) == 1
    assert documents[0].metadata["highlight_count"] == 2


def test_readwise_csv_document_filtering_preserves_highlight_default(tmp_path):
    export = tmp_path / "readwise.csv"
    export.write_text(
        "Highlight,Book Title\nOne,Book\n",
        encoding="utf-8",
    )

    default_result = ReadwiseCsvAdapter(path=str(export)).ingest()
    document_only = ReadwiseCsvAdapter(path=str(export)).ingest(entity_types=["document"])

    assert [unit.source_entity_type for unit in default_result.units] == ["highlight"]
    assert default_result.edges == []
    assert [unit.source_entity_type for unit in document_only.units] == ["document"]
    assert document_only.edges == []
