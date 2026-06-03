from __future__ import annotations

from graph.adapters.kindle_notebook_html import KindleNotebookHtmlAdapter


def test_kindle_notebook_html_ingests_highlight_only(tmp_path):
    export = tmp_path / "notebook.html"
    export.write_text(
        """
        <div class="bookTitle">Designing Data-Intensive Applications</div>
        <div class="authors">by Martin Kleppmann</div>
        <div class="sectionHeading">Highlight(<span>yellow</span>) - Page 42 | Added on Monday, January 1, 2024 12:00:00 PM</div>
        <div class="highlight">Distributed systems are hard.</div>
        """,
        encoding="utf-8",
    )

    unit = KindleNotebookHtmlAdapter(path=str(export)).ingest().units[0]

    assert unit.source_entity_type == "highlight"
    assert unit.metadata["book_title"] == "Designing Data-Intensive Applications"
    assert unit.metadata["author"] == "Martin Kleppmann"
    assert unit.metadata["page"] == "42"
    assert unit.metadata["highlight_text"] == "Distributed systems are hard."
    assert unit.metadata["highlight_date"] == "2024-01-01T12:00:00+00:00"


def test_kindle_notebook_html_ingests_note_only_with_missing_optional_fields(tmp_path):
    export = tmp_path / "notebook.html"
    export.write_text(
        """
        <div class="bookTitle">Sparse Book</div>
        <div class="authors">Unknown Author</div>
        <div class="sectionHeading">Note</div>
        <div class="noteText">Follow up on this idea.</div>
        """,
        encoding="utf-8",
    )

    unit = KindleNotebookHtmlAdapter(path=str(export)).ingest().units[0]

    assert unit.source_entity_type == "note"
    assert unit.metadata["book_title"] == "Sparse Book"
    assert unit.metadata["note_text"] == "Follow up on this idea."
    assert "page" not in unit.metadata
    assert "location" not in unit.metadata


def test_kindle_notebook_html_emits_separate_units_for_mixed_highlight_and_note(tmp_path):
    export = tmp_path / "notebook.html"
    export.write_text(
        """
        <div class="bookTitle">Mixed Book</div>
        <div class="authors">Ada Writer</div>
        <div class="sectionHeading">Highlight - Location 123 | Added on 2024-02-03</div>
        <div class="highlight">Important sentence.</div>
        <div class="noteText">My marginal note.</div>
        """,
        encoding="utf-8",
    )

    first = KindleNotebookHtmlAdapter(path=str(export)).ingest()
    second = KindleNotebookHtmlAdapter(path=str(export)).ingest()

    assert [unit.source_entity_type for unit in first.units] == ["highlight", "note"]
    assert first.units[0].metadata["location"] == "123"
    assert first.units[1].metadata["note_text"] == "My marginal note."
    assert [unit.source_id for unit in first.units] == [unit.source_id for unit in second.units]
